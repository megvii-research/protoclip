import json
import logging
import math
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

#from open_clip import ClipLoss
from training.loss import gather_features, ClipLoss, ProtoLoss
from .distributed import is_master, get_gathered_item
import torch.distributed as dist
from training.evaluations.analyze_features import get_modality_gap

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, clustering, args, tb_writer=None):
    device = torch.device(args.device)
    ZERO = torch.zeros(1).to(args.device)
    
    #autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    autocast = torch.cuda.amp.autocast

    model.train()
    clip_loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    proto_loss = ProtoLoss()
 
    w_clip = args.w_clip
    w_proto = args.w_proto
    w_proto_external = args.w_proto_external

    # optionally entering and LiT epoch
    if args.lit_start_epoch > 0 and epoch >= args.lit_start_epoch - 1:
        w_clip = 1
        w_proto = 0
        w_proto_external = 0
        if is_master(args):
            logging.info('Setting up Locked-image Finetuning (LiT)')

    # optionally warm-up the model with InfoNCE only following PCL
    if epoch < args.infonce_warmup_epoch:
        w_clip = 1
        w_proto = 0
        w_proto_external = 0
        if is_master(args):
            logging.info('Setting up InfoNCE-only warmup')


    clustering.img_centroids = clustering.img_centroids.cuda()
    clustering.text_centroids = clustering.text_centroids.cuda()
    clustering.external_centroids = clustering.external_centroids.cuda()
    if args.PBT:
        clustering.img_centroids_translated_from_text_prototypes = clustering.img_centroids_translated_from_text_prototypes.cuda()
        clustering.text_centroids_translated_from_image_prototypes = clustering.text_centroids_translated_from_image_prototypes.cuda()
        clustering.img_centroids_translated_from_external_prototypes = clustering.img_centroids_translated_from_external_prototypes.cuda()
        clustering.text_centroids_translated_from_external_prototypes = clustering.text_centroids_translated_from_external_prototypes.cuda()


    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        index, images, texts = batch
        if len(index)!=args.batch_size: # drop last incomplete small batch
            continue
        all_index = get_gathered_item(index.cuda(), args)
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        img_labels = clustering.img_labels[all_index].to(device=device, non_blocking=True)
        text_labels = clustering.text_labels[all_index].to(device=device, non_blocking=True)
        external_labels = clustering.external_labels[all_index].to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            # original CLIP
            if not args.add_projection_head: 
                image_features, text_features, logit_scale = model(images, texts)

                if args.distributed:
                    all_image_features, all_text_features = gather_features(image_features, text_features,
                        args.local_loss, args.gather_with_grad, args.rank, args.world_size, args.horovod)
                else:
                    all_image_features, all_text_features = image_features, text_features

                L_clip = clip_loss(all_image_features, all_text_features, logit_scale)
                total_loss = L_clip
            # ProtoCLIP
            else:
                image_features, text_features, image_features_projected, text_features_projected, logit_scale, logit_scale_proto = model(images, texts)
                
                if args.distributed:
                    all_image_features, all_text_features = gather_features(image_features, text_features,
                        args.local_loss, args.gather_with_grad, args.rank, args.world_size, args.horovod)
                    all_image_features_projected, all_text_features_projected = gather_features(image_features_projected, text_features_projected,
                        args.local_loss, args.gather_with_grad, args.rank, args.world_size, args.horovod)
                else:
                    all_image_features, all_text_features = image_features, text_features
                    all_image_features_projected, all_text_features_projected = image_features_projected, text_features_projected

                L_clip = clip_loss(all_image_features, all_text_features, logit_scale)
                
                if args.PBT:
                    img_target = clustering.img_centroids_translated_from_text_prototypes
                    text_target = clustering.text_centroids_translated_from_image_prototypes
                    img_target_external = clustering.img_centroids_translated_from_external_prototypes
                    text_target_external = clustering.text_centroids_translated_from_external_prototypes
                else:
                    img_target = clustering.text_centroids
                    text_target = clustering.img_centroids
                    img_target_external = clustering.img_centroids_translated_from_external_prototypes
                    text_target_external = clustering.text_centroids_translated_from_external_prototypes

                L_proto_img2text, acc_img2text = proto_loss(
                    student_features=all_image_features_projected, student_centroids=img_target, teacher_centroids=clustering.text_centroids, 
                    logit_scale_student=logit_scale_proto, teacher_temperature=args.target_temperature, labels=text_labels
                    )
                L_proto_text2img, acc_text2img = proto_loss(
                    student_features=all_text_features_projected, student_centroids=text_target, teacher_centroids=clustering.img_centroids, 
                    logit_scale_student=logit_scale_proto, teacher_temperature=args.target_temperature, labels=img_labels
                    )

                if args.external_teacher is not None:
                    L_proto_img2external, acc_img2external = proto_loss(
                        student_features=all_image_features_projected, student_centroids=img_target_external, teacher_centroids=clustering.external_centroids, 
                        logit_scale_student=logit_scale_proto, teacher_temperature=-1, labels=external_labels
                        )
                    L_proto_text2external, acc_text2external = proto_loss(
                        student_features=all_text_features_projected, student_centroids=text_target_external, teacher_centroids=clustering.external_centroids, 
                        logit_scale_student=logit_scale_proto, teacher_temperature=-1, labels=external_labels
                        )
                else:
                    L_proto_img2external, acc_img2external = ZERO, ZERO
                    L_proto_text2external, acc_text2external = ZERO, ZERO

                L_proto =  0.5 * (L_proto_img2text + L_proto_text2img)
                L_proto_external = 0.5 * (L_proto_img2external + L_proto_text2external)
                total_loss = w_clip * L_clip + w_proto * L_proto + w_proto_external * L_proto_external

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))
            if args.add_projection_head:
                unwrap_model(model).logit_scale_proto.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 10 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            logging.info(
                f"Train Epoch: {epoch+1}/{args.epochs} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {total_loss.item():.5f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f} "
                f"Temperature: {1 / logit_scale.item():.4f} "
                f"LR (visual/rest): {optimizer.param_groups[0]['lr']:3f}/{optimizer.param_groups[2]['lr']:3f} "
                f"grad: {norm:1f} "
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss_clip": L_clip.item(),
                "temperature":  1 / logit_scale.item(),
                "lr_visual": optimizer.param_groups[0]["lr"],
                "lr_rest": optimizer.param_groups[2]["lr"],
                "gradient-norm": norm,
                
                "feature_std_image": torch.std(image_features, dim=0).mean().item(),
                "feature_std_text": torch.std(text_features, dim=0).mean().item(),
                "feature_modality_gap": get_modality_gap(image_features, text_features),
            }
            profiling = {
                "batch data time (s)": data_time_m.val,
                "bathc total time (s)": batch_time_m.val,
            }

            log_data_protoclip = {}
            if args.add_projection_head:
                log_data_protoclip['loss_proto'] = L_proto.item()
                log_data_protoclip['loss_proto_external'] = L_proto_external.item()
                log_data_protoclip['acc_img2text'] = acc_img2text
                log_data_protoclip['acc_text2img'] = acc_text2img
                log_data_protoclip['acc_img2external'] = acc_img2external
                log_data_protoclip['acc_text2external'] = acc_text2external
                log_data_protoclip['temperature_proto'] = 1 / logit_scale_proto.item()


            for name, val in log_data.items():
                name = "training/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            for name, val in log_data_protoclip.items():
                name = "training_protoclip/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})
            
            for name, val in profiling.items():
                name = "profiling/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def feature_extraction_one_epoch(model, data, epoch, optimizer, scaler, scheduler, clustering, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.eval()

    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):

        indexs, images, texts = batch
        indexs = indexs.to(device=device, non_blocking=True)
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)

        # forward propagation
        with autocast():
            with torch.no_grad():
                image_features, text_features, image_features_projected, text_features_projected, logit_scale, logit_scale_proto = model(images, texts)
        
        # cache features
        indexs = get_gathered_item(indexs, args)
        image_features_projected = get_gathered_item(image_features_projected, args)
        text_features_projected = get_gathered_item(text_features_projected, args)
        if is_master(args):
            clustering.load_batch(indexs, image_features_projected, text_features_projected)
        
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            logging.info(
                f"Feature extraction: {epoch+1}/{args.epochs} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f} "
            )

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
