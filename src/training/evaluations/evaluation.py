import logging
import os
import json
from training.distributed import is_master
from .linear_eval import linear_eval
from .zero_shot import zero_shot_eval
from .coco_retrieval import coco_retrieval_evaluation
from .analyze_features import analyze_features

try:
    import wandb
except ImportError:
    wandb = None

def evaluate(model, epoch, preprocess, tokenizer, args, tb_writer=None):
    if not is_master(args):
        return
    logging.info( f"Starting evaluation of [{args.name}] at epoch {epoch}")

    linear_eval_datasets = ['imagenet', 'cifar10', 'cifar100', 'stl10']
    zeroshot_datasets = ['imagenet', 'cifar10', 'cifar100', 'stl10', 'birdsnap','country211', 'flowers102', 'gtsrb', 'ucf101','stanford_cars']
    
    model.eval()
    all_metrics = {}
    
    # zeroshot classification
    metrics = {}
    for zeroshot_dataset in zeroshot_datasets:
        zeroshot_metrics = zero_shot_eval(model, zeroshot_dataset, epoch, preprocess, tokenizer, args)
        metrics.update(zeroshot_metrics)
        all_metrics.update(zeroshot_metrics)
    for name, val in metrics.items():
        if tb_writer is not None:
            tb_writer.add_scalar(f"eval_zero_shot/{name}", val, epoch)
        if args.wandb:
            wandb.log({f"eval_zero_shot/{name}": val, 'epoch': epoch})
    
    # MS-COCO retrieval
    metrics = {}
    retrieval_metrics, all_image_features, all_text_features= coco_retrieval_evaluation(model, epoch, preprocess, tokenizer, args)
    metrics.update(retrieval_metrics)
    all_metrics.update(retrieval_metrics)
    for name, val in metrics.items():
        if tb_writer is not None:
            tb_writer.add_scalar(f"eval_retrieval/{name}", val, epoch)
        if args.wandb:
            wandb.log({f"eval_retrieval/{name}": val, 'epoch': epoch})
    
    # Analyse COCO features
    feature_metrics = analyze_features(all_image_features, all_text_features, args)
    all_metrics.update(feature_metrics)
    for name, val in feature_metrics.items():
        if tb_writer is not None:
            tb_writer.add_scalar(f"eval_analyze_features/{name}", val, epoch)
        if args.wandb:
            wandb.log({f"eval_analyze_features/{name}": val, 'epoch': epoch})

    # linear evaluation
    metrics = {}
    if linear_eval_datasets:
        linear_metrics = linear_eval(model, linear_eval_datasets, epoch, preprocess, args)    
        metrics.update(linear_metrics)
        all_metrics.update(linear_metrics)

    logging.info( f"Finished evaluation of [{args.name}] at epoch {epoch}\n" + "\n".join([f"\t{k}\t{v}" for k, v in all_metrics.items()]))

    for name, val in metrics.items():
        if tb_writer is not None:
            tb_writer.add_scalar(f"eval_linear_prob/{name}", val, epoch)
        if args.wandb:
            wandb.log({f"eval_linear_prob/{name}": val, 'epoch': epoch})
                
    if args.save_logs:
        with open(os.path.join(args.logs, args.name, "results.jsonl"), "a+") as f:
            f.write(json.dumps(all_metrics))
            f.write("\n")

    return all_metrics
