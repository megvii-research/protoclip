
import torch
import torch.nn as nn

import numpy as np
from sklearn.linear_model import LogisticRegression as sklearnLogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from tqdm import tqdm
import logging

def logistic_regression_pytorch(train_features, train_labels, test_features, test_labels):
    
    class AverageMeter(object):
        """computes and stores the average and current value"""

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

    class TensorDataset():
        def __init__(self, *tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return tuple(tensor[index] for tensor in self.tensors)

        def __len__(self):
            return self.tensors[0].size(0)
        
    class Classifier(nn.Module):
        def __init__(self, feature_dim, num_labels):
            super(Classifier, self).__init__()

            self.linear = nn.Linear(feature_dim, num_labels)
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.linear(x)

    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    train_dataset = TensorDataset(torch.Tensor(train_features), torch.Tensor(train_labels).long())
    val_dataset = TensorDataset(torch.Tensor(test_features), torch.Tensor(test_labels).long())
    train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=5000, num_workers=8, pin_memory=True, persistent_workers=True)
    
    num_labels = int(max(train_labels)+1)
    classifier = Classifier(train_features.shape[1], num_labels).cuda()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0)
    
    criterion = nn.CrossEntropyLoss().cuda()
    best_acc = 0
    for epoch in (pbar := tqdm(range(100))):
        top1_train = AverageMeter()
        top5_train = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        for step, (feature, label) in enumerate(train_loader):
            feature = feature.cuda()
            label = label.cuda()
            output = classifier(feature)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1, acc5 = accuracy(output, label, topk=(1, 5))
            losses.update(loss.item(), feature.size(0))
            top1_train.update(acc1[0], feature.size(0))
            top5_train.update(acc5[0], feature.size(0))
        
        for step, (feature, label) in enumerate(val_loader):
            feature = feature.cuda()
            label = label.cuda()
            with torch.no_grad():
                output = classifier(feature)
            acc1, acc5 = accuracy(output, label, topk=(1, 5))
            top1.update(acc1[0], feature.size(0))
            top5.update(acc5[0], feature.size(0))

        scheduler.step()
        
        if top1.avg.item() > best_acc:
            best_acc = top1.avg.item()
        pbar.set_description(f'Epoch {epoch+1}, test accuracy {top1.avg.item():.2f}, best accuracy {best_acc:.2f}')

    return best_acc
        

def get_features(model, dataset, args):
   
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)):
            images = images.to(args.device)

            if args.distributed and not args.horovod:
                image_features = model.module.encode_image(images)
            else:
                image_features = model.encode_image(images)

            all_features.append(image_features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def get_linear_eval_acc(model, dataset_name, root, preprocess, args):

    if dataset_name=='cifar10':
        train = CIFAR10(root, download=True, train=True, transform=preprocess)
        test = CIFAR10(root, download=True, train=False, transform=preprocess)

    elif dataset_name=='cifar100':
        train = CIFAR100(root, download=True, train=True, transform=preprocess)
        test = CIFAR100(root, download=True, train=False, transform=preprocess)

    elif dataset_name=='stl10':
        train = STL10(root, download=True, split='train', transform=preprocess)
        test = STL10(root, download=True, split='test', transform=preprocess)
    else: 
        train = ImageFolder(f'{args.eval_data_dir}/{dataset_name}/train', transform=preprocess)
        test = ImageFolder(f'{args.eval_data_dir}/{dataset_name}/test', transform=preprocess)

        
    # Calculate the image features
    logging.info(f'extracting featres from {dataset_name} training set...')
    train_features, train_labels = get_features(model, train, args=args)
    logging.info(f'extracting featres from {dataset_name} testing set...')
    test_features, test_labels = get_features(model, test, args=args)

    if args.linear_prob_mode=='sklearn':
        logging.info('Runing sklearn-based logistic regression')
        classifier = sklearnLogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=32)
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(test_features)
        accuracy = 100 * np.mean((test_labels == predictions).astype(np.float)) 
    
    elif args.linear_prob_mode=='pytorch':
        logging.info('Runing pytorch-based logistic regression')
        accuracy = logistic_regression_pytorch(train_features, train_labels, test_features, test_labels)

    return  float(accuracy)


def linear_eval(model, dataset_names, epoch, preprocess, args):
      
    if args.linear_frequency == 0:
        return {}
    if (epoch % args.linear_frequency) != 0 and epoch != args.epochs:
        return {}

    results = {}
    for dataset_name in dataset_names:
        logging.info(f'starting linear evaluation on {dataset_name}...')
        accuracy = get_linear_eval_acc(model, dataset_name, args.eval_data_dir, preprocess, args)
        results[f'{dataset_name}-linear-eval-acc'] = accuracy
        logging.info(f'Finished linear evaluation on  {dataset_name}. accuracy: {accuracy}')

    return results


if __name__=='__main__':
    import open_clip
    import os
    import pickle as pkl
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='OpenAI')
    train_data, val_data = pkl.load(open(os.path.join('train.pkl'),'rb')), pkl.load(open(os.path.join('test.pkl'),'rb'))
    train_features, train_labels = train_data['features'], train_data['labels']
    test_features, test_labels = val_data['features'], val_data['labels']
    logistic_regression_pytorch(train_features, train_labels, test_features, test_labels)
