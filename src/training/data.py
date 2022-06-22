
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Normalize
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


from open_clip import tokenize
def clip_tokenizer(str):
    return tokenize([str])[0]

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", dataset_size=None, index_mapping=None, tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')

        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.inversed_normalize = Compose([
            Normalize((0.0, 0.0, 0.0), (1/0.26862954, 1/0.26130258, 1/0.27577711)),
            Normalize((-0.48145466, -0.4578275, -0.40821073), (1.0, 1.0, 1.0)),
            ])

        # Faster data loading. see https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.images = np.array(df[img_key].tolist()).astype(np.string_)
        self.captions = np.array(df[caption_key].tolist())
        for i in range(len(self.captions)):
            self.captions[i] = self.captions[i].encode('ascii',errors='ignore')
        self.captions = self.captions.astype(np.string_)

        # use a subset of given dataset
        if dataset_size is not None:
            self.images = self.images[:dataset_size]
            self.captions = self.captions[:dataset_size]
        
        if index_mapping is None:
            self.index_mapping=torch.arange(len(self.captions))
        else:
            self.index_mapping = index_mapping
        
        if tokenizer is None:
            self.tokenizer = clip_tokenizer
        else:
            # using the tokenizer of pretrained NLP model
            self.tokenizer = tokenizer
        
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, episodic_index):
        index = self.index_mapping[episodic_index]
        image = Image.open(str(self.images[index].decode('utf-8')))
        image = self.transforms(image)
        texts = self.tokenizer(str(self.captions[index].decode('utf-8')))
        return episodic_index, image, texts
    
    def get_data(self, episode_index):
        idx = self.index_mapping[episode_index]
        pic = Image.open(str(self.images[idx].decode('utf-8')))
        image = self.inversed_normalize(self.transforms(pic))
        texts = self.captions[idx].decode('utf-8')
        
        return image, texts


@dataclass
class DataInfo:
    dataset: Dataset
    dataloader: DataLoader
    sampler: DistributedSampler


def get_csv_dataset(args, preprocess_fn, is_train, index_mapping, tokenizer):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        dataset_size=args.dataset_size,
        index_mapping=index_mapping,
        tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
        persistent_workers=True
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataset, dataloader, sampler)


def get_data(args, preprocess_fns, index_mapping, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_csv_dataset(args, preprocess_train, is_train=True, index_mapping=index_mapping, tokenizer=tokenizer)

    return data
