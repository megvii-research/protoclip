import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import tqdm
import logging
import faiss


class TextDataset():
    def __init__(self, input_filename, tokenizer, caption_key='title', sep="\t") -> None:
        
        df = pd.read_csv(input_filename, sep=sep)
        print(df)
        self.captions = np.array(df[caption_key].tolist())
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        tokens_full = torch.zeros(77)
        token = self.tokenizer(str(self.captions[idx]))
        tokens_full[:len(token)]=token[:77]
            
        return idx, tokens_full.long()


def PCA(dim, feature):
    feature = feature.astype(np.float32)
    pca = faiss.PCAMatrix(feature.shape[1], dim)
    pca.train(feature)
    PCAed_feature = pca.apply_py(feature)

    return PCAed_feature

if __name__ == '__main__':
    csv = input('Input your csv file: ')
    feature_file = input('Input your feature file: ')

    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large').cuda().eval()
    dataset = TextDataset(
            input_filename=csv,
            caption_key='title',
            tokenizer = roberta.encode
        )

    dataloader = DataLoader(dataset, batch_size=256, num_workers=16, drop_last=False)
    all_text_features = np.zeros([len(dataset), 1024])

    
    logging.info(f'Start RoBERTa feature extraction for file "{csv}" (total {len(dataset)} samples).')
    for step, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        idx, texts = batch
        idx = idx.numpy()
        texts = texts.cuda()

        with torch.no_grad():
            text_feature = roberta.extract_features(texts)
        
        text_feature = text_feature.mean(dim=1)
        text_feature = F.normalize(text_feature, dim=-1)
        all_text_features[idx] = text_feature.detach().cpu().numpy()

    logging.info(f'Performing PCA to reduce feature deminsion from {all_text_features.shape[1]} to 64')
    PCAed_text_features = PCA(64, all_text_features)
    logging.info(f'Saving PCA-ed RoBERTa features {PCAed_text_features.shape} to: "{feature_file}".')
    np.save(feature_file, PCAed_text_features)

