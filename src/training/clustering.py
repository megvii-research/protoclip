import logging
import torch
import numpy as np
import faiss

import os
from PIL import Image
try:
    import wandb
except ImportError:
    wandb = None
from utils.plot_pairs import plot_pairs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openTSNE import TSNE
from training.distributed import is_master
import torch.distributed as dist
from torchvision.transforms import ToPILImage

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import _pickle as pickle

class Clustering():

    def __init__(self, args):
        self.episode_size=args.episode_size
        self.feature_dim = args.projection_dim
        self.reset(args.k)
        
    def reset(self, k):
        self.img_feature = torch.zeros(size=(self.episode_size, self.feature_dim)).share_memory_()
        self.text_feature = torch.zeros(size=(self.episode_size, self.feature_dim)).share_memory_()
        
        self.img_labels = torch.zeros(self.episode_size, dtype=torch.long)
        self.text_labels = torch.zeros(self.episode_size, dtype=torch.long)
        self.external_labels = torch.zeros(self.episode_size, dtype=torch.long)

        self.img_centroids = torch.zeros([k, self.feature_dim])
        self.text_centroids = torch.zeros([k, self.feature_dim])
        self.external_centroids = torch.zeros([k, self.feature_dim])
        
        self.img_centroids_translated_from_text_prototypes = torch.zeros([k, self.feature_dim])
        self.text_centroids_translated_from_image_prototypes = torch.zeros([k, self.feature_dim])
        self.img_centroids_translated_from_external_prototypes = torch.zeros([k, self.feature_dim])
        self.text_centroids_translated_from_external_prototypes = torch.zeros([k, self.feature_dim])


    def load_batch(self, index, img_features, text_features):
        self.img_feature[index] = img_features.detach().cpu().type(torch.float32)
        self.text_feature[index] = text_features.detach().cpu().type(torch.float32)

    
    def dump(self, file, item):
        f = open(file, 'wb')
        pickle.dump(item, f, protocol=4)
        f.close() 

    def load(self, file):
        f = open(file, 'rb')
        item = pickle.load(f)
        f.close()
        return item
    
    def sync_prototypes(self, args):
        if is_master(args):
            self.dump(os.path.join(args.cache_path, f'img_labels.pkl'), self.img_labels)
            self.dump(os.path.join(args.cache_path, f'img_centroids.pkl'), self.img_centroids)
            self.dump(os.path.join(args.cache_path, f'text_labels.pkl'), self.text_labels)
            self.dump(os.path.join(args.cache_path, f'text_centroids.pkl'), self.text_centroids)
            self.dump(os.path.join(args.cache_path, f'external_labels.pkl'), self.external_labels)
            self.dump(os.path.join(args.cache_path, f'external_centroids.pkl'), self.external_centroids)
            if args.PBT:
                self.dump(os.path.join(args.cache_path, f'img_centroids_translated_from_text_prototypes.pkl'), self.img_centroids_translated_from_text_prototypes)
                self.dump(os.path.join(args.cache_path, f'text_centroids_translated_from_image_prototypes.pkl'), self.text_centroids_translated_from_image_prototypes)
                self.dump(os.path.join(args.cache_path, f'img_centroids_translated_from_external_prototypes.pkl'), self.img_centroids_translated_from_external_prototypes)
                self.dump(os.path.join(args.cache_path, f'text_centroids_translated_from_external_prototypes.pkl'), self.text_centroids_translated_from_external_prototypes)
        
        if args.distributed:
            dist.barrier()

        if not is_master(args):
            self.img_labels = self.load(os.path.join(args.cache_path, f'img_labels.pkl'))
            self.img_centroids = self.load(os.path.join(args.cache_path, f'img_centroids.pkl'))
            self.text_labels = self.load(os.path.join(args.cache_path, f'text_labels.pkl'))
            self.text_centroids = self.load(os.path.join(args.cache_path, f'text_centroids.pkl'))
            self.external_labels = self.load(os.path.join(args.cache_path, f'external_labels.pkl'))
            self.external_centroids = self.load(os.path.join(args.cache_path, f'external_centroids.pkl'))
            if args.PBT:
                self.img_centroids_translated_from_text_prototypes = self.load(os.path.join(args.cache_path, f'img_centroids_translated_from_text_prototypes.pkl'))
                self.text_centroids_translated_from_image_prototypes = self.load(os.path.join(args.cache_path, f'text_centroids_translated_from_image_prototypes.pkl'))
                self.img_centroids_translated_from_external_prototypes = self.load(os.path.join(args.cache_path, f'img_centroids_translated_from_external_prototypes.pkl'))
                self.text_centroids_translated_from_external_prototypes = self.load(os.path.join(args.cache_path, f'text_centroids_translated_from_external_prototypes.pkl'))
        
        if args.distributed:
            dist.barrier()

        if is_master(args):
            logging.info(f'Constructed prototypes are synchronized')
            for file in os.listdir(args.cache_path):
                os.remove(os.path.join(args.cache_path, file))
            logging.info(f'Cache path {args.cache_path} has been cleared')

                
    def generate_labels(self, k, args):
        # remove possible NaN
        self.img_feature = torch.where(torch.isnan(self.img_feature), torch.full_like(self.img_feature,0), self.img_feature) 
        self.text_feature = torch.where(torch.isnan(self.text_feature), torch.full_like(self.text_feature,0), self.text_feature) 
        
        self.k=k
        logging.info(f'Constructing image prototypes with K-Means')
        self.img_labels, self.img_centroids, img_error_log, self.img_distance = self.kmeans(self.img_feature, k, args)
        logging.info(f'Constructing text prototypes with K-Means')
        self.text_labels, self.text_centroids, text_error_log, self.text_distance = self.kmeans(self.text_feature, k, args)
        return img_error_log, text_error_log
                
    def generate_labels_from_external_teacher(self, external_teacher, k, args):    
        logging.info(f'Constructing external teacher prototypes with K-Means')
        external_teacher = torch.from_numpy(external_teacher.astype(np.float32)) 
        self.external_labels, self.external_centroids, external_error_log, external_distance = self.kmeans(external_teacher, k, args)
    

    def kmeans(self, feature, k, args):
        feature=feature.cpu().numpy()

        centroids = torch.zeros([k, feature.shape[1]])
        
        kmeans = faiss.Kmeans(
            d=feature.shape[1], 
            k=k, 
            niter=args.kmeans_max_iter, 
            nredo=args.kmeans_nredo,
            verbose=True, 
            gpu=True)
        kmeans.train(feature)

        # in case of derived centroid is less than args.k
        centroids[:,:kmeans.centroids.shape[1]] = torch.from_numpy(kmeans.centroids)
        distance, labels = kmeans.index.search(feature, 1)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0])

        return torch.from_numpy(labels), centroids, kmeans.iteration_stats, distance.flatten()

    def log_kmeans_error(self, iteration_stats, epoch, writer, args, name):
        for i in range(len(iteration_stats)):
            if writer is not None:
                writer.add_scalar(f'clustering/kmeans_error_log_{name}', iteration_stats[i]['obj'], epoch*args.kmeans_max_iter+i)
            if args.wandb:
                wandb.log({f'clustering/kmeans_error_log_{name}': iteration_stats[i]['obj'], 'step': epoch*args.kmeans_max_iter+i})

    
    def PBT(self, k, teacher_labels, student_features):
        # teacher_centroids: (n_class, feature_dim)
        n_sample, feature_dim = student_features.size()
        teacher_labels = teacher_labels[:n_sample]
        cluster_indexs = np.unique(teacher_labels)
        centorids = torch.zeros(k, feature_dim)
        
        for k in cluster_indexs:
            cluster_samples = np.where(teacher_labels==k)[0]
            centroid = torch.mean(student_features[cluster_samples], dim=0)
            centorids[k] = centroid
        
        return centorids

    def analyze_labels(self):
        metrics = {}
        logging.info( "Analyzing pseudo labels.")
        for modality in ['image', 'text']:
            if modality=='image':
                label = self.img_labels
                feature = self.img_feature.numpy()
            if modality=='text':
                label = self.text_labels
                feature = self.text_feature.numpy()
                
            unique_labels, n_samples = np.unique(label, return_counts=True)
            metrics[f'{modality}-n_cluster']=len(unique_labels)
            metrics[f'{modality}-Silhouette Coefficient']=silhouette_score(feature, label, sample_size=5000)
            metrics[f'{modality}-Davies-Bouldin Index']=davies_bouldin_score(feature, label)
            metrics[f'{modality}-Calinski and Harabasz score']=calinski_harabasz_score(feature, label)
        
        logging.info( "Pseudo labels metrics:\n" + f"\n".join([f"\t{k}\t{v}" for k, v in metrics.items()]))
        return metrics

    def show_tsne(self, file_name, truncate, title):

        logging.info('Fitting T-SNE')
                       
        tsne_img = TSNE(verbose=True, n_jobs=64, n_iter=1000).fit(self.img_feature[:truncate])
        tsne_text = TSNE(verbose=True, n_jobs=64, n_iter=1000).fit(self.text_feature[:truncate])
        
        plt.figure(figsize=(30,15))
        plt.rc('font', size=20) 
        plt.subplots_adjust(top=0.9,wspace=0.05,hspace=0.05)

        plt.subplot(121)
        plt.xticks([])
        plt.yticks([])
        plt.title('image features')
        plt.scatter(tsne_img[:,0], tsne_img[:,1], s=1.5, c=self.img_labels[:truncate], cmap='tab10', alpha=0.8)

        plt.subplot(122)
        plt.xticks([])
        plt.yticks([])
        plt.title('text features')
        plt.scatter(tsne_text[:,0], tsne_text[:,1], s=1.5, c=self.text_labels[:truncate], cmap='tab10', alpha=0.8)
        
        plt.suptitle(title)
        plt.savefig(file_name, bbox_inches='tight')

        logging.info(f'T-SNE visuallization saved to: {file_name}')
        
    def show_samples(self, dataset, modality, file_name, sample_per_class=16, max_rows=16):
        images = []
        texts = []
        if modality=='image':
            label = self.img_labels
        elif modality=='text':
            label = self.text_labels
        
        logging.info(f'Visuallizing {modality} clustering results')
        unique_labels, n_samples = np.unique(label, return_counts=True)

        for k in unique_labels[:max_rows]:
            cluster_dataset_index = np.squeeze(np.argwhere(label==k))
            if cluster_dataset_index.shape==():
                continue # empty cluster
            # show [sample_per_class] samples for each class
            for i in range(sample_per_class): 
                # sometimes there are not much sample in this cluster
                if i >= len(cluster_dataset_index):
                    images.append(Image.new('RGB', (256,256), (255,255,255)))
                    texts.append(' ')
                else:
                    image, text = dataset.get_data(int(cluster_dataset_index[i]))
                    image = ToPILImage()(image)
                    images.append(image)
                    texts.append(text)

        plot_pairs(
            images[:100*sample_per_class], texts[:100*sample_per_class], 
            suptitle=file_name, file_name=file_name+'.png', 
            sample_per_row=sample_per_class
        )  
        logging.info(f'Sample visuallization saved to: {file_name}')


    
    