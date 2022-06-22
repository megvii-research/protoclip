import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def analyze_features(all_image_features, all_text_features, args):
    if all_image_features is None or all_text_features is None:
        return {}

    image_avg_sim = get_self_cosine_similarity(all_image_features)
    text_avg_sim = get_self_cosine_similarity(all_text_features)
    modality_gap = get_modality_gap(all_image_features, all_text_features)

    results = {
        'image_avg_self_similarity':image_avg_sim,
        'text_avg_self_similarity':text_avg_sim,
        'modality_gap':modality_gap,
        'image_feature_std': float(torch.std(all_image_features, dim=0).mean().item()),
        'text_feature_std': float(torch.std(all_text_features, dim=0).mean().item()),
    }

    return results


def get_self_cosine_similarity(features):
    # reimplement Figure 2(a) in https://arxiv.org/abs/2203.02053v1 
    features = features.numpy()
    similarities = cosine_similarity(features, features).flatten()
    similarities = similarities[similarities>0]
    
    return float(np.average(similarities))


def get_modality_gap(all_image_features, all_text_features):
    # reimplement the "modaility gap" in Section 4.2 of https://arxiv.org/abs/2203.02053v1 
    mean_image_feature = torch.mean(all_image_features, dim=0)
    mean_text_feature = torch.mean(all_text_features, dim=0)
    delta_gap = mean_image_feature - mean_text_feature

    return float(delta_gap.norm().item())
