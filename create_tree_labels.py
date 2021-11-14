"""pre-parse tree structures from paragraphs
"""
import os
import pickle

import h5py
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utils.clustering import *


def get_sentences(encoded_paragraph, length, idx2word):

    sentences = list()

    for sent_i, sent in enumerate(encoded_paragraph.tolist()):

        if length[sent_i] <= 2:  # '<bos>' and '<eos>' are both included in data
            break

        sentences.append(' '.join(idx2word[w] for c, w in enumerate(sent)
                         if 0 < c < length[sent_i]-1))

    return sentences


def create_tree_labels(encoded_paragraphs_path, word2idx_file_path, tree_labels_path, pad=-1, verbose=True):

    with open(word2idx_file_path, 'rb') as f:
        word2idx = pickle.load(f)
    idx2word = {i: w for w, i in word2idx.items()}

    all_sentences = list()
    with h5py.File(encoded_paragraphs_path, 'r') as h:
        s_max = len(h['length'][0])
        if verbose:
            print('[INFO]: preparing sentences...')
        for gid in tqdm(range(len(h['encoded_paragraph'])), disable=not verbose):
            all_sentences.append(get_sentences(h['encoded_paragraph'][gid], h['length'][gid], idx2word))

    encode_model = SentenceTransformer('bert-base-nli-mean-tokens')

    label_data = list()

    if verbose:
        print('[INFO]: start clustering...')
    for sentences in tqdm(all_sentences, disable=not verbose):

        label = np.ones([2*s_max-1,], dtype=np.long) * pad

        if len(sentences) == 1:
            label[0] = 0

            label_data.append({
                'sentences': sentences,
                'cluster_results': [],
                'label': label
            })
        else:
            sentence_embeddings = encode_model.encode(sentences)
            sent_matrix = torch.stack([torch.tensor(sent_emd) for sent_emd in sentence_embeddings])

            cluster_results = hierarchical_cluster_neighbours(sent_matrix, cosine_distance)
            label_list = cluster_results_to_labels(cluster_results)

            label[:len(label_list)] = np.array(label_list, dtype=np.long)

            label_data.append({
                'sentences': sentences,
                'cluster_results': cluster_results,
                'label': label
            })

    with open(tree_labels_path, 'wb') as f:
        pickle.dump(label_data, f)


def tree_labels_to_scores(tree_labels_path, tree_scores_path, pad=-1., c=0.2, verbose=True):

    with open(tree_labels_path, 'rb') as f:
        tree_labels = pickle.load(f)

    score_data = list()
    for label in tqdm(tree_labels, disable=not verbose):

        score = np.ones([2 * s_max - 1, ], dtype=np.float32) * pad

        if len(label['cluster_results']) == 0:
            score[0] = 0.

            score_data.append({
                'sentences': label['sentences'],
                'cluster_results': [],
                'score': score
            })
        else:
            score_list = cluster_results_to_scores(label['cluster_results'], c)
            score[:len(score_list)] = np.array(score_list, dtype=np.float32)

            score_data.append({
                'sentences': label['sentences'],
                'cluster_results': label['cluster_results'],
                'score': score
            })

    with open(tree_scores_path, 'wb') as f:
        pickle.dump(score_data, f)


if __name__ == '__main__':

    s_max = 6
    w_max = 33

    s_min = 3
    w_min = 2

    # epd = './data/cleaned/encoded_paragraphs_s_{}_{}_w_{}_{}.h5'.format(s_min, s_max, w_min, w_max)
    # vfp = './data/cleaned/word2idx_s_min_{}_w_min_{}.pkl'.format(s_min, w_min)
    # tlp = './data/cleaned/tree_labels_stack_s_{}_{}_w_{}_{}.pkl'.format(s_min, s_max, w_min, w_max)
    # pad = -1
    #
    # create_tree_labels(epd, vfp, tlp, pad, verbose=True)

    tlp = './data/cleaned/tree_labels_stack_s_{}_{}_w_{}_{}.pkl'.format(s_min, s_max, w_min, w_max)
    tsp = './data/cleaned/tree_scores_stack_s_{}_{}_w_{}_{}.pkl'.format(s_min, s_max, w_min, w_max)
    pad = -1
    c = 0.2

    tree_labels_to_scores(tlp, tsp, pad, verbose=True)
