import os
import json
import pickle
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from model.encoder import Encoder
from model.decoder import Decoder
from utils.data_loader import CaptionDataset
from utils.DataLoaderPFG import DataLoaderPFG
from torch.utils.tensorboard import SummaryWriter
from captioner import Captioner
from evaluate import quantity_evaluate
from utils.clustering import cluster_results_to_tree

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 200
USE_CONFIG_JSON = False  # whether to use the predefined configurations
USE_TB = False
TB_PATH = './runs'
CONFIG_PATH = './model_params'
MODEL_NAME = 'debug'

VISUAL_FEATURES_PATH = '/data/bu36/parabu_att'
ENCODED_PARAGRAPHS_PATH = './data/cleaned/encoded_paragraphs_s_{}_{}_w_{}_{}.h5'
MAPPING_FILE_PATH = './data/cleaned/mappings.pkl'
WORD2IDX_PATH = './data/cleaned/word2idx_s_min_{}_w_min_{}.pkl'
TREE_LABELS_PATH = './data/cleaned/tree_labels_stack_s_{}_{}_w_{}_{}.pkl'

VAL_BEAM_SIZE = 1
VAL_DECODE_TYPE = 'beam'
EARLY_STOP_THRESHOLD = 20  # stop training if metric doesn't import after EARLY_STOP_THRESHOLD epochs


def set_args():

    args = dict()

    if USE_CONFIG_JSON:
        with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'r') as f:
            args.update(json.load(f))

        word2idx = pickle.load(open(args['word2idx_path'], 'rb'))
    else:
        # Encoder settings
        args['input_size'] = 2048
        args['output_size'] = 1024
        args['f_max'] = 36  # fixed

        # Decoder settings
        args['feat_size'] = args['output_size']
        args['emb_size'] = 512
        args['wrnn_hidden_size'] = 512
        args['wrnn_num_layers'] = 2
        args['emb_dropout'] = 0.5
        args['fc_dropout'] = 0.5

        # Split settings
        args['att_type'] = 'add'  # 'add' or 'scaled_dot'
        args['split_threshold'] = 0.3

        # Input files settings
        args['s_min'] = 3
        args['s_max'] = 6
        args['w_min'] = 2
        args['w_max'] = 33

        args['visual_features_path'] = VISUAL_FEATURES_PATH
        args['encoded_paragraphs_path'] = ENCODED_PARAGRAPHS_PATH.format(args['s_min'], args['s_max'],
                                                                         args['w_min'], args['w_max'])
        args['mapping_file_path'] = MAPPING_FILE_PATH
        args['word2idx_path'] = WORD2IDX_PATH.format(args['s_min'], args['w_min'])
        args['tree_labels_path'] = TREE_LABELS_PATH.format(args['s_min'], args['s_max'], args['w_min'], args['w_max'])

        word2idx = pickle.load(open(args['word2idx_path'], 'rb'))
        args['vocab_size'] = len(word2idx)

        # Training Settings
        args['sent_weight'] = 1.
        args['word_weight'] = 1.
        args['lr'] = 5e-4
        args['batch_size'] = 16
        args['is_grad_clip'] = True
        args['grad_clip'] = 5.
        args['modified_after_epochs'] = 5
        args['modified_lr_ratio'] = 0.8

        if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
            os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
        with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
            json.dump(args, f)

    return args, word2idx


def save_model(encoder, decoder, epoch, metrics_on_val):

    state = {'config_path': os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'),
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'epoch': epoch,
             'metrics_on_val':metrics_on_val}
    filename = os.path.join('model_params', '{}.pth.tar'.format(MODEL_NAME))
    print('Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)


def compute_word_loss(caps_gt, caps_preds, ignore_index):
    """

    :param caps_gt: (tensor) (batch_size, s_max, w_max)
    :param caps_preds: (tensor) (batch_size, s_max, w_max, vocab_size)
    :param ignore_index: (long)  specifies a target value that is ignored
    :return: word loss (tensor)
    """
    caps_preds = caps_preds.view(-1, caps_preds.shape[-1])
    caps_tags = caps_gt.contiguous().view(-1)

    return F.cross_entropy(caps_preds, caps_tags, ignore_index=ignore_index)


def compute_tree_loss(scores, tree_labels, lengths, margin):
    """

    :param scores: (tensor) (batch_size, 2*max_len-1)
    :param tree_labels: (tensor) (batch_size, 2*max_len-1)
    :param lengths: (tensor) (batch_size,)
    :param margin: (float)
    :return: tree loss (tensor)
    """
    pos_score = scores.masked_select(tree_labels == 1).clamp(min=0)
    neg_score = (margin - scores.masked_select(tree_labels == 0)).clamp(min=0)

    return (pos_score.sum() + neg_score.sum()) / (pos_score.shape[0] + neg_score.shape[0])


def train(args, word2idx):

    print('Model {} start training...'.format(MODEL_NAME))

    encoder = Encoder(input_size=args['input_size'],
                      output_size=args['output_size'],
                      f_max = args['f_max'])

    decoder = Decoder(feat_size=args['feat_size'],
                      emb_size=args['emb_size'],
                      wrnn_hidden_size=args['wrnn_hidden_size'],
                      wrnn_num_layers=args['wrnn_num_layers'],
                      vocab_size=args['vocab_size'],
                      s_max=args['s_max'],
                      w_max=args['w_max']-1,  # <bos> is not the generated target of decoder
                      att_type=args['att_type'],
                      split_threshold=args['split_threshold'],
                      emb_dropout=args['emb_dropout'],
                      fc_dropout=args['fc_dropout'])

    # move model to GPU before optimizer
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, encoder.parameters())},
                                   {'params': filter(lambda p: p.requires_grad, decoder.parameters())}],
                                    lr=args['lr'])

    scheduler = StepLR(optimizer, step_size=args['modified_after_epochs'], gamma=args['modified_lr_ratio'])

    train_loader = DataLoaderPFG(CaptionDataset(args['mapping_file_path'], args['visual_features_path'],
                                                args['encoded_paragraphs_path'], args['tree_labels_path'], 'train'),
                                 batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    # use tensorboard to track the loss
    if USE_TB:
        if not os.path.exists(TB_PATH):
            os.mkdir(TB_PATH)
        writer = SummaryWriter(log_dir=os.path.join(TB_PATH, '{}_{}'.format(MODEL_NAME, str(datetime.now().time()))))
    iter_counter = 0

    best_eval_score = 0.
    epochs_since_improvement = 0

    for epoch in range(MAX_EPOCHS):

        for batch, (gids, feats, encoded_caps, cap_lens, tree_labels) in enumerate(train_loader):

            encoder.train()
            decoder.train()

            feats = feats.to(device)
            encoded_caps = encoded_caps.to(device)
            cap_lens = cap_lens.to(device)
            tree_labels = tree_labels.to(device)

            optimizer.zero_grad()

            # === forward ====
            global_feat, features = encoder(feats)
            all_predicts, tree_list, scores = decoder(global_feat, features, encoded_caps[:, :, :-1], cap_lens-1,
                                                      tree_labels)

            cont_stop_loss = compute_tree_loss(scores, tree_labels, 2 * (cap_lens > 0).sum(1) - 1,
                                               args['split_threshold']) * args['sent_weight']
            word_loss = compute_word_loss(encoded_caps[:, :, 1:], all_predicts, word2idx['<pad>']) * args['word_weight']

            # === calculate losses and summarize ====
            total_loss = cont_stop_loss + word_loss

            # record loss
            if USE_TB:
                writer.add_scalar('batch_loss/total', total_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/cont_stop', cont_stop_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/word', word_loss.item(), iter_counter)

            if iter_counter % 500 == 0:

                encoder.eval()
                decoder.eval()

                print('quick quality check at iter {}'.format(iter_counter))
                # === quick check random image===
                sample_idx = np.random.randint(feats.shape[0])

                print('\n============')
                print('>>>> gid {}'.format(gids[sample_idx]))
                cap = Captioner(encoder, decoder, word2idx, device)
                paragraph, all_cands, all_scores, tree_scores, tree = cap.describe_feat(feats[sample_idx].unsqueeze(0),
                                                                                        feat_src='densecap',
                                                                                        decode=VAL_DECODE_TYPE,
                                                                                        beam_size=VAL_BEAM_SIZE)
                sentence_tree, _ = cap.get_sentence_tree(feats[sample_idx].unsqueeze(0), decode=VAL_DECODE_TYPE,
                                                         beam_size=VAL_BEAM_SIZE)

                print('>>>> ground truth paragraph')
                for sent in encoded_caps[sample_idx].tolist():  # (1, s_max, w_max)
                    print(' '.join(cap.idx2word[idx] for idx in sent if idx != word2idx['<pad>']))
                print()

                print('>>>> candidate paragraph by {}'.format(VAL_DECODE_TYPE))
                for sent in paragraph:
                    print(sent)
                print()

                print('>>>> candidate paragraph by true input')

                for sent_i, sent in enumerate(all_predicts[sample_idx].argmax(-1).tolist()):
                    print(' '.join(cap.idx2word[w] for c, w in enumerate(sent)
                                   if w != word2idx['<pad>'] and c < cap_lens[sample_idx][sent_i]-1))
                print('============\n')

                if VAL_DECODE_TYPE == 'beam' and VAL_BEAM_SIZE > 1:
                    print('>>>> different choices in beam search')
                    cap.output_cands_with_scores(all_cands, all_scores)
                    print()

                print('>>>> tree structures', sample_idx)
                print('[ground truth structure]')
                tree_label_data = train_loader.dataset.tree_labels[gids[sample_idx]]
                for sent_i, sent in enumerate(tree_label_data['sentences']):
                    print('{}: {}'.format(sent_i, sent))
                print('label: ', tree_label_data['label'])
                cluster_results_to_tree(tree_label_data['cluster_results']).show(key=lambda n: n.data.order)

                print('[during training]')
                tree_list[sample_idx].show()

                print('\n[during evaluating]')
                tree.show()
                print(tree_scores)

                print('\n[sentence tree]')
                sentence_tree.show(key=lambda n: n.identifier, data_property='sent')

                print('\n[attention top 5 areas]')
                sentence_tree.show(key=lambda n: n.identifier, data_property='score_att_top_5')

                print()

                encoder.train()
                decoder.train()

            # === backward ====
            total_loss.backward()
            if args['is_grad_clip']:
                clip_grad_norm_(encoder.parameters(), args['grad_clip'])
                clip_grad_norm_(decoder.parameters(), args['grad_clip'])
            optimizer.step()

            if iter_counter % 100 == 0:
                print("""[{}][{}]  total_loss {:.3f}  cont_stop_loss {:.3f}  word_loss {:.3f}""".format(epoch,
                                                                                                        batch,
                                                                                                        total_loss.item(),
                                                                                                        cont_stop_loss.item(),
                                                                                                        word_loss.item(),
                                                                                                        ))
            iter_counter += 1

        # === validate on val set ====
        print('start validation')
        metrics = quantity_evaluate(encoder, decoder, word2idx, 'val', args, device, VAL_DECODE_TYPE, VAL_BEAM_SIZE, verbose=True)

        if USE_TB:
            writer.add_scalar('[metric] beam size:{}/BLEU-1'.format(VAL_BEAM_SIZE), metrics['Bleu_1'], epoch)
            writer.add_scalar('[metric] beam size:{}/BLEU-2'.format(VAL_BEAM_SIZE), metrics['Bleu_2'], epoch)
            writer.add_scalar('[metric] beam size:{}/BLEU-3'.format(VAL_BEAM_SIZE), metrics['Bleu_3'], epoch)
            writer.add_scalar('[metric] beam size:{}/BLEU-4'.format(VAL_BEAM_SIZE), metrics['Bleu_4'], epoch)
            writer.add_scalar('[metric] beam size:{}/METEOR'.format(VAL_BEAM_SIZE), metrics['METEOR'], epoch)
            writer.add_scalar('[metric] beam size:{}/CIDEr'.format(VAL_BEAM_SIZE), metrics['CIDEr'], epoch)
            writer.add_scalar('[metric] beam size:{}/AVGS'.format(VAL_BEAM_SIZE), metrics['AVGS'], epoch)

        eval_score = 1/2 * metrics['Bleu_4'] + 1/2 * metrics['METEOR']
        if eval_score > best_eval_score:
            print('current eval_score {} is better than previous one {}'.format(eval_score, best_eval_score))
            epochs_since_improvement = 0
            best_eval_score = eval_score
            save_model(encoder, decoder, epoch, metrics)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement ==  EARLY_STOP_THRESHOLD:
                print('eval_score not improve after {} epochs. Stop training.'.format(epochs_since_improvement))
                break

        scheduler.step(epoch)

    if USE_TB:
        writer.close()


if __name__ == '__main__':

    args, word2idx = set_args()
    train(args, word2idx)
