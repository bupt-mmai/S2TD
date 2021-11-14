import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.split.split_net_stack import SplitNet
from model.highway import Highway

__all__ = [
    "Decoder"
]


class Decoder(nn.Module):

    def __init__(self, feat_size, emb_size, wrnn_hidden_size, wrnn_num_layers, vocab_size, s_max, w_max,
                 att_type, split_threshold, emb_dropout=0., fc_dropout=0.):

        super(Decoder, self).__init__()

        self.feat_size = feat_size
        self.emb_size = emb_size
        self.wrnn_hidden_size = wrnn_hidden_size
        self.wrnn_num_layers = wrnn_num_layers
        self.vocab_size = vocab_size
        self.s_max = s_max
        self.w_max = w_max

        self.split_net = SplitNet(feat_size, s_max, att_type, split_threshold)
        self.topic_layer = Highway(feat_size, emb_size)

        self.embedding_layer = nn.Embedding(vocab_size, emb_size)
        self.emb_dropout_layer = nn.Dropout(p=emb_dropout)

        self.init_hidden_project_layer = Highway(emb_size, wrnn_hidden_size)

        self.word_rnn = nn.LSTM(input_size=emb_size, hidden_size=wrnn_hidden_size, num_layers=wrnn_num_layers,
                                batch_first=True)

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(wrnn_hidden_size + emb_size, vocab_size)  # merge structure
        )

    def init_wrnn_hidden(self, feature):
        """

        :param feature: (batch_size, emb_size)
        :return:
        """
        feature = self.init_hidden_project_layer(feature)

        h0 = feature.repeat(self.wrnn_num_layers, 1, 1)  # (wrnn_num_layers, batch_size, wrnn_hidden_size)
        c0 = torch.zeros_like(h0)

        return h0, c0

    def generate_topics(self, global_feat, features, para_lengths=None, tree_labels=None):
        """Use Sentence RNN to generate s_max topic vectors

        :param global_feat: (batch_size, feat_size)
        :param features: (batch_size, f_max, feat_size)
        :param para_lengths: None or (batch_size, )
        :param tree_labels: None or (batch_size, 2*max_len-1)
        :return: topic_vec (batch_size, s_max, emb_size), tree_list (list[Tree]),
                 scores (tensor) (batch_size, 2*max_len-1)
        """

        if self.training:
            leaf_tensor, tree_list, scores = self.split_net(global_feat, features, para_lengths, tree_labels)
        else:
            leaf_tensor, tree_list, scores = self.split_net(global_feat, features)

        topic_vec = self.topic_layer(leaf_tensor)  # (batch_size, s_max, emb_size)

        return topic_vec, tree_list, scores

    def forward(self, global_feat, features, encoded_captions, caption_lengths, tree_labels):
        """

        :param global_feat: (batch_size, feat_size)
        :param features: (batch_size, f_max, feat_size)
        :param encoded_captions: (batch_size, s_max, w_max)
        :param caption_lengths: (batch_size, s_max)
        :param tree_labels: (batch_size, 2*s_max-1)
        :return: all_predicts (batch_size, s_max, w_max, vocab_size), tree_list, scores
        """

        batch_size = global_feat.shape[0]
        device = global_feat.device

        embeddings = self.embedding_layer(encoded_captions)  # (batch_size, s_max, w_max, embed_size)
        embeddings = self.emb_dropout_layer(embeddings)

        # === Sentence RNN Part ====

        topic_vec, tree_list, scores = self.generate_topics(global_feat, features, (caption_lengths > 0).sum(1),
                                                            tree_labels)

        # === Word RNN Part ====

        all_predicts = torch.zeros(batch_size, self.s_max, self.w_max, self.vocab_size).to(device)

        for i in range(self.s_max):

            valid_batch_ind = caption_lengths[:, i] > 0
            valid_batch_size = valid_batch_ind.sum().item()

            if valid_batch_size == 0:
                break

            wrnn_input = embeddings[valid_batch_ind, i]  # (valid_batch_size, w_max, embed_size)
            seq_len = caption_lengths[valid_batch_ind, i]  # (valid_batch_size, )

            wrnn_input_pps = pack_padded_sequence(wrnn_input, lengths=seq_len, batch_first=True, enforce_sorted=False)
            h0, c0 = self.init_wrnn_hidden(topic_vec[valid_batch_ind, i])

            wrnn_output_pps, _ = self.word_rnn(wrnn_input_pps, (h0, c0))

            # output (valid_batch_size, w_max, wrnn_hidden_size)
            wrnn_output, _ = pad_packed_sequence(wrnn_output_pps, batch_first=True, total_length=self.w_max)

            # merge structure
            topic = topic_vec[valid_batch_ind, i, None, :].expand(valid_batch_size, self.w_max, self.emb_size)
            fc_input = torch.cat([wrnn_output, topic], dim=-1)  # (valid_batch_size, w_max, wrnn_hidden_size + emb_size)

            predicts = self.fc_layer(fc_input)  # (valid_batch_size, w_max, vocab_size)

            all_predicts[valid_batch_ind, i] = predicts

        return all_predicts, tree_list, scores
