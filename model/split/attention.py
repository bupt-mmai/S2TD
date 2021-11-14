import torch
from torch import nn


class AdditiveAttention(nn.Module):

    def __init__(self, feat_size, query_size, hidden_size):

        super(AdditiveAttention, self).__init__()

        self.feat_size = feat_size
        self.query_size = query_size
        self.hidden_size = hidden_size

        self.feat_project_layer = nn.Linear(feat_size, hidden_size, bias=False)
        self.query_project_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.v_dot = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, features_matrix, query_vec):
        """
        Arguments:
            features_matrix (Tensor): features tensor matrix with shape (batch_size, f_max, feat_size)
            query_vec (Tensor): query tensor vector with shape (batch_size, query_size)
        Returns:
            weighted_feat (Tensor): weighted features tensor of features_matrix with shape（batch_size, feat_size）
            scores (Tensor): attention distribution scores with shape (batch_size, f_max)
        """

        # === compute attention distribution scores ====
        query = self.query_project_layer(query_vec)  # (batch_size, hidden_size)
        feats = self.feat_project_layer(features_matrix)  # (batch_size, f_max, hidden_size)
        scores_unnorm = self.tanh(query.unsqueeze(1) + feats)  # (batch_size, f_max, hidden_size)
        scores_unnorm = self.v_dot(scores_unnorm)  # (batch_size, f_max, 1)

        scores = self.softmax(scores_unnorm)  # (batch_size, f_max, 1)

        # === soft-attention mechanism ====
        weighted_feat = torch.sum((features_matrix * scores), dim=1)  # (batch_size, feat_size)

        return weighted_feat, scores.squeeze(-1)


class ScaledDotAttention(nn.Module):

    def __init__(self, feat_size, query_size):

        super(ScaledDotAttention, self).__init__()

        self.feat_size = feat_size
        self.query_size = query_size
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features_matrix, query_vec):
        """
        Arguments:
            features_matrix (Tensor): features tensor matrix with shape (batch_size, f_max, feat_size)
            query_vec (Tensor): query tensor vector with shape (batch_size, query_size)
        Returns:
            weighted_feat (Tensor): weighted features tensor of features_matrix with shape（batch_size, feat_size）
            scores (Tensor): attention distribution scores with shape (batch_size, f_max)
        """

        # === compute attention distribution scores ====
        # (batch_size, f_max, 1)
        scores_unnorm = torch.bmm(features_matrix, query_vec.unsqueeze(-1))/(self.query_size**(1/2))

        scores = self.softmax(scores_unnorm)  # (batch_size, f_max, 1)

        # === soft-attention mechanism ====
        weighted_feat = torch.sum((features_matrix * scores), dim=1)  # (batch_size, feat_size)

        return weighted_feat, scores.squeeze(-1)
