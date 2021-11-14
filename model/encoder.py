from torch import nn

from .highway import Highway


__all__ = [
    "Encoder"
]


class Encoder(nn.Module):

    def __init__(self, input_size, output_size, f_max):

        super(Encoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.f_max = f_max

        self.project_layer = Highway(input_size, output_size)

    def forward(self, input_):

        features = self.project_layer(input_)
        global_feat = features.max(dim=1)[0]  # (batch_size, output_size)

        return global_feat, features
