import torch
import torch.nn as nn

from model.FM.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class DFM_CONTENT(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, conti_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.embed_output_dim = (len(field_dims)+1) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        # add
        self.emb_linear = nn.Linear(conti_dim, embed_dim)
        self.add_linear = nn.Linear(conti_dim, 1, bias=False)
        nn.init.xavier_normal_(self.emb_linear.weight)
        nn.init.xavier_normal_(self.add_linear.weight)

    def forward(self, x, conti_x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :param conti_x: Long tensor of size ``(batch_size, conti_dim)``
        """

        embed_x = self.embedding(x) # batch_size x num_fields x embed_dim
        added_emb = self.emb_linear(conti_x) # batch_size x embed_dim
        embed_x = torch.cat([embed_x, added_emb.unsqueeze(1)], dim=1) # batch_size x (num_fields + 1) x embed_dim

        x = self.linear(x) + self.add_linear(conti_x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))

    def get_embs(self, sel_x, selected_idx_list:list):
        embs = self.embedding.get_selected_embedding(sel_x, selected_idx_list)

        return torch.sum(embs, dim=1)

    def get_mixed_embs(self, sel_x, selected_idx_list: list, conti_x):
        embed_x = self.embedding.get_selected_embedding(sel_x, selected_idx_list) # batch_size x sel_num_fields x embed_dim

        added_emb = self.emb_linear(conti_x)  # batch_size x embed_dim

        embed_x = torch.cat([embed_x, added_emb.unsqueeze(1)], dim=1)  # batch_size x (sel_num_fields + 1) x embed_dim

        return torch.sum(embed_x, dim=1)