import torch
import torch.nn as nn

from model.FM.layer import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron, FeaturesLinear


class NFM_CONTENT(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.

    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts, conti_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropouts[1])

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

        embed_x = self.embedding(x)  # batch_size x num_fields x embed_dim
        added_emb = self.emb_linear(conti_x)  # batch_size x embed_dim
        embed_x = torch.cat([embed_x, added_emb.unsqueeze(1)], dim=1)  # batch_size x (num_fields + 1) x embed_dim

        cross_term = self.fm(embed_x) # batch_size x embed_dim
        x = self.linear(x) + self.add_linear(conti_x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))

    def get_embs(self, sel_x, selected_idx_list: list):
        embs = self.embedding.get_selected_embedding(sel_x, selected_idx_list)

        return torch.sum(embs, dim=1)

    def get_mixed_embs(self, sel_x, selected_idx_list: list, conti_x):
        embed_x = self.embedding.get_selected_embedding(sel_x,
                    selected_idx_list)  # batch_size x sel_num_fields x embed_dim

        added_emb = self.emb_linear(conti_x)  # batch_size x embed_dim

        embed_x = torch.cat([embed_x, added_emb.unsqueeze(1)], dim=1)  # batch_size x (sel_num_fields + 1) x embed_dim

        return torch.sum(embed_x, dim=1)