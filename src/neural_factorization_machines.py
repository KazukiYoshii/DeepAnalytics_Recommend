# 高速化のため
import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import chainer
import chainer.functions as F
import chainer.links as L


class NeuralFactorizationMachines(chainer.Chain):

    """Neural Factorization Machinesのオレオレchainer実装
    https://arxiv.org/abs/1708.05027

    実装力不足なので、入力が二又
    """

    def __init__(self, feature_dim, factor_dim=8):
        super(NeuralFactorizationMachines, self).__init__()
        with self.init_scope():
            w = chainer.initializers.HeNormal()
            self.linear_regression = L.Linear(None, 1, initialW=w)
            self.embed = L.EmbedID(feature_dim, factor_dim, ignore_label=0)
            self.fc1 = L.Linear(None, 128, initialW=w)
            self.ln1 = L.LayerNormalization()
            self.fc2 = L.Linear(128, 128, initialW=w)
            self.ln2 = L.LayerNormalization()
            self.fc3 = L.Linear(128, 1, initialW=w)

    def __call__(self, x1, x2):
        # linear regression
        reg = self.linear_regression(x1)

        # embedding
        factors = self.embed(x2)

        # Bi-Interaction Layer
        # https://arxiv.org/abs/1708.05027 formula (4)
        sum_square = F.square(F.sum(factors, axis=1))
        square_sum = F.sum(F.square(factors), axis=1)
        second_order = 0.5 * (sum_square + square_sum)

        # MLP
        h1 = F.dropout(F.tanh(self.ln1(self.fc1(second_order))))
        h2 = F.dropout(F.tanh(self.ln2(self.fc2(h1))))
        y = self.fc3(h2) + reg

        return y
