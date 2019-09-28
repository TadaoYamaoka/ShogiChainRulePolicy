import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import Chain
import chainer.functions as F
import chainer.links as L

# 入力特徴量：駒の種類×手番 + 持ち駒の種類×手番
FEATURES_NUM = 14 * 2 + 7 * 2
# 駒打ち
FROM_HAND_PIECE = 81
# 移動
FROM_MOVE = 7

k = 192
dropout_ratio = 0.1
class PolicyNetwork(Chain):
    def __init__(self):
        super(PolicyNetwork, self).__init__(
            l1=L.Convolution2D(in_channels = FEATURES_NUM, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l2=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l3=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l4=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l5=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l6=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l7=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l8=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l9=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l10=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l11=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            # 移動先座標
            l12_to=L.Convolution2D(in_channels = k, out_channels = 1, ksize = 1, nobias = True),
            l12_to_b=L.Bias(shape=(9 * 9)),
            # 移動元座標（駒打ちの場合は81）
            l12_from=L.Convolution2D(in_channels = k, out_channels = 1, ksize = 1),
            l13_from=L.Linear(9 * 9, 9 * 9 + 1),
            # 打つ駒（移動の場合は7）
            l12_hand_piece=L.Convolution2D(in_channels = k, out_channels = 1, ksize = 1),
            l13_hand_piece=L.Linear(9 * 9, 7 + 1),

            norm1=L.BatchNormalization(k),
            norm2=L.BatchNormalization(k),
            norm3=L.BatchNormalization(k),
            norm4=L.BatchNormalization(k),
            norm5=L.BatchNormalization(k),
            norm6=L.BatchNormalization(k),
            norm7=L.BatchNormalization(k),
            norm8=L.BatchNormalization(k),
            norm9=L.BatchNormalization(k),
            norm10=L.BatchNormalization(k)
        )

    def __call__(self, x):
        u1 = self.l1(x)
        # Residual block
        h1 = F.relu(self.norm1(u1))
        h2 = F.dropout(F.relu(self.norm2(self.l2(h1))), ratio=dropout_ratio)
        u3 = self.l3(h2) + u1
        # Residual block
        h3 = F.relu(self.norm3(u3))
        h4 = F.dropout(F.relu(self.norm4(self.l4(h3))), ratio=dropout_ratio)
        u5 = self.l5(h4) + u3
        # Residual block
        h5 = F.relu(self.norm5(u5))
        h6 = F.dropout(F.relu(self.norm6(self.l6(h5))), ratio=dropout_ratio)
        u7 = self.l7(h6) + u5
        # Residual block
        h7 = F.relu(self.norm7(u7))
        h8 = F.dropout(F.relu(self.norm8(self.l8(h7))), ratio=dropout_ratio)
        u9 = self.l9(h8) + u7
        # Residual block
        h9 = F.relu(self.norm9(u9))
        h10 = F.dropout(F.relu(self.norm10(self.l10(h9))), ratio=dropout_ratio)
        u11 = self.l11(h10) + u9
        # output to
        h12_to = self.l12_to(u11)
        out_to = self.l12_to_b(F.reshape(h12_to, (-1, 9 * 9)))
        # output from
        h12_from = F.relu(self.l12_from(u11))
        out_from = self.l13_from(h12_from)
        # output hand piece
        h12_hand_piece = F.relu(self.l12_hand_piece(u11))
        out_hand_piece = self.l13_hand_piece(h12_hand_piece)

        return out_to, out_from, out_hand_piece
