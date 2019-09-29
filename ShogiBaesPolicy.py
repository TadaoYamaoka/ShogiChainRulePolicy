import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from cshogi import *
from policy_network_base import *

import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('train_hcpe')
parser.add_argument('test_hcpe')
parser.add_argument('--model', default='model')
parser.add_argument('--state', default='state')
parser.add_argument('--epoch', '-e', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--testbatchsize', type=int, default=640)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=int, default=1e-4)
parser.add_argument('--eval_interval', type=int, default=1000)
parser.add_argument('--log', default=None)

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

model = PolicyNetwork()
model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=args.lr)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(args.weight_decay))

train_data = np.fromfile(args.train_hcpe, dtype=HuffmanCodedPosAndEval)
test_data = np.fromfile(args.test_hcpe, dtype=HuffmanCodedPosAndEval)

def make_position_features(board, features):
    # piece
    board.piece_planes_rotate(features)
    pieces_in_hand = board.pieces_in_hand
    if board.turn == WHITE:
        # 白の場合、反転
        pieces_in_hand = (pieces_in_hand[1], pieces_in_hand[0])
    for c, hands in enumerate(pieces_in_hand):
        for hp, num in enumerate(hands):
            if hp == HPAWN:
                max_hp_num = 8
            elif hp == HBISHOP or hp == HROOK:
                max_hp_num = 2
            else:
                max_hp_num = 4

            features[28 + c * 7 + hp].fill(num / max_hp_num)

def make_output_labels(move):
    # 移動先座標
    to_sq = move_to(move)
    # 移動方向((8方向+桂馬2方向)×成2+持ち駒7種類)
    if not move_is_drop(move):
        from_sq = move_from(move)

        to_file, to_rank = divmod(to_sq, 9)
        from_file, from_rank = divmod(from_sq, 9)

        if to_file == from_file:
            if to_rank > from_rank:
                dir = UP
            else:
                dir = DOWN
        elif to_rank == from_rank:
            if to_file > from_file:
                dir = LEFT
            else:
                dir = RIGHT
        elif to_rank - from_rank == 2 and abs(to_file - to_file):
            if to_file > from_file:
                dir = UP2_LEFT
            else:
                dir = UP2_RIGHT
        elif to_file > from_file:
            if to_rank > from_rank:
                dir = UP_LEFT
            else:
                dir = DOWN_LEFT
        else:
            if to_rank > from_rank:
                dir = UP_RIGHT
            else:
                dir = DOWN_RIGHT

        if move_is_promotion(move):
            dir += 10
    else:
        dir = 20 + move_drop_hand_piece(move)

    return 9 * 9 * dir + to_sq

board = Board()
def mini_batch(hcpevec):
    features = np.empty((len(hcpevec), FEATURES_NUM, 9, 9), dtype=np.float32)
    target = np.empty((len(hcpevec)), dtype=np.int32)

    for i, hcpe in enumerate(hcpevec):
        board.set_hcp(hcpe['hcp'])
        make_position_features(board, features[i])
        target[i] = make_output_labels(hcpe['bestMove16'] if board.turn == BLACK else move_rotate(hcpe['bestMove16']))

    return (Variable(cuda.to_gpu(features)),
            Variable(cuda.to_gpu(target)),
            )

itr = 0
sum_loss = 0
eval_interval = args.eval_interval
for e in range(args.epoch):
    np.random.shuffle(train_data)

    itr_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(train_data) - args.batchsize + 1, args.batchsize):
        x, t = mini_batch(train_data[i:i+args.batchsize])
        y = model(x)

        model.cleargrads()
        loss = F.softmax_cross_entropy(y, t)
        loss.backward()
        optimizer.update()

        itr += 1
        sum_loss += loss.data
        itr_epoch += 1
        sum_loss_epoch += loss.data

        # print train loss
        if optimizer.t % args.eval_interval == 0:
            x, t = mini_batch(np.random.choice(test_data, args.testbatchsize))
            with chainer.no_backprop_mode():
                with chainer.using_config('train', False):
                    y = model(x)

                loss = F.softmax_cross_entropy(y, t)

                logging.info('epoch = {}, iteration = {}, loss = {}, test loss = {}, test accuracy = {}'.format(
                    optimizer.epoch + 1, optimizer.t,
                    sum_loss / itr,
                    loss.data,
                    F.accuracy(y, t).data,
                    ))
            itr = 0
            sum_loss = 0

    # print train loss for each epoch
    itr_test = 0
    sum_test_loss = 0
    sum_test_accuracy = 0
    for i in range(0, len(test_data) - args.testbatchsize, args.testbatchsize):
        x, t = mini_batch(test_data[i:i+args.testbatchsize])
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                y = model(x)

            itr_test += 1
            loss = F.softmax_cross_entropy(y, t)
            sum_test_loss += loss.data

            sum_test_accuracy += F.accuracy(y, t).data

    logging.info('epoch = {}, iteration = {}, train loss avr = {}, test_loss = {}, test accuracy = {}'.format(
        optimizer.epoch + 1, optimizer.t,
        sum_loss_epoch / itr_epoch,
        sum_test_loss / itr_test,
        sum_test_accuracy / itr_test,
        ))

    optimizer.new_epoch()

print('save the model')
serializers.save_npz(args.model, model)
print('save the optimizer')
serializers.save_npz(args.state, optimizer)
