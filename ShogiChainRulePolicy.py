import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from cshogi import *
from policy_network import *

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
    # 移動か駒打ちか
    if not move_is_drop(move):
        # 移動元座標
        from_sq = move_from(move)
        # 移動
        hand_piece = FROM_MOVE
    else:
        # 駒打ち
        from_sq = FROM_HAND_PIECE
        # 打ち駒
        hand_piece = move_drop_hand_piece(move)

    return to_sq, from_sq, hand_piece

board = Board()
def mini_batch(hcpevec):
    features = np.empty((len(hcpevec), FEATURES_NUM, 9, 9), dtype=np.float32)
    target_to = np.empty((len(hcpevec)), dtype=np.int32)
    target_kind = np.empty((len(hcpevec)), dtype=np.int32)
    target_from = np.empty((len(hcpevec)), dtype=np.int32)
    target_hand_piece = np.empty((len(hcpevec)), dtype=np.int32)

    for i, hcpe in enumerate(hcpevec):
        board.set_hcp(hcpe['hcp'])
        make_position_features(board, features[i])
        target_to[i], target_from[i], target_hand_piece[i] = make_output_labels(hcpe['bestMove16'] if board.turn == BLACK else move_rotate(hcpe['bestMove16']))

    return (Variable(cuda.to_gpu(features)),
            Variable(cuda.to_gpu(target_to)),
            Variable(cuda.to_gpu(target_from)),
            Variable(cuda.to_gpu(target_hand_piece)),
            )

itr = 0
sum_loss1 = 0
sum_loss2 = 0
sum_loss3 = 0
sum_loss = 0
eval_interval = args.eval_interval
for e in range(args.epoch):
    np.random.shuffle(train_data)

    itr_epoch = 0
    sum_loss1_epoch = 0
    sum_loss2_epoch = 0
    sum_loss3_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(train_data) - args.batchsize + 1, args.batchsize):
        x, t1, t2, t3 = mini_batch(train_data[i:i+args.batchsize])
        y1, y2, y3 = model(x)

        model.cleargrads()
        loss1 = F.softmax_cross_entropy(y1, t1)
        loss2 = F.softmax_cross_entropy(y2, t2)
        loss3 = F.softmax_cross_entropy(y3, t3)
        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.update()

        itr += 1
        sum_loss1 += loss1.data
        sum_loss2 += loss2.data
        sum_loss3 += loss3.data
        sum_loss += loss.data
        itr_epoch += 1
        sum_loss1_epoch += loss1.data
        sum_loss2_epoch += loss2.data
        sum_loss3_epoch += loss3.data
        sum_loss_epoch += loss.data

        # print train loss
        if optimizer.t % args.eval_interval == 0:
            x, t1, t2, t3 = mini_batch(np.random.choice(test_data, args.testbatchsize))
            with chainer.no_backprop_mode():
                with chainer.using_config('train', False):
                    y1, y2, y3 = model(x)

                loss1 = F.softmax_cross_entropy(y1, t1)
                loss2 = F.softmax_cross_entropy(y2, t2)
                loss3 = F.softmax_cross_entropy(y3, t3)
                loss = loss1 + loss2 + loss3

                # calc accuracy
                p1 = F.softmax(y1)
                p2 = F.softmax(y2)
                p3 = F.softmax(y3)
                p12 = F.batch_matmul(p1.reshape(-1, 81, 1), p2.reshape(-1, 1, 82)).reshape(-1, 81 * 82, 1)
                p = F.batch_matmul(p12, p3.reshape(-1, 1, 8)).reshape(-1, 81 * 82 * 8)
                t = t1 * 82 * 8 + t2 * 8 + t3

                logging.info('epoch = {}, iteration = {}, loss = {}, {}, {}, {}, test loss = {}, {}, {}, {}, test accuracy = {}, {}, {}, {}'.format(
                    optimizer.epoch + 1, optimizer.t,
                    sum_loss1 / itr, sum_loss2 / itr, sum_loss3 / itr, sum_loss / itr,
                    loss1.data, loss2.data, loss3.data, loss.data,
                    F.accuracy(y1, t1).data,
                    F.accuracy(y2, t2).data,
                    F.accuracy(y3, t3).data,
                    F.accuracy(p, t).data,
                    ))
            itr = 0
            sum_loss1 = 0
            sum_loss2 = 0
            sum_loss3 = 0
            sum_loss = 0

    # print train loss for each epoch
    itr_test = 0
    sum_test_loss1 = 0
    sum_test_loss2 = 0
    sum_test_loss3 = 0
    sum_test_loss = 0
    sum_test_accuracy1 = 0
    sum_test_accuracy2 = 0
    sum_test_accuracy3 = 0
    sum_test_accuracy = 0
    for i in range(0, len(test_data) - args.testbatchsize, args.testbatchsize):
        x, t1, t2, t3 = mini_batch(test_data[i:i+args.testbatchsize])
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                y1, y2, y3 = model(x)

            itr_test += 1
            loss1 = F.softmax_cross_entropy(y1, t1)
            loss2 = F.softmax_cross_entropy(y2, t2)
            loss3 = F.softmax_cross_entropy(y3, t3)
            loss = loss1 + loss2 + loss3
            sum_test_loss1 += loss1.data
            sum_test_loss2 += loss2.data
            sum_test_loss3 += loss3.data
            sum_test_loss += loss.data

            # calc accuracy
            p1 = F.softmax(y1)
            p2 = F.softmax(y2)
            p3 = F.softmax(y3)
            p12 = F.batch_matmul(p1.reshape(-1, 81, 1), p2.reshape(-1, 1, 82)).reshape(-1, 81 * 82, 1)
            p = F.batch_matmul(p12, p3.reshape(-1, 1, 8)).reshape(-1, 81 * 82 * 8)
            t = t1 * 82 * 8 + t2 * 8 + t3
            sum_test_accuracy1 += F.accuracy(y1, t1).data
            sum_test_accuracy2 += F.accuracy(y2, t2).data
            sum_test_accuracy3 += F.accuracy(y3, t3).data
            sum_test_accuracy += F.accuracy(p, t).data

    logging.info('epoch = {}, iteration = {}, train loss avr = {}, {}, {}, {}, test_loss = {}, {}, {}, {}, test accuracy = {}, {}, {}, {}'.format(
        optimizer.epoch + 1, optimizer.t,
        sum_loss1_epoch / itr_epoch, sum_loss2_epoch / itr_epoch, sum_loss3_epoch / itr_epoch, sum_loss_epoch / itr_epoch,
        sum_test_loss1 / itr_test, sum_test_loss2 / itr_test, sum_test_loss3 / itr_test, sum_test_loss / itr_test,
        sum_test_accuracy1 / itr_test,
        sum_test_accuracy2 / itr_test,
        sum_test_accuracy3 / itr_test,
        sum_test_accuracy / itr_test,
        ))

    optimizer.new_epoch()

print('save the model')
serializers.save_npz(args.model, model)
print('save the optimizer')
serializers.save_npz(args.state, optimizer)
