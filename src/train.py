import argparse
import numpy as np
import pickle
from multiprocessing import Process, Queue
from sklearn.model_selection import train_test_split
from progressbar import ProgressBar


# 高速化のため
import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import chainer
import chainer.functions as F
from chainer import cuda, Variable, optimizers, serializers
from neural_factorization_machines import NeuralFactorizationMachines

OUTPUT_DIR = ""


def load_dataset(filepath):
    with open(filepath, "rb") as f:
        X, y, user_id2index, product_id2index, feature_dim = pickle.load(f)

    return train_test_split(X, y, test_size=0.2, random_state=917), feature_dim


def save_model(model, suffix=""):
    serializers.save_hdf5("{}/my_model_{}.h5".format(OUTPUT_DIR, suffix), model)


def make_minibatch(batch_queue, X, y, batch_size):
    x1 = []
    x2 = []
    t = []
    for i in np.random.permutation(range(len(X))):
        feature = np.zeros(feature_dim, dtype=xp.float32)
        feature[X[i]] = 1
        x1.append(feature)

        feature2 = np.zeros(feature_dim, dtype=xp.int32)
        feature2[X[i]] = X[i]
        x2.append(feature2)

        t.append(y[i])

        if len(x1) == batch_size:

            x1 = Variable(xp.array(x1, dtype=xp.float32))
            x2 = Variable(xp.array(x2, dtype=xp.int32))
            t = Variable(xp.array(t, dtype=xp.float32).reshape(len(t), 1))

            batch_queue.put((x1, x2, t))
            x1 = []
            x2 = []
            t = []

    batch_queue.put((x1, x2, t))
    batch_queue.put(None)


def one_epoch(model, optimizer, X, y, batch_size, feature_dim, xp, train=True):
    x1 = []
    x2 = []
    t = []
    sum_loss = 0
    workers = []
    queue = Queue(maxsize=10)
    X = np.random.permutation(X)
    num_workers = 10

    for i in range(num_workers):
        tmp_X = X[i * (len(X) // num_workers):(i + 1) * (len(X) // num_workers)]
        tmp_y = y[i * (len(y) // num_workers):(i + 1) * (len(y) // num_workers)]
        worker = Process(target=make_minibatch, args=(queue, tmp_X, tmp_y, batch_size))
        workers.append(worker)

    for i in range(num_workers):
        workers[i].start()

    p = ProgressBar(max_value=(len(X) // num_workers) + 1)
    i = 0
    while(True):
        data = queue.get()
        if data is None:
            end_worker_num += 1
            if end_worker_num == num_workers:
                break

        x1, x2, t = data

        predict = model(x1, x2)
        loss = F.mean_squared_error(predict, t)
        loss.backward()
        optimizer.update(model, x1, x2)
        sum_loss += float(loss.data) * batch_size
        p.update(i + 1)
        i += 1
    p.finish()

    if train:
        print("   train RMSE:{}".format(sum_loss / len(X)))
    else:
        print("   test RMSE:{}".format(sum_loss / len(X)))

    return model


def train(model, optimizer, X_train, y_train, X_test, y_test, num_iter,
          batch_size, feature_dim, test=True, save=False, snapshot=10, gpu=-1):
    xp = cuda.cupy if gpu >= 0 else np

    # iterを回す
    for epoch in range(1, num_iter + 1):
        print("epoch {}".format(epoch))
        model = one_epoch(model, optimizer, X_train, y_train, batch_size,
                          feature_dim, xp, train=True)
        if test:
            model = one_epoch(model, optimizer, X_test, y_test, batch_size,
                              feature_dim, xp, train=False)

        if epoch % snapshot == 0 and save:
            save_model(model, suffix=str(epoch))

    if save:
        save_model(model, suffix="final")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Factorization Machines trainer')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-d', default="data/train_dataA.pkl", type=str,
                        help='dataset file path')
    parser.add_argument('--output', '-o', required=True, type=str,
                        help='output model file path without extension')
    parser.add_argument('--iter', default=100, type=int,
                        help='number of iteration')
    parser.add_argument('--factor_dim', default=8, type=int,
                        help='number of factor dimension')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='minibatch size')
    parser.add_argument('--snapshot', default=10, type=int,
                        help='snapshot iterations')
    args = parser.parse_args()

    OUTPUT_DIR = args.output

    # データセットのロード
    (X_train, X_test, y_train, y_test), feature_dim = load_dataset(args.dataset)

    # モデルのロード
    model = NeuralFactorizationMachines(feature_dim)

    # optimiserの設定
    optimizer = optimizers.Adam(alpha=1e-5)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-6))

    # gpuの設定
    gpu_device = None
    xp = np
    if args.gpu >= 0:
        cuda.check_cuda_available()
        model.to_gpu()

    train(model, optimizer, X_train, y_train, X_test,
          y_test, args.iter, args.batch_size, feature_dim,
          save=True, snapshot=args.snapshot, gpu=args.gpu)
