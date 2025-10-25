# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
sess = tf.Session()

from keras import regularizers
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

import argparse
import sys
from scipy.sparse import vstack, csc_matrix
from utils import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file
from sklearn.model_selection import train_test_split
import time

MAX_INT = np.iinfo(np.int32).max
data_format = 0


def soft_deviation_loss(y_true, y_pred):
    ref = K.variable(np.random.normal(loc=0., scale=1., size=5000), dtype='float32')
    mu_R = K.mean(ref)
    sigma_R = K.std(ref) + K.epsilon()
    dev = (y_pred - mu_R) / sigma_R
    inlier_term = K.tanh(K.abs(dev))
    a = 5.0
    outlier_term = K.log(1.0 + K.exp(a - dev))
    loss = (1.0 - y_true) * inlier_term + y_true * outlier_term
    return loss


def dual_soft_deviation_loss(y_true, y_pred):
    loss_e = soft_deviation_loss(y_true[0], y_pred[0])
    loss_i = soft_deviation_loss(y_true[1], y_pred[1])
    return loss_e + loss_i


def dhdn_network_d(input_shape):
    x_input = Input(shape=input_shape, name='input_z')
    h = Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hl1')(x_input)
    h = Dense(250, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hl2')(h)
    h = Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hl5')(h)
    score_e = Dense(1, activation='linear', name='score_explicit')(h)

    res_input = Input(shape=input_shape, name='input_residual')
    h_res = Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hl1_res')(res_input)
    h_res = Dense(250, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hl2_res')(h_res)
    h_res = Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hl5_res')(h_res)
    score_i = Dense(1, activation='linear', name='score_implicit')(h_res)

    return Model(inputs=[x_input, res_input], outputs=[score_e, score_i])


def dhdn_network(input_shape, network_depth):
    if network_depth == 4:
        model = dhdn_network_d(input_shape)
    else:
        sys.exit("Only depth 4 is implemented")
    rms = RMSprop(clipnorm=1.)
    model.compile(loss=dual_soft_deviation_loss, optimizer=rms)
    return model


def batch_generator_dual(x, outlier_idx, inlier_idx, batch_size, nb_batch, rng, mu_N):
    counter = 0
    while True:
        ref_z = np.empty((batch_size, x.shape[1]))
        ref_res = np.empty((batch_size, x.shape[1]))
        labels = []
        n_in = len(inlier_idx)
        n_out = len(outlier_idx)
        for i in range(batch_size):
            if i % 2 == 0:
                sid = rng.choice(n_in, 1)
                ref_z[i] = x[inlier_idx[sid]]
                labels.append(0)
            else:
                sid = rng.choice(n_out, 1)
                ref_z[i] = x[outlier_idx[sid]]
                labels.append(1)
        ref_res = ref_z - mu_N
        y = np.array(labels)
        yield [ref_z, ref_res], [y, y]
        counter += 1
        if counter > nb_batch:
            counter = 0


def compute_class_center(x_inliers):
    return np.mean(x_inliers, axis=0, keepdims=True)


def inject_noise(seed, n_out, random_seed):
    rng = np.random.RandomState(random_seed)
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace=False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace=False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise


def inject_noise_sparse(seed, n_out, random_seed):
    rng = np.random.RandomState(random_seed)
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    seed = seed.tocsc()
    noise = csc_matrix((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace=False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace=False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[0, swap_feats]
    return noise.tocsr()


def load_model_weight_predict(model_name, x_test, mu_N):
    input_shape = x_test.shape[1:]
    network_depth = 4
    model = dhdn_network(input_shape, network_depth)
    model.load_weights(model_name)
    scores_e, scores_i = model.predict([x_test, x_test - mu_N])
    final_scores = (scores_e.ravel() + scores_i.ravel()) / 2.0
    return final_scores


def run_devnet(args):
    names = args.data_set.split(',')
    names = ['train_data']
    network_depth = int(args.network_depth)
    random_seed = args.ramdn_seed
    for nm in names:
        runs = args.runs
        rauc = np.zeros(runs)
        ap = np.zeros(runs)
        filename = nm.strip()
        global data_format
        data_format = int(args.data_format)
        if data_format == 0:
            x, labels = dataLoading(args.input_path + filename + ".csv")
        else:
            x, labels = get_data_from_svmlight_file(args.input_path + filename + ".svm")
            x = x.tocsr()
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices]
        n_outliers_org = outliers.shape[0]

        train_time = 0
        test_time = 0
        for i in np.arange(runs):
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.21, random_state=42,
                                                                stratify=None, shuffle=False)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            n_outliers = len(outlier_indices)

            n_noise = len(np.where(y_train == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
            n_noise = int(n_noise)

            rng = np.random.RandomState(random_seed)
            if data_format == 0:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)
                    x_train = np.delete(x_train, remove_idx, axis=0)
                    y_train = np.delete(y_train, remove_idx, axis=0)
                noises = inject_noise(outliers, n_noise, random_seed)
                x_train = np.append(x_train, noises, axis=0)
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
            else:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)
                    retain_idx = set(np.arange(x_train.shape[0])) - set(remove_idx)
                    retain_idx = list(retain_idx)
                    x_train = x_train[retain_idx]
                    y_train = y_train[retain_idx]
                noises = inject_noise_sparse(outliers, n_noise, random_seed)
                x_train = vstack([x_train, noises])
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            inliers_train = x_train[inlier_indices]
            mu_N = compute_class_center(inliers_train)
            input_shape = x_train.shape[1:]
            n_samples_trn = x_train.shape[0]
            n_outliers = len(outlier_indices)

            start_time = time.time()
            epochs = args.epochs
            batch_size = args.batch_size
            nb_batch = args.nb_batch
            model = dhdn_network(input_shape, network_depth)
            model_name = "./model/dhdn_" + filename + "_" + str(args.cont_rate) + "cr_" + str(
                args.batch_size) + "bs_" + str(args.known_outliers) + "ko_" + str(network_depth) + "d.h5"
            checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                           save_best_only=True, save_weights_only=True)

            model.fit_generator(
                batch_generator_dual(x_train, outlier_indices, inlier_indices, batch_size, nb_batch, rng, mu_N),
                steps_per_epoch=nb_batch,
                epochs=epochs,
                callbacks=[checkpointer])
            train_time += time.time() - start_time

            start_time = time.time()
            scores = load_model_weight_predict(model_name, x_test, mu_N)
            test_time += time.time() - start_time
            rauc[i], ap[i] = aucPerformance(scores, y_test)

        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        train_time = train_time / runs
        test_time = test_time / runs
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))
        print("average runtime: %.4f seconds" % (train_time + test_time))
        writeResults(filename + '_' + str(network_depth), x.shape[0], x.shape[1], n_samples_trn, n_outliers_org,
                     n_outliers, network_depth, mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time,
                     path=args.output)


parser = argparse.ArgumentParser()
parser.add_argument("--network_depth", choices=['1', '2', '4'], default='4')
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--nb_batch", type=int, default=30)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--known_outliers", type=int, default=20)
parser.add_argument("--cont_rate", type=float, default=0.1)
parser.add_argument("--input_path", type=str, default='./dataset/')
parser.add_argument("--data_set", type=str, default='test_data_snr10db')
parser.add_argument("--data_format", choices=['0', '1'], default='0')
parser.add_argument("--output", type=str, default='./results/dhdn_auc_performance.csv')
parser.add_argument("--ramdn_seed", type=int, default=42)
args = parser.parse_args()
run_devnet(args)