import argparse
from dataset import LoadDataset
import os
from test import *
from utils import *
import time
import tensorflow as tf
from numpy.random import seed
import scipy.io as sio

seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Learning to Classify')
    parser.add_argument('--outer_iter_num', type=int, default=600, help='the outer iteration number')
    parser.add_argument('--inner_iter_num', type=int, default=6000, help='the inner iteration number')
    parser.add_argument('--batch_size', type=int, default=256, help='the number of batch_size')
    parser.add_argument('--hid_dim', type=int, default=600, help='the dimensionality of hidden layer')
    parser.add_argument('--data_dir', type=str, default='/data/ZSL/datasets/AwA1', help='the train directory')
    parser.add_argument('--lamb', type=float, default=0.3, help='the balance parameter')  # AwA 0.2
    parser.add_argument('--alpha', type=float, default=1e-4, help='the balance parameter')
    parser.add_argument('--outer_lr', type=float, default=1e-6, help='the outer learning rate')  # 5e-6
    parser.add_argument('--inner_lr', type=float, default=5e-5, help='the inner learning rate')  # 5e-5
    parser.add_argument('--T', type=int, default=2, help='the temperature')
    parser.add_argument('--interval', type=int, default=200, help='the interval number')
    parser.add_argument('--manualSeed', type=int, default=1000, help='the random seed')
    args = parser.parse_args()
    return args

class Model(object):
    def __init__(self, args):
        self.args = args
        self.outer_iter_num = args.outer_iter_num
        self.inner_iter_num = args.inner_iter_num
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.lamb = args.lamb
        self.alpha = args.alpha
        self.hid_dim = args.hid_dim
        self.inner_lr = args.inner_lr
        self.outer_lr = args.outer_lr
        self.interval = args.interval
        self.T = args.T

    def create_model(self):
        ## load the data
        self.dataset = LoadDataset(self.data_dir)

        # Train data
        train_fea = self.dataset.train_fea  # the train feature
        train_sem = self.dataset.train_sem  # the train semantic
        train_lab = self.dataset.train_onehot  # the train onehot label

        # Define the placeholder
        img_dim = train_fea.shape[1]  # the image dimension
        sem_dim = train_sem.shape[1]  # the semantic dimension
        lab_dim = train_lab.shape[1]  # the label dimension
        hid_dim = self.hid_dim

        self.img_pl = tf.placeholder(tf.float32, [None, img_dim], name='image')
        self.sem_pl = tf.placeholder(tf.float32, [None, sem_dim], name='semantic')
        self.pro_pl = tf.placeholder(tf.float32, [None, sem_dim])
        self.lab_pl = tf.placeholder(tf.float32, [None, lab_dim])
        self.cla_pl = tf.placeholder(tf.float32, [None, hid_dim])
        self.sim_pl = tf.placeholder(tf.float32, [None, 1], name='similarity')
        similarity_float = tf.to_float(self.sim_pl)
        self.lr_pl = tf.placeholder(tf.float32)

        with tf.name_scope('image_channel'):
            W_img = self.weight_variable([img_dim, hid_dim])
            b_img = self.bias_variable([hid_dim])
            self.hid_img = self.Dense(self.img_pl, W_img, b_img, tf.nn.relu)
            self.W_classifier = self.weight_variable([hid_dim, lab_dim])
            logits = tf.matmul(self.hid_img, self.W_classifier)
            prediction = tf.nn.log_softmax(logits)

        with tf.variable_scope('semantic_channel') as scope:
            W_sem = self.weight_variable([sem_dim, hid_dim])
            b_sem = self.bias_variable([hid_dim])
            hid_sem = self.Dense(self.sem_pl, W_sem, b_sem, tf.nn.tanh)
            scope.reuse_variables()
            hid_pro = self.Dense(self.pro_pl, W_sem, b_sem, tf.nn.tanh)

        with tf.name_scope('outer_loss'):
            margin = 0.01  # 0.01
            base_loss = contrastive_loss(self.hid_img, hid_sem, similarity_float, margin)
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.lab_pl))
            self.outer_loss = cross_entropy + base_loss

        with tf.name_scope('classifier'):
            W_cla = self.weight_variable([hid_dim, hid_dim])
            b_cla = self.bias_variable([hid_dim])
            self.G_out = self.Dense(hid_pro, W_cla, b_cla, tf.nn.relu)
            predict = tf.matmul(self.hid_img, tf.transpose(self.G_out))
            self.predict = tf.nn.log_softmax(predict)

        with tf.name_scope('inner_loss'):
            classifier_loss = tf.reduce_mean(-tf.reduce_mean(self.lab_pl * self.predict, reduction_indices=[1]))
            kl_loss = kl_for_log_probs(logits, predict, self.T)
            loss_a = self.lamb*classifier_loss+tf.reduce_mean(tf.square(self.G_out-self.cla_pl))

            vars = tf.trainable_variables()
            regularisers = tf.add_n([tf.nn.l2_loss(v) for v in vars])
            self.inner_loss = loss_a + kl_loss + 1e-3*regularisers  #+ self.alpha*base_loss

        self.outer_optimizer = tf.train.AdamOptimizer(self.lr_pl).minimize(self.outer_loss)
        self.inner_optimizer = tf.train.AdamOptimizer(self.lr_pl).minimize(self.inner_loss)

    def train(self):
        ##
        loss = []
        seen_acc = []
        unseen_acc = []
        H_acc = []
        print('Start training!')
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.dataset = LoadDataset(self.data_dir)
        next_batch = self.dataset.get_batch
        train_pro = self.dataset.train_pro
        for j in range(self.outer_iter_num):

            batch_img, batch_lab, batch_sem, batch_sim = next_batch(self.batch_size)
            _, loss_value = self.sess.run([self.outer_optimizer, self.outer_loss], feed_dict={self.img_pl: batch_img,
                                                                                              self.lab_pl: batch_lab,
                                                                                              self.sem_pl: batch_sem,
                                                                                              self.sim_pl: batch_sim,
                                                                                              self.lr_pl: self.outer_lr})
        weight = self.sess.run(tf.transpose(self.W_classifier))
        for k in range(self.inner_iter_num + 1):
            batch_img, batch_lab, batch_sem, batch_sim = next_batch(self.batch_size)
            _, loss_value = self.sess.run([self.inner_optimizer, self.inner_loss],
                                          feed_dict={self.img_pl: batch_img, self.pro_pl: train_pro,
                                                     self.sim_pl: batch_sim, self.lab_pl: batch_lab,self.cla_pl:weight,
                                                     self.sem_pl: batch_sem, self.lr_pl: self.inner_lr})


            if k % self.interval == 0:
                print('Iteration is :', k)
                acc_seen, acc_unseen, H = self.test()
                loss.append(loss_value)
                # print(loss_value)
                seen_acc.append(acc_seen)
                unseen_acc.append(acc_unseen)
                H_acc.append(H)
        sio.savemat('awa1.mat', {'loss': loss, 'seen_acc': seen_acc, 'unseen_acc': unseen_acc, 'H_acc': H_acc})

    def test(self):
        start_time = time.time()
        nn = Test_nn(self.sess, self.img_pl, self.pro_pl, self.hid_img, self.G_out)
        sim_zsl, acc = nn.test_zsl(self.dataset)
        print('Accuracy: unseen class:%4.4f'%acc)

        sim_seen_gzsl, acc_seen = nn.test_seen(self.dataset)
        sim_unseen_gzsl, acc_unseen = nn.test_unseen(self.dataset)
        H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        print('accuracy: unseen class:%4.4f'%acc_unseen, '| seen class:%4.4f'%acc_seen, '| harmonic:%4.4f'%H)

        end_time = time.time()
        print('# The test time is: %4.4f' % (end_time - start_time))
        # sio.savemat('sim.mat', {'sim_zsl': sim_zsl, 'sim_seen_gzsl': sim_seen_gzsl, 'sim_unseen_gzsl': sim_unseen_gzsl})
        return acc_seen, acc_unseen, H

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=1e-2)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(1e-2, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def Dense(input, weights, bias, activate_fc=None):
        output = tf.matmul(input, weights) + bias
        if activate_fc:
            output = activate_fc(output)
        return output

def main():
    args = parse_args()
    if args is None:
        exit()
    random.seed(args.manualSeed)
    tf.set_random_seed(args.manualSeed)
    model = Model(args)
    model.create_model()
    model.train()
    print("Training finished!")
    model.test()
    print("Test finished!")

if __name__ == '__main__':
    main()
