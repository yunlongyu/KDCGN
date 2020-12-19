import tensorflow as tf
import numpy as np
from numpy import *

def consine_distance(a,b):
    # a.shape = N x D
    # b.shape = M x D
    a_normalized = tf.nn.l2_normalize(a, dim=1)  # 0 is colum, 1 is row
    b_normalized = tf.nn.l2_normalize(b, dim=1)
    product = tf.matmul(a_normalized, b_normalized, adjoint_b=True)
    dist = 1-product
    return dist

def euclidean_distance(a,b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a-b), axis=2)

def contrastive_loss(x1, x2, y, margin):
    with tf.name_scope("contrastive_loss"):
        distance = tf.reduce_mean(tf.square(x1-x2))
        similarity = y * distance           # keep the similar label (1) close to each other
        dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))  # give penalty to dissimilar label if the distance is bigger than margin
    return tf.reduce_mean(dissimilarity + similarity) / 2

def get_batch(img,sem,label,batch_size):
    while True:
        idx = np.arange(0,len(img))
        np.random.shuffle(idx)
        shuf_img = img[idx]
        shuf_sem = sem[idx]
        shuf_lab = label[idx]
        for batch_index in range(0, len(img), batch_size):
            img_batch = shuf_img[batch_index:batch_index + batch_size]
            img_batch = img_batch.astype("float32")
            sem_batch = shuf_sem[batch_index:batch_index + batch_size]
            sem_batch = sem_batch.astype("float32")
            lab_batch = shuf_lab[batch_index:batch_index + batch_size]
            yield img_batch, sem_batch, lab_batch


def kl_for_log_probs(p, q, T):
    log_q = tf.nn.log_softmax(q/T)
    p = tf.nn.softmax(p/T)
    kl = -tf.reduce_mean(tf.reduce_mean(p * log_q, reduction_indices=[1]))
    return kl

