import numpy as np
from numpy import *
from sklearn.metrics import accuracy_score

def cosine_distance(v1, v2):
    v1_sq = np.inner(v1, v1)
    v2_sq = np.inner(v2, v2)
    dis = 1 - np.inner(v1, v2) / math.sqrt(v1_sq * v2_sq)
    return dis

def kNNClassify(newInput, dataSet, test_id, k):
    numSamples = dataSet.shape[0]

    distance_cos = [0] * numSamples
    for i in range(numSamples):
        distance_cos[i] = cosine_distance(newInput, dataSet[i])

    diff = (tile(newInput, (numSamples, 1)) - dataSet)
    squareDiff = diff ** 2
    squareDist = sum(squareDiff, axis=1)
    distance_euc = squareDist**0.5

    distance = distance_cos
    # sort the distance
    sortedDisIndices = np.argsort(distance)
    classCount = {}  # difine a dictionary

    for i in range(k):
        voteLabel = test_id[sortedDisIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex

def compute_accuracy(att_pre, test_visual, test_idex, test_id):
    outpre = [0] * test_visual.shape[0]
    test_label = np.squeeze(np.asarray(test_idex))
    test_label = test_label.astype("float32")

    for i in range(test_visual.shape[0]):  # CUB 2933
        outputLabel = kNNClassify(test_visual[i, :], att_pre, test_id, 1)
        outpre[i] = outputLabel
    # compute averaged per class accuracy
    outpre = np.array(outpre, dtype='int')
    unique_labels = np.unique(test_label)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(test_label == l)[0]
        acc += accuracy_score(test_label[idx], outpre[idx])
    acc = acc / unique_labels.shape[0]
    return acc


# def compute_accuracy(att_pre, test_visual, test_idex, test_id):
#     outpre = [0]*test_visual.shape[0]  # CUB 2933
#     test_label = np.squeeze(np.asarray(test_idex))
#     test_label = test_label.astype("float32")
#     for i in range(test_visual.shape[0]):  # CUB 2933
#         outputLabel = kNNClassify(test_visual[i,:], att_pre, test_id, 1)
#         outpre[i] = outputLabel
#     #compute averaged per class accuracy
#     outpre = np.array(outpre, dtype='int')
#     unique_labels = np.unique(test_label)
#     acc = 0
#
#     test_id = np.squeeze(np.asarray(test_id))
#     sim = zeros((unique_labels.shape[0],test_id.shape[0]))
#     for i in range(0,unique_labels.shape[0]):
#         idx = np.nonzero(test_label == unique_labels[i])[0]
#         acc += accuracy_score(test_label[idx], outpre[idx])
#         for j in range(0,test_id.shape[0]):
#             idex = np.nonzero(outpre[idx]==test_id[j])[0]
#             sim_ = float(idex.shape[0]*1000/test_label[idx].shape[0])
#             sim[i,j] = sim_/1000
#
#     acc = acc / unique_labels.shape[0]
#     return sim, acc

class Test_nn():
    def __init__(self, sess, img, pro, hid_img, f_out):
        self.img = img
        self.pro = pro
        self.hid_img = hid_img
        self.f_out = f_out
        self.sess = sess

    def test_zsl(self, data):
        attribute = data.attribute
        test_fea = data.test_unseen_fea
        test_idx = data.test_unseen_idex
        test_id = np.unique(test_idx)
        test_pro = attribute[test_id]

        hid_fea, att_pre = self.sess.run([self.hid_img, self.f_out], feed_dict={self.img:test_fea, self.pro: test_pro})
        acc = compute_accuracy(att_pre, hid_fea, test_idx, test_id)
        return acc

    def test_seen(self, data):
        attribute = data.attribute
        cla_num = attribute.shape[0]
        test_fea = data.test_seen_fea #4958 x 2048
        test_idex = data.test_seen_idex
        test_id = np.arange(cla_num)
        test_pro = attribute[test_id]

        hid_fea, att_pre = self.sess.run([self.hid_img, self.f_out], feed_dict={self.img:test_fea, self.pro: test_pro})
        acc = compute_accuracy(att_pre, hid_fea, test_idex, test_id)
        return acc

    def test_unseen(self, data):
        attribute = data.attribute
        cla_num = attribute.shape[0]
        test_fea = data.test_unseen_fea  # 5685 x 2048
        test_idex = data.test_unseen_idex
        test_id = np.arange(cla_num)
        test_pro = attribute[test_id]

        hid_fea, att_pre = self.sess.run([self.hid_img, self.f_out], feed_dict={self.img:test_fea, self.pro: test_pro})
        acc = compute_accuracy(att_pre, hid_fea, test_idex,test_id)
        return acc
