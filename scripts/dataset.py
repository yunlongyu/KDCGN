import numpy as np
import scipy.io as sio
# from data_handler import *

def load_data(data_dir):
    image_embedding = 'res101'
    class_embedding = 'att'

    matcontent = sio.loadmat(data_dir + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
    matcontent = sio.loadmat(data_dir + "/" + class_embedding + "_splits.mat")
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    attribute = matcontent['att'].T

    train_fea = feature[trainval_loc]
    train_lab = label[trainval_loc].astype(int)
    train_sem = attribute[train_lab]
    train_id = np.unique(train_lab)
    train_pro = attribute[train_id]

    train_cla_num = train_id.shape[0]
    train_sam_num = train_lab.shape[0]
    I_eye = np.eye(train_cla_num)
    onehot_label = np.zeros((train_sam_num, train_cla_num))
    for i in range(train_cla_num):
        index = np.where(train_lab==train_id[i])
        onehot_label[index] = I_eye[i,:]

    train_data = {
        'fea': train_fea,
        'lab': train_lab,
        'sem': train_sem,
        'onehot':onehot_label,
        'pro': train_pro,
        'attribute':attribute
    }

    test_seen_fea = feature[test_seen_loc]
    test_seen_label = label[test_seen_loc].astype(int)

    test_seen_sam_num = test_seen_fea.shape[0]
    test_seen_cla_num = train_cla_num
    onehot_label = np.zeros((test_seen_sam_num,test_seen_cla_num))
    for i in range(train_cla_num):
        index = np.where(test_seen_label == train_id[i])
        onehot_label[index] = I_eye[i,:]

    test_seen_data = {
        'fea': test_seen_fea,
        'lab': test_seen_label,
        'onehot':onehot_label,
        'pro': train_pro
    }

    test_unseen_fea = feature[test_unseen_loc]
    test_unseen_label = label[test_unseen_loc].astype(int)
    test_unseen_id = np.unique(test_unseen_label)
    test_unseen_pro = attribute[test_unseen_id]

    unseen_cla_num = test_unseen_id.shape[0]
    unseen_sam_num = test_unseen_label.shape[0]
    I_eye = np.eye(unseen_cla_num)
    test_unseen_onehot = np.zeros((unseen_sam_num, unseen_cla_num))
    for i in range(unseen_cla_num):
        index = np.where(test_unseen_label == test_unseen_id[i])
        test_unseen_onehot[index] = I_eye[i, :]

    test_unseen_data = {
        'fea': test_unseen_fea,
        'onehot': test_unseen_onehot,
        'pro': test_unseen_pro,
        'lab':test_unseen_label
        }
    return train_data, test_seen_data, test_unseen_data


class Dataset(object):
    train_fea = np.array([])
    train_sem = np.array([])
    train_lab = np.array([])
    train_onehot = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def _get_similar_pair(self):
        label = np.random.choice(self.unique_train_label)
        l, r = np.random.choice(self.map_train_label_indices[label],2,replace=False)
        return l, r, 1

    def _get_dissimilar_pair(self):
        label_l, label_r = np.random.choice(self.unique_train_label,2,replace=False)
        l = np.random.choice(self.map_train_label_indices[label_l])
        r = np.random.choice(self.map_train_label_indices[label_r])
        return l, r, 0

    def _get_pair(self):
        if np.random.random() < 0.5:
            return self._get_similar_pair()
        else:
            return self._get_dissimilar_pair()

    def get_batch(self, n):
        idxs_left, idxs_right, labels = [], [], []

        for _ in range(n):
            l, r, x = self._get_pair()
            idxs_left.append(l)
            idxs_right.append(r)
            labels.append(x)
        return self.train_fea[idxs_left,:], self.train_onehot[idxs_left,:], self.train_sem[idxs_right,:], np.expand_dims(labels,axis=1)

class LoadDataset(Dataset):
    def __init__(self, dir):
        # print ("===Loading the dataset===")
        train_data, val_data, test_data = load_data(dir)
        self.train_fea = train_data['fea']               # the train feature
        self.train_sem = train_data['sem']               # the train semantic
        self.train_lab = train_data['lab']               # the train label
        self.train_pro = train_data['pro']             # the train prototype
        self.attribute = train_data['attribute']
        self.train_onehot = train_data['onehot']         # the train onehot label
        self.train_lab = np.expand_dims(self.train_lab,axis=1)
        self.unique_train_label = np.unique(self.train_lab)
        self.map_train_label_indices = {label: np.flatnonzero(self.train_lab == label) for label in self.unique_train_label}
        
        # return train_data, val_data, test_data
        #
        self.test_seen_fea = val_data['fea']                   # the test seen feature
        self.test_seen_idex = val_data['lab']                   # the test seen label
        self.test_seen_onehot = val_data['onehot']             # the test seen onehot label
        self.test_seen_pro = val_data['pro']               # the test seen prototype

        #
        self.test_unseen_fea = test_data['fea']
        self.test_unseen_onehot = test_data['onehot']
        self.test_unseen_pro = test_data['pro']
        self.test_unseen_idex = test_data['lab']

#if __name__== "__main__":
#    data_dir = './data/AwA1_data'
#    rate = 0.9
#    batch_size = 64
#    a = LoadDataset(data_dir,rate)
#    img, lab, sem, sim = a.get_batch(batch_size)


