"""
author:hao
date:16/10/2020

point cloud data preprocessing
"""

from traceback import print_list
from numpy.core.function_base import add_newdoc
import pandas as pd
import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def get_kflod_complex(k,i,X):
    fold_size = len(X) // k
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid = X[val_start:val_end]
        X_train =  X[0:val_start]+X[val_end:]
    else:
        X_valid = X[val_start:]
        X_train = X[0:val_start]

    return X_train,X_valid

def get_kfold_data(k, i, X, y):  
    fold_size = len(X) // k  
    
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = X[0:val_start]+X[val_end:]
        y_train = y[0:val_start]+y[val_end:]
    else:
        X_valid, y_valid = X[val_start:], y[val_start:]     
        X_train = X[0:val_start]
        y_train = y[0:val_start]
        
    return X_train, y_train, X_valid,y_valid

def split_train_valid(data,train=0.9):
    train_size = int(train*len(data))
    train_set = data[:train_size]
    valid_set = data[train_size:len(data)]
    return train_set,valid_set

def pc_normalize(pc):
    """
    Normalize coords for all sample points 
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def interface_nearest_point_sample(point,npoint):
    R= point[point[:,-1]==0]
    L= point[point[:,-1]==1]

    num_r = R.shape[0]
    num_l = L.shape[0]

    distances = []
    dicts = []
    index_r = []
    index_l = []
    
    def euclidean(x, y):
        return np.sqrt(np.sum((x - y)**2))

    for i in range(num_r):
        r = R[i]
        for j in range(num_l):
            l = L[j]
            d = euclidean(r[:3],l[:3])
            distances.append(d)
            dicts.append([i,j,d])
    
    dicts = np.array(dicts)

    dicts = dicts[dicts[:,2].argsort()]

    for i in range(dicts.shape[0]):
        r = dicts[i,0]
        l = dicts[i,1]
        if (r not in index_r) and len(index_r) < npoint//2:
            index_r.append(r)
        if (l not in index_l) and len(index_l) < npoint//2:
            index_l.append(l)
    
    indexs = index_r+index_l
    indexs = [int(x) for x in indexs]

    point = point[indexs]

    return point

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class PPI4DOCKDataLoader(Dataset):
    def __init__(self, root,  npoint=720, split='train', uniform=False, normal_channel=True, cache_size=15000,use_ph=False):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'class.txt')
        self.use_ph =use_ph
        if self.use_ph:
            ph_table = pd.read_csv(os.path.join(self.root, 'physicochemical_class_onehot.csv'),index_col=0)
            self.ph_table = ph_table.values

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        #print(self.classes['postive'])
        self.normal_channel = normal_channel

        data_paths = {}
        #data_paths['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'train.txt'))]
        #data_paths['valid'] = [line.rstrip() for line in open(os.path.join(self.root, 'valid.txt'))]
        #data_paths['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'test.txt'))]

        data_paths['train'] = [os.path.join(self.root,line.rstrip()) for line in open(os.path.join(self.root, 'train.txt'))]
        data_paths['valid'] = [os.path.join(self.root,line.rstrip()) for line in open(os.path.join(self.root, 'valid.txt'))]
        data_paths['test'] = [os.path.join(self.root,line.rstrip()) for line in open(os.path.join(self.root, 'test.txt'))]


        assert (split == 'train' or split == 'valid' or split == 'test')
        #shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        #self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         #in range(len(shape_ids[split]))]
        self.datapath = data_paths[split]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        data = []
        labels = []
        for index in range(len(self.datapath)):
            path_and_index = self.datapath[index]
            path_and_index = path_and_index.split(' ')
            path = path_and_index[0]
            label = int(path_and_index[1])
            point_set = np.loadtxt(path, delimiter=' ').astype(np.float32) #[N,3+4+20+2]
            # point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if self.use_ph:
                npoints = point_set.shape[0]
                table = np.tile(self.ph_table,[npoints,1,1])
                res_f = point_set[:,7:27]
                ph_f = table[res_f==1]
                a = point_set[:,0:27]
                b = point_set[:,[27,28]]
                point_set = np.concatenate((a,ph_f,b),axis=1) #29+6=35
            
            data.append(point_set)
            labels.append(label)

        self.data = data
        self.labels = labels

    def get_labels(self): return self.labels

    def get_size(self):
        return len(self.datapath)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        point_set = self.data[index]
        label = self.labels[index]

        return point_set,label

class DataLoaderForSplit(Dataset):
    """
    Dataloader for Split in Cross validation
    """
    def __init__(self,root,input_complex,all_data_path,use_res=True):
        self.complex = input_complex
        self.all_data_path = all_data_path
        self.root = root
        self.use_res = use_res
        
        data_path = []
        complex_names = set()
        for dp in self.all_data_path:
            complex_name = dp.split('/')[0]
            if (complex_name[:4] in self.complex):
                data_path.append(os.path.join(self.root,dp))
                complex_names.add(complex_name)
        
        self.complex = list(complex_names)
        self.datapath = data_path

        data = []
        labels = []
        for index in range(len(self.datapath)):
            path_and_index = self.datapath[index]
            path_and_index = path_and_index.split(' ')
            path = path_and_index[0]
            label = int(path_and_index[1])
            point_set = np.loadtxt(path, delimiter=' ').astype(np.float32) #[N,3+4+20+2]
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            if not self.use_res:
                point_set = np.concatenate((point_set[:, 0:7],point_set[:, -2:]),axis=1) #only atom        
            data.append(point_set)
            labels.append(label)

        self.data = data
        self.labels = labels
    def get_labels(self): return self.labels

    def get_size(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def get_complex_names(self):
        return self.complex

    def __getitem__(self, index):
        point_set = self.data[index]
        label = self.labels[index]

        return point_set,label

class DataLoaderForCV(Dataset):
    """
    Dataloader for Cross validation
    """
    def __init__(self,root, npoint=1000,use_res=True):
        self.root = root
        self.npoints = npoint
        self.catfile = os.path.join(self.root, 'class.txt')
        self.use_res= use_res

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        all_complex_unique = []
        if not os.path.exists(os.path.join(self.root, 'all.txt')):

            all_complex = {}
            data_paths = {}
            data_paths['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'train.txt'))]
            data_paths['valid'] = [line.rstrip() for line in open(os.path.join(self.root, 'valid.txt'))]
            data_paths['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'test.txt'))]
            data_path = data_paths['train']+data_paths['valid']+data_paths['test']
            self.datapath = data_path
            all_complex['train'] = [line.split('/')[0] for line in open(os.path.join(self.root, 'train.txt'))]
            all_complex['valid'] = [line.split('/')[0] for line in open(os.path.join(self.root, 'valid.txt'))]
            all_complex['test'] = [line.split('/')[0] for line in open(os.path.join(self.root, 'test.txt'))]

            complex = all_complex['train']+all_complex['valid']+all_complex['test']
            for c in complex:
                if c not in all_complex_unique:
                    all_complex_unique.append(c)
            self.all_complex_unique = all_complex_unique
        else:
            data_path = [line.rstrip() for line in open(os.path.join(self.root, 'all.txt'))]
            complex = [line.split('/')[0] for line in open(os.path.join(self.root, 'all.txt'))]

            self.datapath = data_path
            for c in complex:
                if c not in all_complex_unique:
                    all_complex_unique.append(c)
            self.all_complex_unique = all_complex_unique

        print('The size of all data is %d and have %d complexes'%(len(self.datapath),len(self.all_complex_unique)))

    def get_labels(self): return self.labels

    def get_size(self):
        return len(self.datapath)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        pass
    
    def get_cv_dataloader_use_txt(self,txt_dir,i):
        """
        use preset to split complexes
        i:fold num [1~4]
        """
        complexs = []
        with open(txt_dir) as f:
            line = f.readline().strip()
            while line:
                complexs.append(line.split(','))
                line = f.readline().strip()
        k = len(complexs)
        complex_test= complexs[i]
        complex_train = []
        for j in range(k):
            if j!=i-1:
                complex_train+=complexs[j]
        
        test_dataloader = DataLoaderForSplit(self.root,complex_test,self.datapath,self.use_res)
        complex_train,complex_valid= split_train_valid(complex_train)

        train_dataloader = DataLoaderForSplit(self.root,complex_train,self.datapath,self.use_res)
        valid_dataloader = DataLoaderForSplit(self.root,complex_valid,self.datapath,self.use_res)

        print('In fold {}/{} ,The size of train size is {} and have {} complex'.format(i,k,train_dataloader.get_size(),len(complex_train)))
        print('In fold {}/{} ,The size of valid size is {} and have {} complex'.format(i,k,valid_dataloader.get_size(),len(complex_valid)))
        print('In fold {}/{} ,The size of test size is {} and have {} complex'.format(i,k,test_dataloader.get_size(),len(complex_test)))

        return train_dataloader,valid_dataloader,test_dataloader


    def get_cv_dataloader(self,k,i):
        """
        use random function to split complexes
        i:fold num [1~4]
        """
        complex_train, complex_test = get_kflod_complex(k,i-1,self.all_complex_unique)
        test_dataloader = DataLoaderForSplit(self.root,complex_test,self.datapath,self.use_res)

        complex_train,complex_valid= split_train_valid(complex_train)

        train_dataloader = DataLoaderForSplit(self.root,complex_train,self.datapath,self.use_res)
        valid_dataloader = DataLoaderForSplit(self.root,complex_valid,self.datapath,self.use_res)

        print('In fold {}/{} ,The size of train size is {} and have {} complex'.format(i,k,train_dataloader.get_size(),len(complex_train)))
        print('In fold {}/{} ,The size of valid size is {} and have {} complex'.format(i,k,valid_dataloader.get_size(),len(complex_valid)))
        print('In fold {}/{} ,The size of test size is {} and have {} complex'.format(i,k,test_dataloader.get_size(),len(complex_test)))

        return train_dataloader,valid_dataloader,test_dataloader

    def __getitem__(self, index):
        point_set = self.data[index]
        label = self.labels[index]

        return point_set,label

class DataLoaderForHitOnComplex(Dataset):
    """
    Dataloader for Hit (load result path)
    """
    def __init__(self, root,  npoint=1000, use_res=True):
        self.root = root
        print(self.root)
        self.npoints = npoint
        self.use_res = use_res

        self.file_list = os.listdir(self.root)
        self.datapath = [os.path.join(self.root,line.rstrip()) for line in self.file_list]
        print('The size of data is %d'%(len(self.datapath)))

        self.cache_size = 15000  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple
    
    def get_size(self):
        return len(self.datapath)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set,label = self.cache[index]
        else:
            label = -1
            path = self.datapath[index]
            point_set = np.loadtxt(path, delimiter=' ').astype(np.float32)
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])


            if not self.use_res:
                point_set = np.concatenate((point_set[:, 0:7],point_set[:, -2:]),axis=1) #only atom        

            if len(self.cache) < self.cache_size:
                self.cache[index] = point_set,label

        return point_set,label,self.datapath[index]

    def __getitem__(self, index):
        return self._get_item(index)

def collate_fn(batch):
    f_num = batch[0][0].shape[1]
    max_natoms = 1000
    labels = []
    point_set = np.zeros((len(batch), max_natoms, f_num))

    for i in range(len(batch)):
        item = batch[i]
        num_atom = item[0].shape[0]
        point_set[i,:num_atom] = item[0]
        labels.append(item[1])
    
    labels = np.array(labels)

    return point_set,labels,max_natoms


def collate_fn_hit(batch):
    f_num = batch[0][0].shape[1]
    max_natoms = 1000
    labels = []
    paths = []
    point_set = np.zeros((len(batch), max_natoms, f_num))

    for i in range(len(batch)):
        item = batch[i]
        num_atom = item[0].shape[0]
        point_set[i,:num_atom] = item[0]
        labels.append(item[1])
        paths.append(item[2])
    
    labels = np.array(labels)

    return point_set,labels,max_natoms,paths


if __name__ == '__main__':
    import torch
    data_all = DataLoaderForCV('C:\\Users\\cxhrzh\\Desktop\\PointScore\\Data\\dockground61_1000', npoint=1000)
    TRAIN_DATASET ,VALID_DATASET ,TEST_DATASET = data_all.get_cv_dataloader_use_txt('C:\\Users\\cxhrzh\\Desktop\\PointScore\\dockground_split.txt',1)
