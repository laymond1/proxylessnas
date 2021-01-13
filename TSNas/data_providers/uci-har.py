# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import *

def Read_Data(dataset, input_nc):
    data_path = os.path.join('Data', 'preprocessed', dataset)
    train_X = np.load(data_path+'/train_x.npy')
    train_Y = np.load(data_path+'/train_y.npy')
    test_X = np.load(data_path+'/test_x.npy')
    test_Y = np.load(data_path+'/test_y.npy')

    return To_DataSet(train_X, train_Y), To_DataSet(test_X, test_Y), test_Y

class To_DataSet(Dataset):
    def __init__(self, X, Y):
        self.data_num = Y.shape[0]
        self.x = torch.as_tensor(X)
        self.y = torch.as_tensor(Y)#torch.max(torch.as_tensor(Y), 1)[1]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.data_num

class HCIHARDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=128, test_batch_size=128, valid_size=None,
                n_worker=0):

        self._save_path = save_path # internal variable
        # train_transforms = self.build_train_transform() # not yet
        train_dataset = 

    @staticmethod
    def name():
        return 'UCI-HAR'

    @property
    def data_shape(self):
        return 6, self.data_size # C, length

    @property
    def n_classes(self):
        return 6

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/dataset/uci_har'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download UCI-HAR')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def normalize(self):
        """ Minmax normalize the data """
        raise NotImplementedError


    def build_train_transform(self, distort_color, resize_scale):


    @property
    def data_size(self):
        return 224