from medmnist.info import INFO
from medmnist.dataset import MedMNIST3D
from torchvision import transforms
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch import from_numpy
import torchio as tio
import torch

class PretrainNoduleMNISTDataset(MedMNIST3D):
    def __init__(self, split = 'train'):
        ''' Dataset class for NoduleMNIST.
        The provided init function will automatically download the necessary
        files at the first class initialistion.

        :param split: 'train', 'val' or 'test', select subset

        '''
        self.flag = "nodulemnist3d"
        self.size = 64
        self.size_flag = ""
        self.root = './datasets/'
        self.info = INFO[self.flag]
        self.download()

        npz_file = np.load(os.path.join(self.root, "nodulemnist3d.npz"))

        self.split = split

        # Load all the images
        assert self.split in ['train','val','test']
        self.imgs = npz_file[f'{self.split}_images']
        self.labels = npz_file[f'{self.split}_labels']

        if self.split == 'test':
            self.imgs = self.imgs[0:int(0.7*len(self.imgs))]
            self.labels = self.labels[0:int(0.7*len(self.labels))]
            
        print(len(self.imgs), 'the number of ', self.split, 'images' )


        # data augmentation pipeline (description is in cell below)
        #            transforms.ToPILImage()
        self.augmentation_pipeline = tio.Compose([
                tio.transforms.RandomElasticDeformation(p=0.2),
                tio.transforms.RandomMotion(p=0.5),
                tio.transforms.RandomBlur(p=0.2),
                tio.transforms.RandomNoise(p=0.2),
                tio.transforms.RandomAffine(p=0.4,degrees=0,scales=(0.8, 1.2))])
                # transforms.ToTensor(),
                # transforms.Normalize(mean=0.5,std=0.5)
        self.preprocessing_pipeline = transforms.Compose([ transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(mean=0.5,std=0.5)])


    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        #retrieve image and its label
        img=self.imgs[index]
        # label=self.labels[index]
        #
        img = img[np.newaxis]
        #apply the augmentation pipeline to the image
        img_view1=self.augmentation_pipeline(img)
        l = []
        for i, element in enumerate(img_view1[0]):
            l.append(self.preprocessing_pipeline(element).squeeze(0))
        img_view1 = torch.stack(l).unsqueeze(dim=0).type('torch.FloatTensor')

        img_view2=self.augmentation_pipeline(img)
        l = []
        for i, element in enumerate(img_view2[0]):
            l.append(self.preprocessing_pipeline(element).squeeze(0))
        img_view2 = torch.stack(l, dim=0).unsqueeze(dim=0).type('torch.FloatTensor')

        
        #return the augmented views
        return img_view1, img_view2
    
class PretrainNoduleMNISTDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.train_set = PretrainNoduleMNISTDataset(split='train')
        self.val_set = PretrainNoduleMNISTDataset(split='val')
        self.test_set = PretrainNoduleMNISTDataset(split='test')

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False)





#labelled datastet
class Nodule3DMNISTDataset(MedMNIST3D):
    def __init__(self, split = 'train', augmentation: bool = False):
        ''' Dataset class for 3D Nodule MNST.
        The provided init function will automatically download the necessary
        files at the first class initialistion.

        :param split: 'train', 'val' or 'test', select subset

        '''
        self.flag = "nodulemnist3d"
        self.size = 64
        self.size_flag = ""
        self.root = './dataset/'
        self.info = INFO[self.flag]
        self.download()

        npz_file = np.load(os.path.join(self.root, "nodulemnist3d.npz"))
        self.split = split

        # Load all the images
        assert self.split in ['train','val','test']
        self.imgs = npz_file[f'{self.split}_images']
        self.labels = npz_file[f'{self.split}_labels']

        if self.split == 'test':
            None 
        else:
            self.imgs = self.imgs[int(0.8*len(self.imgs)):]
            self.labels = self.labels[int(0.8*len(self.imgs)):]
        print(len(self.imgs), 'the number of ', self.split, 'images' )

        self.preprocessing_pipeline = transforms.Compose([ transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(mean=0.5,std=0.5)])

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):

        #retrieve image and its label
        img=self.imgs[index]
        label=self.labels[index]

        #apply the augmentation pipeline to the image
        l = []
        for i, element in enumerate(img):
            l.append(self.preprocessing_pipeline(element).squeeze(0))
        img_view1 = torch.stack(l).unsqueeze(dim=0).type('torch.FloatTensor')

        return img_view1 #, int(label[0])
    

    
class Nodule3DMNISTDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 16):
        super().__init__()
        self.batch_size = batch_size
        self.train_set = Nodule3DMNISTDataset(split='train', augmentation=True)
        self.val_set = Nodule3DMNISTDataset(split='val', augmentation=False)
        self.test_set = Nodule3DMNISTDataset(split='test', augmentation=False)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False)