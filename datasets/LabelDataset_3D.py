import nibabel as nib
import sys
import matplotlib.pyplot as plt 
import os
import pandas as pd
import numpy as np
import torch as torch

from torchvision.transforms import functional as tf
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from  concurrent.futures import ThreadPoolExecutor, as_completed
from pytorch_lightning import LightningDataModule
sys.path
import warnings
warnings.simplefilter('ignore')
import random



class ClassImgDataset():
    def __init__(self,root_dir:str, modality_path:str , table_dir,split = 'train') -> None:


        self.root_dir = root_dir
        self.table_dir = table_dir
        self.all_folders, self.all_labels = self.load_sex_table()
        self.modality_path =  modality_path
        self.existing_folders = self.check_paths_and_get_folders(self.all_folders)

        self.len_dataset_all = len(self.existing_folders)
        self.split = split

        #split to finetuning dataset
        seed_everything(33)
        random.shuffle(self.existing_folders)
        #self.existing_folders = random.shuffle(self.existing_folders)
        self.existing_folders = self.existing_folders[int(0.8*self.len_dataset_all):]
        self.len_dataset = len(self.existing_folders)
        self.index_split = (int(0.8*self.len_dataset), int(0.9*self.len_dataset)) 

        # find list of eid's of patients for train val and test set
        assert self.split in ['train','val','test']
        if self.split == 'train':
            self.folders = self.existing_folders[:self.index_split[0]]
            print('Number of train images')

        elif self.split == 'val':
            self.folders = self.existing_folders[self.index_split[0]:self.index_split[1]]
            print('Number of val images')

        elif self.split == 'test':
            self.folders = self.existing_folders[self.index_split[1]:]
            print('Number of test images')
            
        else:
            print('incorrect split variable')
        #load sex labels 
        self.labels = self.all_labels[self.all_labels.index.isin(self.folders)]['Sex'].to_list()
        
        #Load all the images

        self.imgs = self.load_images()


    
    def check_paths_and_get_folders(self,foldernames):
        def check_path(foldername):
            return foldername if os.path.exists(os.path.join(self.root_dir, str(foldername), self.modality_path)) else None
        existing_folders = []
    
        # Use ThreadPoolExecutor to parallelize the path checks
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Create a future for each path check
            futures = {executor.submit(check_path, foldername): foldername for foldername in foldernames}
        
        # As each future completes, process the result
        for future in as_completed(futures):
            result = future.result()
            if result:
                existing_folders.append(result)
        return existing_folders
    

    
    def load_sex_table(self):
        path_demographics = self.table_dir + 'ukb_cardiac_image_subset_Primary demographics.csv'
        df_demo = pd.read_csv(path_demographics)
        df_demo = df_demo.drop(df_demo.index[0])
        df_demo = df_demo.set_index('eid')
        df_demo['Sex'] = df_demo['Sex'].astype(str)
        df_demo = df_demo[['Sex']]

        num_unexpected_sex_val =len(df_demo.loc[~df_demo['Sex'].isin(['0', '1']), 'Sex'].unique())
        if  num_unexpected_sex_val != 0:
            print(f"Number of unexpected Sex sentries that aren't 1 or 0: {num_unexpected_sex_val}")
        return df_demo.index.to_list(), df_demo
    
    def find_min_size(self):
        heights = []
        widths = []
        depths = []
        
        # Load images and store their sizes
        for folder in self.folders:
            path = os.path.join(self.root_dir, str(folder), self.modality_path)
            img = nib.load(path)
            img_array_3d = img.get_fdata()
            img_array_slice = img_array_3d[:, :, 90]
            heights.append(img_array_slice.shape[0])
            widths.append(img_array_slice.shape[1])
            depths.append(img_array_slice.shape[2])
        min_height = min(heights)
        min_width = min(widths)
        min_depth = min(depths)
        return min_height, min_width, min_depth
    
    def load_images(self):
        imgs = []
        halfway = int(len(self.folders)/2)
        for i,folder in enumerate(self.folders):
            if i == halfway:
                print('halfwayish', i)

            path =os.path.join(self.root_dir,str(folder), self.modality_path)
            try:
                img = nib.load(path)
                img_array_3d = img.get_fdata()
                slice_tensor = torch.from_numpy(img_array_3d)
                imgs.append(slice_tensor.type('torch.FloatTensor'))
            except:
                None
        print(self.split, len(imgs))   
        return torch.stack(imgs)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self,idx):
        
        img = self.imgs[idx].unsqueeze(0)
        label = int(self.labels[idx])
        return img, label
    

class ClassImgDataModule3D(LightningDataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.root_dir =  '/vol/biodata/data/biobank/18545/brain_data/'
        self.modality_path = 'T1/T1_brain_to_MNI.nii.gz'
        self.table_dir = '/vol/biodata/data/biobank/18545/downloaded/ukb_45k/'
        self.batch_size = batch_size
        self.train_set = ClassImgDataset(self.root_dir, self.modality_path,self.table_dir, split='train')
        self.val_set = ClassImgDataset(self.root_dir, self.modality_path,self.table_dir,split='val')
        self.test_set = ClassImgDataset(self.root_dir, self.modality_path,self.table_dir,split='test')

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False)
    
