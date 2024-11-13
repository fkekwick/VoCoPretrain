import nibabel as nib
import sys
import matplotlib.pyplot as plt 
import os
import pandas as pd
import numpy as np
import torch as torch

from torchvision.transforms import functional as tf
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, LightningDataModule,  seed_everything
from  concurrent.futures import ThreadPoolExecutor, as_completed
from pytorch_lightning import LightningDataModule
sys.path
import warnings
warnings.simplefilter('ignore')
import random





class ClassImgDataset():
    def __init__(self,root_dir:str, modality_path:str , table_dir,split = 'train', percent_split=0.7, target_size=(182,216,182)) -> None:
    
        self.root_dir = root_dir
        self.table_dir = table_dir

        self.all_folders, self.all_labels = self.load_sex_table()
        self.modality_path =  modality_path
        self.existing_folders = self.check_paths_and_get_folders(self.all_folders)
        self.len_dataset_all = len(self.existing_folders)
        print('whole dataset size', self.len_dataset_all)

        #split to pretraining dataset
        seed_everything(33)
        random.shuffle(self.existing_folders)
        self.existing_folders = self.existing_folders[:int(percent_split*self.len_dataset_all)]
        self.len_dataset = len(self.existing_folders)
        self.split = split
        self.index_split = (int(0.85*self.len_dataset), int(0.87*self.len_dataset)) 
        self.target_size = target_size

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
        print(len(self.folders))

        self.labels_table =  self.all_labels[self.all_labels.index.isin(self.folders)]
        self.paths, self.labels, self.corrupted_folders, self.usable_folders = self.create_paths()
        assert len(self.labels) == len(self.paths)
        print('Number of ',self.split,'images ', len(self.paths),'number of ', self.split, 'labels' )
        

    
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
    
    def create_paths(self):
        paths = []
        labels = []
        f=[]
        corrupted_folders = []
        for folder in self.folders:
            path =os.path.join(self.root_dir,str(folder), self.modality_path)
            try:
                img = nib.load(path)
                label = self.labels_table.loc[folder]
                del img
                paths.append(path)
                labels.append(label)
                f.appen(folder)
            except:
                corrupted_folders.append(folder)
        
        return paths, labels, corrupted_folders, f

    
    def load_sex_table(self):
        path_demographics = os.path.join( self.table_dir, 'ukb_cardiac_image_subset_Primary demographics.csv')
        df_demo = pd.read_csv(path_demographics)
        df_demo = df_demo.drop(df_demo.index[0])
        df_demo = df_demo.set_index('eid')
        df_demo['Sex'] = df_demo['Sex'].astype(str)
        df_demo = df_demo[['Sex']]

        num_unexpected_sex_val =len(df_demo.loc[~df_demo['Sex'].isin(['0', '1']), 'Sex'].unique())
        if  num_unexpected_sex_val != 0:
            print(f"Number of unexpected Sex sentries that aren't 1 or 0: {num_unexpected_sex_val}")
        return df_demo.index.to_list(), df_demo

    

    



    
    def center_crop_3d(self, img, target_size):
        """Center crop a 3D image to the target size (depth, height, width)."""
        current_depth, current_height, current_width = img.shape

        depth_crop = (current_depth - target_size[0]) // 2
        height_crop = (current_height - target_size[1]) // 2
        width_crop = (current_width - target_size[2]) // 2

        return img[
            depth_crop:depth_crop + target_size[0],
            height_crop:height_crop + target_size[1],
            width_crop:width_crop + target_size[2]
        ]
    
    def reduce_slices(self,img, f_w, f_d):
        f_h = f_w
        image = img[0::f_h,0::f_w,0::f_d]

        return image


    def __len__(self):   
        l= len(self.paths)
        return l
    


    def __getitem__(self,idx):
        #load image from file
        path = self.paths[idx]
        img = nib.load(path)

        img = img.get_fdata() #[:, 1:217, :]

        img = self.center_crop_3d(img, self.target_size)
        img = self.reduce_slices(img, 2,2)

        img = torch.from_numpy(img).type('torch.FloatTensor').unsqueeze(0)

        return img



class MAEImgDataModule3D(LightningDataModule):
    def __init__(self, batch_size: int = 4,
                  paths = {    'root_dir':  '/data2/biodata/biobank/brain_data/',
        'table_dir' : '/data2/biodata/biobank/tables/'},num_workers=3,
                  **kwargs):
        super().__init__()
        self.root_dir =  paths['root_dir']
        self.modality_path = 'T1/T1_brain_to_MNI.nii.gz'
        self.table_dir = paths['table_dir']
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = ClassImgDataset(self.root_dir, self.modality_path,self.table_dir, split='train',**kwargs)
        self.val_set = ClassImgDataset(self.root_dir, self.modality_path,self.table_dir,split='val',**kwargs)
        self.test_set = ClassImgDataset(self.root_dir, self.modality_path,self.table_dir,split='test',**kwargs)
        self.corrupted = self.train_set.corrupted_folders + self.val_set.corrupted_folders + self.test_set.corrupted_folders
        self.all_folders = self.train_set.usable_folders + self.val_set.usable_folders +self.test_set.usable_folders


        

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
