import nibabel as nib
import sys
import matplotlib.pyplot as plt 
import os
import pandas as pd
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

from sklearn.preprocessing import StandardScaler
import numpy as np



class SexClassImgDataset():
    def __init__(self,root_dir:str, modality_path:str , table_dir,split = 'train', percent_split= (0.75,0.795,0.8)) -> None:
        seed_everything(33)    



        self.root_dir = root_dir
        self.table_dir = table_dir

        self.all_folders, self.all_labels = self.load_sex_table()
        self.modality_path =  modality_path
        self.existing_folders = self.check_paths_and_get_folders(self.all_folders)
        self.len_dataset_all = len(self.existing_folders)
        print('whole dataset size', self.len_dataset_all)

         #split to pretraining dataset
       
        random.shuffle(self.existing_folders)
        # self.existing_folders = self.existing_folders[(1-int(percent_split*self.len_dataset_all)):]
        self.len_dataset = len(self.existing_folders)
        self.split = split
        self.index_split = (int(percent_split[0]*self.len_dataset),int(percent_split[1]*self.len_dataset), int(percent_split[2]*self.len_dataset)) 

        # find list of eid's of patients for train val and test set
        assert self.split in ['train','val','test']
        if self.split == 'train':
            self.folders = self.existing_folders[self.index_split[0]:self.index_split[1]]   

        elif self.split == 'val':
            self.folders = self.existing_folders[self.index_split[1]:self.index_split[2]]

        elif self.split == 'test':
            self.folders = self.existing_folders[self.index_split[2]:]
            
        else:
            print('incorrect split variable')
            

        self.labels_table = self.all_labels[self.all_labels.index.isin(self.folders)]
        self.paths, self.labels, self.corrupted_folders = self.create_paths()

        assert len(self.labels) == len(self.paths)
        assert type(self.labels[0]) == int
        print('Number of ',self.split,'images ', len(self.paths),'number of ', self.split, 'labels' , 'corrupted', len(self.corrupted_folders))
        

    
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
                label = self.labels_table.loc[folder, 'Sex']
                del img
                paths.append(path)
                labels.append((int(label)))
                # f.append(folder)
            except:
                corrupted_folders.append(folder)
        
        return paths, labels, corrupted_folders

    
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
    

    



    def __len__(self):   
        l= len(self.paths)
        return l
    


    def __getitem__(self,idx):
        #load image from file
        path = self.paths[idx]
        img = nib.load(path)
        img = img.get_fdata()
        img = torch.from_numpy(img).type('torch.FloatTensor').unsqueeze(0)
        label = self.labels[idx] 

        return img, label



class SexClassImgDataModule3D(LightningDataModule):
    def __init__(self, batch_size: int = 4,
                  paths = {    'root_dir':  '/data2/biodata/biobank/brain_data/',
        'table_dir' : '/data2/biodata/biobank/tables/'},num_workers=2,
                  **kwargs):
        super().__init__()
        self.root_dir =  paths['root_dir']
        self.modality_path = 'T1/T1_brain_to_MNI.nii.gz'
        self.table_dir = paths['table_dir']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set =SexClassImgDataset(self.root_dir, self.modality_path,self.table_dir, split='train',**kwargs)
        self.val_set =SexClassImgDataset(self.root_dir, self.modality_path,self.table_dir,split='val',**kwargs)
        self.test_set =SexClassImgDataset(self.root_dir, self.modality_path,self.table_dir,split='test',**kwargs)
        self.corrupted = self.train_set.corrupted_folders + self.val_set.corrupted_folders + self.test_set.corrupted_folders
        # self.all_folders = self.train_set.usable_folders + self.val_set.usable_folders +self.test_set.usable_folders


        seed_everything(33)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)










class AgeClassImgDataset():
    def __init__(self, root_dir: str, modality_path: str, table_dir, split='train',  percent_split= (0.75,0.795,0.8), scaler=None) -> None:
        seed_everything(33)

        self.root_dir = root_dir
        self.table_dir = table_dir
        self.all_folders, self.all_labels = self.load_sex_table()

        self.modality_path = modality_path
        self.existing_folders = self.check_paths_and_get_folders(self.all_folders)
        self.len_dataset_all = len(self.existing_folders)
        print('whole dataset size', self.len_dataset_all)

        random.shuffle(self.existing_folders)
        self.len_dataset = len(self.existing_folders)
        self.split = split
        self.index_split = (int(percent_split[0]*self.len_dataset),int(percent_split[1]*self.len_dataset), int(percent_split[2]*self.len_dataset)) 

        assert self.split in ['train', 'val', 'test']
        if self.split == 'train':
            self.folders = self.existing_folders[self.index_split[0]:self.index_split[1]]
        elif self.split == 'val':
            self.folders = self.existing_folders[self.index_split[1]:self.index_split[2]]
        elif self.split == 'test':
            self.folders = self.existing_folders[self.index_split[2]:]
        else:
            print('incorrect split variable')

        self.labels_table = self.all_labels[self.all_labels.index.isin(self.folders)]
        self.paths, self.labels, self.corrupted_folders = self.create_paths()
        assert len(self.labels) == len(self.paths)

        print(split, len(self.labels))


        if self.split == 'train':
            self.scaler = StandardScaler()
            self.labels = self.scaler.fit_transform(np.array(self.labels).reshape(-1, 1)).flatten()
        else:
            # Use the scaler fitted on the training data to transform labels
            print(len(self.labels))
            self.labels = scaler.transform(np.array(self.labels).reshape(-1, 1)).flatten()

        print('Number of ', self.split, 'images ', len(self.paths), 'number of ', self.split, 'labels', len(self.labels), 'corrupted', len(self.corrupted_folders))


    def check_paths_and_get_folders(self, foldernames):
        def check_path(foldername):
            return foldername if os.path.exists(os.path.join(self.root_dir, str(foldername), self.modality_path)) else None
        existing_folders = []

        # Use ThreadPoolExecutor to parallelize the path checks
        with ThreadPoolExecutor(max_workers=13) as executor:
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
        corrupted_folders = []
        for folder in self.folders:
            path = os.path.join(self.root_dir, str(folder), self.modality_path)

            try:
                img = nib.load(path)
                label = self.labels_table.loc[folder, 'Age when attended assessment centre.2']
                del img
                paths.append(path)
                labels.append(float(label))

            except Exception as e:
                print(f"Error loading {path}: {e}")
                corrupted_folders.append(folder)

        return paths, np.array(labels), corrupted_folders

    def load_sex_table(self):
        path_demographics = os.path.join(self.table_dir, 'ukb_cardiac_image_subset_Primary demographics.csv')
        df_demo = pd.read_csv(path_demographics)
        df_demo = df_demo.drop(df_demo.index[0])
        df_demo = df_demo.set_index('eid')
        df_demo = df_demo.dropna(subset=['Age when attended assessment centre.2'])

        df_demo['Age when attended assessment centre.2'] = df_demo['Age when attended assessment centre.2'].astype(str)
        df_demo = df_demo[['Age when attended assessment centre.2']]

        return df_demo.index.to_list(), df_demo

    def __len__(self):   
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = nib.load(path)
        img = img.get_fdata()
        img = torch.from_numpy(img).type('torch.FloatTensor').unsqueeze(0)
        label = self.labels[idx]

        return img, label


class AgeClassImgDataModule3D(LightningDataModule):
    def __init__(self, batch_size: int = 4,
                 paths={'root_dir': '/data2/biodata/biobank/brain_data/',
                        'table_dir': '/data2/biodata/biobank/tables/'}, num_workers=2,
                 **kwargs):
        super().__init__()
        self.root_dir = paths['root_dir']
        self.modality_path = 'T1/T1_brain_to_MNI.nii.gz'
        self.table_dir = paths['table_dir']
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize train dataset to compute scaler
        self.train_set = AgeClassImgDataset(self.root_dir, self.modality_path, self.table_dir, split='train', **kwargs)
        
        # Store the scaler used in training
        self.scaler = self.train_set.scaler
        
        # Initialize validation and test datasets using the same scaler
        self.val_set = AgeClassImgDataset(self.root_dir, self.modality_path, self.table_dir, split='val', scaler=self.scaler, **kwargs)
        self.test_set = AgeClassImgDataset(self.root_dir, self.modality_path, self.table_dir, split='test', scaler=self.scaler, **kwargs)



        self.corrupted = self.train_set.corrupted_folders + self.val_set.corrupted_folders + self.test_set.corrupted_folders
        # self.all_folders = self.train_set.usable_folders + self.val_set.usable_folders + self.test_set.usable_folders

        seed_everything(33)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)


class RawT1AgeClassImgDataset():
    def __init__(self, root_dir: str, modality_path: str, table_dir, split='train',  percent_split= (0.75,0.795,0.8), scaler=None) -> None:
        seed_everything(33)

        self.root_dir = root_dir
        self.table_dir = table_dir
        self.all_folders, self.all_labels = self.load_sex_table()

        self.modality_path = modality_path

        self.existing_folders = self.check_paths_and_get_folders(self.all_folders)

        self.len_dataset_all = len(self.existing_folders)
        print('whole dataset size', self.len_dataset_all)

        random.shuffle(self.existing_folders)
        self.len_dataset = len(self.existing_folders)
        self.split = split
        self.index_split = (int(percent_split[0]*self.len_dataset),int(percent_split[1]*self.len_dataset), int(percent_split[2]*self.len_dataset)) 

        assert self.split in ['train', 'val', 'test']
        if self.split == 'train':
            self.folders = self.existing_folders[self.index_split[0]:self.index_split[1]]
        elif self.split == 'val':
            self.folders = self.existing_folders[self.index_split[1]:self.index_split[2]]
        elif self.split == 'test':
            self.folders = self.existing_folders[self.index_split[2]:]
        else:
            print('incorrect split variable')

        self.labels_table = self.all_labels[self.all_labels.index.isin(self.folders)]
        self.paths, self.labels, self.corrupted_folders, self.folders = self.create_paths()
        assert len(self.labels) == len(self.paths)


        if self.split == 'train':
            self.scaler = StandardScaler()
            self.labels = self.scaler.fit_transform(np.array(self.labels).reshape(-1, 1)).flatten()
        else:
            # Use the scaler fitted on the training data to transform labels
            self.labels = scaler.transform(np.array(self.labels).reshape(-1, 1)).flatten()

        print('Number of ', self.split, 'images ', len(self.paths), 'number of ', self.split, 'labels', len(self.labels), 'corrupted', len(self.corrupted_folders))


    def check_paths_and_get_folders(self, foldernames):
        def check_path(foldername):
            return foldername if os.path.exists(os.path.join(self.root_dir, str(foldername), self.modality_path)) else None
            
        existing_folders = []

        # Use ThreadPoolExecutor to parallelize the path checks
        with ThreadPoolExecutor(max_workers=15) as executor:
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
        corrupted_folders = []
        folders=[]
        for folder in self.folders:
            path = os.path.join(self.root_dir, str(folder), self.modality_path)

            try:
                img = nib.load(path)
                label = self.labels_table.loc[folder, 'Age when attended assessment centre.2']
                del img
                paths.append(path)
                labels.append(float(label))
                folders.append(folder)

            except Exception as e:
                print(f"Error loading {path}: {e}")
                corrupted_folders.append(folder)

        return paths, np.array(labels), corrupted_folders, folders

    def load_sex_table(self):
        path_demographics = os.path.join(self.table_dir, 'ukb_cardiac_image_subset_Primary demographics.csv')
        df_demo = pd.read_csv(path_demographics)
        df_demo = df_demo.drop(df_demo.index[0])
        df_demo = df_demo.set_index('eid')
        df_demo = df_demo.dropna(subset=['Age when attended assessment centre.2'])

        df_demo['Age when attended assessment centre.2'] = df_demo['Age when attended assessment centre.2'].astype(str)
        df_demo = df_demo[['Age when attended assessment centre.2']]

        print(len(df_demo.index.to_list()))
        return df_demo.index.to_list(), df_demo


    def pad_3d(self, img, target_size):
        """Pads a 3D image to the target size (depth, height, width)."""
        current_depth, current_height, current_width = img.shape

        pad_depth = max(target_size[0] - current_depth, 0)
        pad_height = max(target_size[1] - current_height, 0)
        pad_width = max(target_size[2] - current_width, 0)

        padding = (
            pad_width // 2, pad_width - pad_width // 2, 
            pad_height // 2, pad_height - pad_height // 2,  
            pad_depth // 2, pad_depth - pad_depth // 2 
        )
        img = torch.nn.functional.pad(img.unsqueeze(0), padding, "constant", 0)
        return img.squeeze(0)  # Remove the added batch dimension

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

    def __len__(self):   
        return len(self.paths)



    def __getitem__(self, idx):
        path = self.paths[idx]
        img = nib.load(path)
        img = img.get_fdata()  
        img = torch.from_numpy(img)
        target_size = (182, 216, 182)

        # If the image is larger than the target size, crop it
        if any(dim > target_size[i] for i, dim in enumerate(img.shape)):
            img = self.center_crop_3d(img, target_size)
            

        # If the image is smaller than the target size, pad it
        if any(dim < target_size[i] for i, dim in enumerate(img.shape)):
            img = self.pad_3d(img, target_size)

        img = img.type('torch.FloatTensor').unsqueeze(0)
        label = self.labels[idx]

        return img, label



class RAWT1AgeClassImgDataModule3D(LightningDataModule):
    def __init__(self, batch_size: int = 4,
                 paths={'root_dir': '/vol/biodata/data/biobank/18545/brain_data/',
                        'table_dir': '/data2/biodata/biobank/tables/'}, num_workers=2,
                 **kwargs):
        super().__init__()
        self.root_dir = paths['root_dir']
        self.modality_path = 'visit_2/T1/T1_brain.nii.gz'
        self.table_dir = paths['table_dir']
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize train dataset to compute scaler
        self.train_set = RawT1AgeClassImgDataset(self.root_dir, self.modality_path, self.table_dir, split='train', **kwargs)
        
        # Store the scaler used in training
        self.scaler = self.train_set.scaler
        
        # Initialize validation and test datasets using the same scaler
        self.val_set = RawT1AgeClassImgDataset(self.root_dir, self.modality_path, self.table_dir, split='val', scaler=self.scaler, **kwargs)
        self.test_set = RawT1AgeClassImgDataset(self.root_dir, self.modality_path, self.table_dir, split='test', scaler=self.scaler, **kwargs)



        self.corrupted = self.train_set.corrupted_folders + self.val_set.corrupted_folders + self.test_set.corrupted_folders
        # self.all_folders = self.train_set.usable_folders + self.val_set.usable_folders + self.test_set.usable_folders

        seed_everything(33)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)



