import nibabel as nib
import sys
import os
import pandas as pd
import numpy as np
import torch as torch

import torch.nn.functional as F

from torch.utils.data import DataLoader
import random
import math



from  concurrent.futures import ThreadPoolExecutor, as_completed

sys.path
import warnings
warnings.simplefilter('ignore')
import random






class BrainImgDataset():
    def __init__(self,root_dir:str, modality_path:str , table_dir,split = 'train', percent_split = 0.6, scale_factor=1) -> None:
        self.root_dir = root_dir
        self.table_dir = table_dir
        self.sf = scale_factor
        self.all_folders, self.all_labels = self.load_sex_table()
        self.modality_path =  modality_path
        self.existing_folders = self.check_paths_and_get_folders(self.all_folders)
        self.len_dataset_all = len(self.existing_folders)
        print('whole dataset size', self.len_dataset_all)

        #split to pretraining dataset

        random.shuffle(self.existing_folders)
        self.existing_folders= self.existing_folders[0:int(percent_split*self.len_dataset_all)]
        self.len_dataset = len(self.existing_folders)
        self.split = split
        self.index_split = (int(0.75*self.len_dataset), int(0.9*self.len_dataset)) 

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



        self.dict_paths =  self.create_paths()
        print('Number of ',self.split,'images ', len(self.dict_paths) )

        
        


    
    def check_paths_and_get_folders(self, foldernames):
        existing_folders = []
        for foldername in foldernames:
            path = os.path.join(self.root_dir, str(foldername), self.modality_path)
            if os.path.exists(path):
                existing_folders.append(foldername)
        
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
                label = int(self.labels_table.loc[folder, 'Sex'])
                del img
                paths.append(path)
                labels.append(label)
                f.appen(folder)
            except:
                corrupted_folders.append(folder)

        dict_paths = [{'image': path, 'label': label} for path, label in zip(paths, labels)]
        
        return dict_paths

    
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

        return img
    




# class PatchBrainImgDataset():
#     def __init__(self,root_dir:str, modality_path:str , table_dir,split = 'train', percent_split = 0.7, scale_factor=1, target_size=(142,178,152)) -> None:
#         self.target_size = target_size
#         self.root_dir = root_dir
#         self.table_dir = table_dir
#         self.sf = scale_factor
#         self.all_folders, self.all_labels = self.load_sex_table()
#         self.modality_path =  modality_path
#         self.existing_folders = self.check_paths_and_get_folders(self.all_folders)
#         self.len_dataset_all = len(self.existing_folders)
#         print('whole dataset size', self.len_dataset_all)

#         #split to pretraining dataset
#         random.seed(33)
#         random.shuffle(self.existing_folders)
#         self.existing_folders= self.existing_folders[0:int(percent_split*self.len_dataset_all)]
#         self.len_dataset = len(self.existing_folders)
#         self.split = split
#         self.index_split = (int(0.75*self.len_dataset), int(0.8*self.len_dataset)) 

#         # find list of eid's of patients for train val and test set
#         assert self.split in ['train','val','test']
#         if self.split == 'train':
#             self.folders = self.existing_folders[:self.index_split[0]]
#             print('Number of train images')

#         elif self.split == 'val':
#             self.folders = self.existing_folders[self.index_split[0]:self.index_split[1]]
#             print('Number of val images')

#         elif self.split == 'test':
#             self.folders = self.existing_folders[self.index_split[1]:]
#             print('Number of test images')
            
#         else:
#             print('incorrect split variable')
#         print(len(self.folders))

#         self.labels_table =  self.all_labels[self.all_labels.index.isin(self.folders)]
#         self.paths, self.labels, self.corrupted_folders, self.usable_folders = self.create_paths()
#         assert len(self.labels) == len(self.paths)
#         print('Number of ',self.split,'images ', len(self.paths),'number of ', self.split, 'labels' )
        
        
        

    
#     def check_paths_and_get_folders(self, foldernames):
#         existing_folders = []
#         for foldername in foldernames:
#             path = os.path.join(self.root_dir, str(foldername), self.modality_path)
#             if os.path.exists(path):
#                 existing_folders.append(foldername)

#         return existing_folders
    
#     def create_paths(self):
#         paths = []
#         labels = []
#         f=[]
#         corrupted_folders = []
#         for folder in self.folders:
#             path =os.path.join(self.root_dir,str(folder), self.modality_path)
#             try:
#                 img = nib.load(path)
#                 label = self.labels_table.loc[folder]
#                 del img
#                 paths.append(path)
#                 labels.append(label)
#                 f.appen(folder)

#             except:
#                 corrupted_folders.append(folder)
#         return paths, labels, corrupted_folders, f

    
#     def load_sex_table(self):
#         path_demographics = os.path.join( self.table_dir, 'ukb_cardiac_image_subset_Primary demographics.csv')
#         df_demo = pd.read_csv(path_demographics)
#         df_demo = df_demo.drop(df_demo.index[0])
#         df_demo = df_demo.set_index('eid')
#         df_demo['Sex'] = df_demo['Sex'].astype(str)
#         df_demo = df_demo[['Sex']]

#         num_unexpected_sex_val =len(df_demo.loc[~df_demo['Sex'].isin(['0', '1']), 'Sex'].unique())
#         if  num_unexpected_sex_val != 0:
#             print(f"Number of unexpected Sex sentries that aren't 1 or 0: {num_unexpected_sex_val}")
#         return df_demo.index.to_list(), df_demo
    

#     def center_crop_3d(self, img, target_size):
#         """Center crop a 3D image to the target size (depth, height, width)."""
#         current_depth, current_height, current_width = img.shape

#         depth_crop = (current_depth - target_size[0]) // 2
#         height_crop = (current_height - target_size[1]) // 2
#         width_crop = (current_width - target_size[2]) // 2

#         return img[
#             depth_crop:depth_crop + target_size[0],
#             height_crop:height_crop + target_size[1],
#             width_crop:width_crop + target_size[2]
#         ]
    
#     def downsample_slices(self,img, f_w, f_d):
#         f_h = f_w
#         image = img[0::f_h,0::f_w,0::f_d]

#         return image

#     def reduce_slices(self,img, every_nth_slice=4):
#         image = np.delete(img, np.arange(every_nth_slice - 1, img.shape[2], every_nth_slice), axis=2)

#         return image

#     def random_resized_crop(self, image: torch.Tensor, probability: float, scale=(0.1, 1.0), ratio=(0.7, 1)) -> torch.Tensor:
#         """
#         Randomly crops a 3D image with given probability, scale, and aspect ratio bounds, and resizes the crop 
#         back to the original dimensions of the input image.

#         Args:
#             image (torch.Tensor): The input image tensor with shape (C, D, H, W).
#             probability (float): Probability of applying the crop.
#             scale (tuple of float): Specifies the lower and upper bounds for the random volume of the crop, before resizing.
#             ratio (tuple of float): Specifies the lower and upper bounds for the random aspect ratio of the crop, before resizing.

#         Returns:
#             torch.Tensor: The cropped and resized image or the original image.
#         """
        
#         if random.random() > probability:
#             return image 
        
#         channels, orig_depth, orig_height, orig_width = image.shape
#         volume = orig_depth * orig_height * orig_width

#         # Calculate the logarithmic ratio bounds for sampling
#         log_ratio_min = math.log(ratio[0])
#         log_ratio_max = math.log(ratio[1])

#         # Try up to 10 times to find a valid crop within the bounds
#         for _ in range(10):
#             # Sample target volume and aspect ratios
#             target_volume = volume * random.uniform(scale[0], scale[1])
#             aspect_ratio_xy = math.exp(random.uniform(log_ratio_min, log_ratio_max))
#             aspect_ratio_z = math.exp(random.uniform(log_ratio_min, log_ratio_max))

#             # Compute target dimensions (depth, height, width) based on the sampled volume and aspect ratios
#             new_width = int(round((target_volume * aspect_ratio_xy * aspect_ratio_z) ** (1/3)))
#             new_height = int(round((target_volume * aspect_ratio_xy / aspect_ratio_z) ** (1/3)))
#             new_depth = int(round((target_volume / (aspect_ratio_xy * aspect_ratio_z)) ** (1/3)))

#             # Check if the new dimensions fit within the original image dimensions
#             if new_width <= orig_width and new_height <= orig_height and new_depth <= orig_depth:
#                 # Randomly choose the top-left-front corner for the crop
#                 front = random.randint(0, orig_depth - new_depth)
#                 top = random.randint(0, orig_height - new_height)
#                 left = random.randint(0, orig_width - new_width)

#                 # Perform the crop
#                 cropped_image = image[:, front: front + new_depth, top: top + new_height, left: left + new_width]
                
#                 # Resize the cropped image back to the original dimensions
#                 resized_image = F.interpolate(
#                     cropped_image.unsqueeze(0),  # Add a batch dimension
#                     size=(orig_depth, orig_height, orig_width),
#                     mode='trilinear',  # Use trilinear interpolation for 3D
#                     align_corners=False
#                 ).squeeze(0)  # Remove the batch dimension
#                 return resized_image

#         # if no valid crop was found, return the original image
#         return image



#     def __len__(self):   
#         l= len(self.paths)
#         return l
    

#     def __getitem__(self,idx):
#         #load image from file
#         path = self.paths[idx]
#         img = nib.load(path)
#         img = img.get_fdata()
#         img = self.center_crop_3d(img, self.target_size)
#         img = self.reduce_slices(img, 6)
        
#         img = torch.from_numpy(img).type('torch.FloatTensor').unsqueeze(0)

#         #reduce resolution
#         # img = F.interpolate(img, scale_factor=self.sf)
        
#         img_view1 = img.type('torch.FloatTensor')

#         img = self.random_resized_crop(img, 1 ,scale=(0.1, 1.0), ratio=(0.7, 1))
#         #img_view1=self.augmentation_pipeline(img).type('torch.FloatTensor')
#         img_view2= self.augmentation_pipeline(img).type('torch.FloatTensor')
#         del img
#         return img_view1, img_view2
    





