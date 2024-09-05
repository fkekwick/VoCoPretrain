import nibabel as nib
import sys
import os
import pandas as pd
import numpy as np



from  concurrent.futures import ThreadPoolExecutor, as_completed

sys.path
import warnings
warnings.simplefilter('ignore')
import random




class BrainImgDataset():
    def __init__(self,root_dir:str, modality_path:str , table_dir,split = 'train', percent_split = 0.001) -> None:
        self.root_dir = root_dir
        self.table_dir = table_dir
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
        self.index_split = (int(0.4*self.len_dataset), int(0.7*self.len_dataset)) 

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
    

    



    


