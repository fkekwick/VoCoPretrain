import torch as torch
import warnings
warnings.simplefilter('ignore')

from time import time
from datasets.BrainImagingDataset_3D import BrainImgDataset
from  monai import transforms
import numpy as np
from collections.abc import Callable, Sequence
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
import collections
import numpy as np
from monai.data import *
from monai.transforms import *
import argparse
import pickle
from utils.brain_data_utils import threshold

import torch.optim as optim
from models.voco_head import VoCoHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast

from utils.data_utils import get_position_label, get_crop_transform, get_vanilla_transform, VoCoAugmentation
from utils.utils import AverageMeter

import torch as torch
# from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
# from lightning.loggers import WandbLogger
import warnings
import os
import sys
# from lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping,  LearningRateMonitor

# from lightning.loggers import WandbLogger
import wandb
import monai.transforms as transforms
from monai import data
import argparse


import torch.optim as optim
from models.voco_head import VoCoHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import *
from utils.ops import *

roi = 142
num_workers = 1
logdir = "logs"
epochs = 10
num_steps = 100  #
eval_num = 100
warmup_steps = 5000
in_channels = 1
feature_size = 48 #

dropout_path_rate = 0.0
use_checkpoint = True
spatial_dims = 3
a_min = -175.0
a_max = 250.0
b_min = 0.0
b_max = 1.0
space_x = 1.5
space_y = 1.5
space_z = 1.5
roi_x = roi
roi_y = roi
roi_z = roi
batch_size = 1
sw_batch_size = 2
lr = 1e-4
decay = 0.1
momentum = 0.9
lrdecay = True
max_grad_norm = 1.0
loss_type = "SSL"
opt = "adamw"
lr_schedule = "warmup_cosine"





paths  = {    'root_dir':  '/data2/biodata/biobank/brain_data/',
    'table_dir' : '/data2/biodata/biobank/tables/'}
root_dir =  paths['root_dir']
modality_path = 'T1/T1_brain_to_MNI.nii.gz'
table_dir = paths['table_dir']

split = 0.002
batch_size=1
num_workers=4

datalist = BrainImgDataset(root_dir, modality_path,table_dir, percent_split = split,split='train').dict_paths
val_files = BrainImgDataset(root_dir, modality_path,table_dir,percent_split = split, split='val').dict_paths








parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")

parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=10, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=3e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=0.005, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", default=False, help="do NOT use amp for training")

parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=3, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")

parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", default=True, help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=142, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=178, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=154, type=int, help="roi size in z direction")

parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=100, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", default=True, help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
args = parser.parse_args()




train_transforms = transforms.Compose([LoadImaged(keys=["image"], image_only=True, dtype=np.int16),
                                EnsureChannelFirstd(keys=["image"]),
                                Orientationd(keys=["image"], axcodes="RAS"),
                                ScaleIntensityRanged(
                                    keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                    b_min=args.b_min, b_max=args.b_max, clip=True),
                                SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y,
                                                                        args.roi_z]),
                                CropForegroundd(keys=["image"], source_key="image"),
                                SpatialCropd(keys=["image"], roi_start=[60, 80, 0],
                                             roi_end=[440, 380, 10000]),
                                Resized(keys=["image"], mode="trilinear", align_corners=True,
                                        spatial_size=(384, 384, 96)),

                                # Random
                                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.0),
                                CropForegroundd(keys="image", source_key="image", select_fn=threshold),
                                Resized(keys="image", mode="bilinear", align_corners=True,
                                        spatial_size=(384, 384, 96)),

                                VoCoAugmentation(args, aug=True)
                                ])

val_transforms = transforms.Compose([LoadImaged(keys=["image"], image_only=True, dtype=np.int16),
                              EnsureChannelFirstd(keys=["image"]),
                              Orientationd(keys=["image"], axcodes="RAS"),
                              ScaleIntensityRanged(
                                  keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                  b_min=args.b_min, b_max=args.b_max, clip=True),
                              SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y,
                                                                      args.roi_z]),
                              CropForegroundd(keys=["image"], source_key="image"),
                              SpatialCropd(keys=["image"], roi_start=[60, 80, 0],
                                           roi_end=[440, 380, 10000]),
                              Resized(keys=["image"], mode="trilinear", align_corners=True,
                                      spatial_size=(384, 384, 96)),
                              VoCoAugmentation(args, aug=False)
                              ])





train_ds = data.PersistentDataset(data=datalist,
                                     transform=train_transforms,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir='/data/saved/voco_cache')
train_sampler = None
train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, sampler=None, shuffle=True,
        drop_last=True, pin_memory=True
    )



model = VoCoHead(args)

model.train()
loss_train = []
run_loss = AverageMeter()
pos_avg, neg_avg, base_avg = AverageMeter(), AverageMeter(), AverageMeter()
torch.cuda.set_device(0)
model.cuda()

def concat_image(imgs):
    output = []
    for img in imgs:
        img = img['image']
        output.append(img)
        
    output = torch.concatenate(output, dim=1)

    bs, sw_s, x, y, z = output.size()
    output = output.view(-1, 1, x, y, z)
    return output

for step, batch in enumerate(train_loader):
    t1 = time()
    img, labels, crops = batch
    print(step, len(batch))

    img, crops = concat_image(img), concat_image(crops)




    # print(img.size(), crops.size(), labels.size())
    img, crops, labels = img.cuda(), crops.cuda(), labels.cuda()

    with autocast(enabled=False):
        pos, neg, b_loss = model(img, crops, labels)
        loss = pos + neg + b_loss
        loss_train.append(loss.item())


