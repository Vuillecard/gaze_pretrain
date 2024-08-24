import rootutils
import random
import pickle
import numpy as np
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.ops import box_convert
from functools import partial
import einops
from tqdm import tqdm

path = '/idiap/temp/pvuillecard/projects/gaze_pretrain/'
rootutils.setup_root(path, indicator=".project-root", pythonpath=True)

from gaze_module.data.components.gaze_dataset import (
    Gaze360Image, 
    Gaze360Video,
    GFIEImage,
    GFIEVideo,
    MPSGazeImage, 
    GazeFollowImage,
    ChildPlayImage,
    EyediapImage,
    VATImage,
)
from gaze_module.data.components.transforms import (BboxRandomJitter,BboxReshape,Crop,
                                  ToTensor, Normalize,CropRandomResize,
                                  ToImage, HorizontalFlip,ColorJitter,RandomGaussianBlur)
from gaze_module.data.combined_datamodule import ConcatenateDataModule, SimpleDataModule

from gaze_module.models.components.gaze_models import GazeNet, TorchvisionEncoder, HeadCartesian
from gaze_module.utils.metrics import AngularError

def main_og(path_model):

    transorm_simple = Compose([    
                            BboxReshape( square = True, ratio = 0.1),
                            ToImage(),
                            Crop(224),
                            #CropRandomResize(224),
                            #HorizontalFlip(),
                            ToTensor(),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])

    dataset = Gaze360Image(
        image_db_path='/idiap/temp/pvuillecard/projects/face_analyser/datasets/Gaze360/gaze360_image_database.pkl',
        sample_path='/idiap/temp/pvuillecard/projects/face_analyser/datasets/Gaze360/samples/image_samples.csv',
        split='test',
        head_bbox_name='head_bbox_yolo',
        transform=transorm_simple,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10)
    device = 'cuda'

    #load model
    encoder = TorchvisionEncoder("omnivore",True,224)
    head = partial(HeadCartesian)
    net = GazeNet(
        encoder = encoder,
        head = head,
        activation= nn.ReLU(),
    )
    
    net_weight = torch.load(path_model)
    net_weight = { k.replace('net.',''):v for k,v in net_weight['state_dict'].items()}
    net.load_state_dict(net_weight, strict=False)
    net.to(device)
    net.eval()
    
    # metrics 
    metrics = AngularError(mode='cartesian')
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        
        images = batch['images']
        b,_,c,_,_ = images.shape

        images = images.to(device)
    
        with torch.no_grad():
            out = net(images,1)
        
        predictions = out['cartesian'].detach().cpu()
        metrics.update(predictions, batch["task_gaze_vector"])
    
    angular_error = metrics.compute()
    print(f'angular error og: {angular_error}')
    
def main(path_model):
    # explore the sensitivity of the model to random crops
    # q(1+0.1) = 224 Original crop thus 
    # q = 224/1.1 = 203.6
    # new_size = 203.6*(1+new_ratio) 
    transorm_simple = Compose([    
                            BboxReshape( square = True, ratio = 0.2),
                            ToImage(),
                            Crop(260),
                            #CropRandomResize(224),
                            #HorizontalFlip(),
                            ToTensor(),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])

    dataset = Gaze360Image(
        image_db_path='/idiap/temp/pvuillecard/projects/face_analyser/datasets/Gaze360/gaze360_image_database.pkl',
        sample_path='/idiap/temp/pvuillecard/projects/face_analyser/datasets/Gaze360/samples/image_samples.csv',
        split='test',
        head_bbox_name='head_bbox_yolo',
        transform=transorm_simple,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10)
    device = 'cuda'

    #load model
    encoder = TorchvisionEncoder("omnivore",True,224)
    head = partial(HeadCartesian)
    net = GazeNet(
        encoder = encoder,
        head = head,
        activation= nn.ReLU(),
    )

    net_weight = torch.load(path_model)
    net_weight = { k.replace('net.',''):v for k,v in net_weight['state_dict'].items()}
    net.load_state_dict(net_weight, strict=False)
    net.to(device)
    net.eval()
    
    # metrics 
    metrics = AngularError(mode='cartesian')
    metrics_middle = AngularError(mode='cartesian')

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        
        images = batch['images']
        b,_,c,_,_ = images.shape
        images_fives_crop = torchvision.transforms.functional.five_crop(images, 224)
        images = torch.stack(images_fives_crop, dim=1)
        images = einops.rearrange(images, 'b nb_crop f c h w -> (b nb_crop) f c h w')
        images = images.to(device)
    
        with torch.no_grad():
            out = net(images,1)
        
        predictions = out['cartesian'].detach().cpu()
     
        predictions = einops.rearrange(predictions, '(b nb_crop) c -> b nb_crop c', b = b)
        
        metrics_middle.update(predictions[:,4], batch["task_gaze_vector"])

        predictions_mean = predictions.mean(dim=1)
        metrics.update(predictions_mean, batch["task_gaze_vector"])
    
    angular_error = metrics.compute()
    print(f'angular error mean: {angular_error}')
    angular_error_middle = metrics_middle.compute()
    print(f'angular error center: {angular_error_middle}')

    

if __name__ == "__main__":

    #path_model = '/idiap/temp/pvuillecard/projects/gaze_pretrain/logs/experiments/omnivore/run_2024-07-03_12-50-08/logs/train/runs/run_0/checkpoints/best_epoch_033.ckpt'
    # model trained on gaze360, gfie, mps , gf
    #path_model = '/idiap/temp/pvuillecard/projects/gaze_pretrain/logs/experiments/generalize/run_2024-07-07_11-01-56/logs/train/runs/run_0/checkpoints/best_epoch_039.ckpt'
    
    # path model 
    path_model = '/idiap/temp/pvuillecard/projects/gaze_pretrain/logs/experiments/generalize/run_2024-07-10_17-41-24/logs/train/runs/run_0/checkpoints/last.ckpt'
    main_og(path_model)
    main(path_model)