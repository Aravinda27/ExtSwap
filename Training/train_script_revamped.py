#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 12:08:49 2022

@author: user1
"""

import os
import sys
import dill

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import wandb
from Training.trainer import Trainer
from torch.utils.data import DataLoader, random_split
from Models.Encoders.Landmark_Encoder import Landmark_Encoder
from Models.Encoders.id_encoder import IDEncoder
from Models.Encoders.attr_embeddings import Attrencoder
from Models.Encoders.Inception import Inception
from Models.Discrimanator import Discriminator
from Models.LatentMapper import LatentMapper
from Models.StyleGan2.model import Generator
from Utils.data_utils import get_w_image, Image_W_Dataset, cycle_images_to_create_diff_order,restore_checkpoint,tensor2im,vis_faces
from Utils.ema import ExponentialMovingAverage
import time
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from Losses import id_loss
from random import choice
from string import ascii_uppercase
#sys.path.append("/content/drive/MyDrive/ID-disentanglement-Pytorch-master")
from Configs import Global_Config
from Configs.training_config import config, GENERATOR_IMAGE_SIZE
BASE_PATH = Global_Config.BASE_PATH
from torch.utils.tensorboard import SummaryWriter
from Utils import data_loader_Swapping
import os
import matplotlib.pyplot as plt


MOBILE_FACE_NET_WEIGHTS_PATH = BASE_PATH + '/home/user1/aravind/New_ID/mobilefacenet_model_best.pth.tar'
workers_id='cpu'
GENERATOR_WEIGHTS_PATH = BASE_PATH + '/home/user1/aravind/New_ID/550000.pt'
E_ID_WEIGHTS_PATH = BASE_PATH + 'resnet50_scratch_dag.pth'
E_ID_NEW__WEIGHTS_PATH = BASE_PATH + 'resnet50_scratch_weight.pkl'
E_ID_LOSS_PATH = BASE_PATH + '/home/user1/aravind/New_ID/model_ir_se50.pth'
DLIB_WEIGHT_PATH = BASE_PATH + 'mmod_human_face_detector.dat'
IMAGE_DATA_DIR = BASE_PATH + '/home/user1/aravind/New_ID/Utils/fake/small_image'
W_DATA_DIR = BASE_PATH + '/home/user1/aravind/New_ID/Utils/fake/small_w'
MODELS_DIR = BASE_PATH + '/home/user1/aravind/New_ID/Models/'

class Coach:
    def __init__(self):
        self.id_encoder = IDEncoder()
        self.mlp = torch.nn.DataParallel(LatentMapper())
        self.landmark_encoder = Landmark_Encoder.Encoder_Landmarks(MOBILE_FACE_NET_WEIGHTS_PATH)
        self.generator = Generator(GENERATOR_IMAGE_SIZE, 512, 8)
        
        state_dict = torch.load(GENERATOR_WEIGHTS_PATH)
        self.generator.load_state_dict(state_dict['g_ema'], strict=False)
        self.id_encoder = self.id_encoder.to(Global_Config.device)
        self.mlp = self.mlp.to(Global_Config.device)
        self.generator = self.generator.to(Global_Config.device)
        
        
        self.id_encoder = self.id_encoder.eval()
        self.generator = self.generator.eval()
        self.mlp = self.mlp.train()
        self.landmark_encoder = self.landmark_encoder.eval()
        self.ema = ExponentialMovingAverage(self.mlp.parameters(), decay=0.999)
        self.optimizer_non_adv_M = torch.optim.Adam(list(self.mlp.parameters()),
                                       lr=config['non_adverserial_lr'], betas=(config['beta1'], config['beta2']))
        
        state = dict(optimizer=self.optimizer_non_adv_M, model=self.mlp, ema=self.ema, step=0)
        self.sample_dir = os.path.join(BASE_PATH, 'samples')
        os.makedirs(self.sample_dir, exist_ok=True)
        self.logger = SummaryWriter(self.sample_dir)
        checkpoint_dir = os.path.join(self.sample_dir, "checkpoints")
        # Intermediate checkpoints to resume training after pre-emption in cloud environments
        checkpoint_meta_dir = os.path.join(self.sample_dir, "checkpoints-meta", "checkpoint.pth")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
        
        # Resume training when intermediate checkpoints are detected
        state = restore_checkpoint(checkpoint_meta_dir, state, Global_Config.device)
        self.initial_step = int(state['step'])
        train_loader=data_loader_Swapping.GetLoader(config['data_path'],config['batchSize'],8,1234)
        
        #w_image_dataset = Image_W_Dataset(W_DATA_DIR, IMAGE_DATA_DIR)
        
        #train_size = int(config['train_precentege'] * len(config['batchSize']))
        #test_size = len( train_loader) - train_size
        #train_data, test_data = random_split(train_loader, [train_size, test_size])
        
        #train_loader = DataLoader(dataset=train_data, batch_size=config['batchSize'], shuffle=False)
        with tqdm(total=config['epochs'] * len(train_loader)) as pbar:
            for step in range(self.initial_step, config['number_training_steps'] + 1):
                    id_images,attr_images=train_loader.next()
                    #ws, images = data
                    id_images = id_images.to(Global_Config.device).float()
                    attr_images = attr_images.to(Global_Config.device).float()
                    try:
                        with torch.no_grad():
                            id_vec=self.id_encoder(id_images)
                            id_vec = id_vec.to(Global_Config.device)
                            real_landmarks, real_landmarks_nojawline = self.landmark_encoder(attr_images)
                    except Exception as e:
                        print(e)
                    attr_vec = self.id_encoder(attr_images)
                    attr_vec = attr_vec.to(Global_Config.device)
                    try:
                        encoded_vec = torch.cat((id_vec, attr_vec), dim=1)
                        print(encoded_vec.shape)
                        #encded_vec=encoded_vec.to(torch.float32)
                        encoded_vec = encoded_vec.to(Global_Config.device)
                    except Exception as e:
                        print(e)
                    fake_data= self.mlp(encoded_vec)
                    fake_data = fake_data.to(Global_Config.device)
                    fake_data=fake_data.to(torch.float32)
                    sample, latents = self.generator(
                        [fake_data], input_is_latent=True, randomize_noise=False,return_latents=True)
                    if step % config['training.log_freq'] == 0:
                        self.parse_and_log_images(id_images,attr_images,state,sample, title='images/train/faces')
                        pbar.update(1)  			             	     
    
    
    
    def parse_and_log_images(self,x, y,state, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
				'input_face': tensor2im(x[i]),
				'target_face': tensor2im(y[i]),
				'output_face': tensor2im(y_hat[i]),
            }
            im_data.append(cur_im_data)
        
        self.log_images(title, state,im_data=im_data, subscript=subscript)
    
    def log_images(self, name,state, im_data, subscript=None, log_latest=False):
        fig =vis_faces(im_data)
        step=int(state['step'])
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.sample_dir, name, f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.sample_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        
        
        
                        
                    
                    
                
        
        
        
        
