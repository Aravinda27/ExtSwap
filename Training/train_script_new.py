#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 11:38:59 2022

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
from Utils.data_utils import get_w_image, Image_W_Dataset, cycle_images_to_create_diff_order
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



def prepeare_env_for_local_use():
    CUDA_VISIBLE_DEVICES = '4'
    os.chdir('..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES



if Global_Config.run_in_slurm:
    os.chdir('..')
else:
    prepeare_env_for_local_use()
    
    
    
id_encoder = IDEncoder()
discriminator = Discriminator()
mlp = LatentMapper()
landmark_encoder = Landmark_Encoder.Encoder_Landmarks(MOBILE_FACE_NET_WEIGHTS_PATH)
generator = Generator(GENERATOR_IMAGE_SIZE, 512, 8)

state_dict = torch.load(GENERATOR_WEIGHTS_PATH)
generator.load_state_dict(state_dict['g_ema'], strict=False)

id_encoder = id_encoder.to(Global_Config.device)
discriminator = discriminator.to(Global_Config.device)

mlp = mlp.to(Global_Config.device)
generator = generator.to(Global_Config.device)

landmark_encoder = landmark_encoder.to(Global_Config.device)

id_encoder = id_encoder.eval()
discriminator = discriminator.train()
generator = generator.eval()
mlp = mlp.train()
landmark_encoder = landmark_encoder.eval()

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


toggle_grad(id_encoder, True)
toggle_grad(generator, True)
toggle_grad(mlp, True)
toggle_grad(landmark_encoder, True)

w_image_dataset = Image_W_Dataset(W_DATA_DIR, IMAGE_DATA_DIR)


train_size = int(config['train_precentege'] * len(w_image_dataset))
test_size = len(w_image_dataset) - train_size
train_data, test_data = random_split(w_image_dataset, [train_size, test_size])



train_loader = DataLoader(dataset=train_data, batch_size=config['batchSize'], shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=config['batchSize'], shuffle=False)


optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['adverserial_D'],
                               betas=(config['beta1'], config['beta2']))

optimizer_adv_M = torch.optim.Adam(mlp.parameters(), lr=config['adverserial_M'],
                                   betas=(config['beta1'], config['beta2']))

optimizer_non_adv_M = torch.optim.Adam(list(mlp.parameters()),
                                       lr=config['non_adverserial_lr'], betas=(config['beta1'], config['beta2']))

trainer = Trainer(config, optimizer_D, optimizer_adv_M, optimizer_non_adv_M, discriminator, generator,
                  id_encoder, landmark_encoder)


run_name = ''.join(choice(ascii_uppercase) for i in range(12))
#wandb.login(key=[7a746f79d9dfaec2423cfdab425c9d9792a00bce])
run = wandb.init(project="Face_swap_new", reinit=True, config=config, name=run_name)

def get_concat_vec(id_images, attr_images, id_encoder):
    with torch.no_grad():
        id_vec=id_encoder(id_images)
        id_vec = id_vec.to(Global_Config.device)
        #print("ID_shape_get_concat_vec",id_vec.shape)
        attr_vec = id_encoder(attr_images)
        attr_vec = attr_vec.to(Global_Config.device)
        #attr_vec=torch.mean(attr_vec.view(attr_vec.size(0),attr_vec.size(1),-1),dim=2)
        test_vec = torch.cat((id_vec, attr_vec), dim=1)
        test_vec=test_vec.to(torch.float32)
        encoded_vec = test_vec.to(Global_Config.device)
        return encoded_vec
    
data = next(iter(test_loader))
ws, images = data


test_id_images = images.to(Global_Config.device)
test_attr_images = test_id_images
test_ws = ws.to(Global_Config.device)
#print("Ws",test_ws.shape)
if config['use_cycle']:
    test_attr_images_cycled = cycle_images_to_create_diff_order(test_id_images)
    
with torch.no_grad():
    for idx in range(len(test_ws)):
     # print("Test_idx",test_ws[idx].shape)
      w_image= get_w_image(test_ws[idx], generator)
      

use_descrimantive_loss = config['use_adverserial']

with tqdm(total=config['epochs'] * len(train_loader)) as pbar:
    for epoch in range(config['epochs']):
        
        for idx, data in enumerate(train_loader):
            Global_Config.step += 1
            ws, images = data
            wandb.log({f"ID_Image{idx}": [wandb.Image(images * 255, caption=f"ID_Image{idx}")]}, step=0)
            id_images = images.detach().clone().to(Global_Config.device)
            #print(id_images.shape)
            attr_images = images.detach().clone().to(Global_Config.device)
            ws = ws.to(Global_Config.device)
            if config['use_cycle'] and idx % config['IdDiffersAttrTrainRatio'] == 0:
                attr_images = cycle_images_to_create_diff_order(attr_images)
            try:
            
            	with torch.no_grad():
              		id_vec=id_encoder(id_images)
              
              		id_vec = id_vec.to(Global_Config.device)
              		real_landmarks, real_landmarks_nojawline = landmark_encoder(attr_images)
            except Exception as e:
                print(e)
            
            attr_vec = id_encoder(attr_images)
            attr_vec = attr_vec.to(Global_Config.device)
            #print("Attr_vec",attr_vec.shape)
            try:
            
            #attr_vec=torch.mean(attr_vec.view(attr_vec.size(0),attr_vec.size(1),-1),dim=2)
            	encoded_vec = torch.cat((id_vec, attr_vec), dim=1)
            	encded_vec=encoded_vec.to(torch.float32)
            	encoded_vec = encoded_vec.to(Global_Config.device)
            #print("Encoded_vector",encoded_vec.shape)
            except Exception as e:
                print(e)
            fake_data= mlp(encoded_vec)
            use_rec_extra_term = not config['use_cycle']
            if config['use_cycle']:
                use_rec_extra_term = idx % config['IdDiffersAttrTrainRatio'] != 0
            
            
            
            if use_descrimantive_loss and idx % 2 == 0:
                error_real, error_fake, prediction_real, prediction_fake, g_error, g_pred = trainer. \
                    adversarial_train_step(ws, fake_data)
            
            else:
                total_error = trainer.non_adversarial_train_step(
                    id_images, attr_images,fake_data, real_landmarks_nojawline, use_rec_extra_term)
                wandb.log({'total error': total_error.detach().cpu()}, step=Global_Config.step)

            pbar.update(1)
            
            if idx % 1000 == 0 and idx != 0:
                with torch.no_grad():
                    concat_vec = get_concat_vec(test_id_images, test_attr_images, id_encoder)
                    if config['use_cycle']:
                        concat_vec_cycled = get_concat_vec(test_id_images, test_attr_images_cycled, id_encoder)
                    with torch.no_grad():
                        for idx in range(len(test_ws)):
                            id_generated_image = get_w_image(mlp(concat_vec)[idx], generator)
                            if config['use_cycle']:
                                cycled_generated_image = get_w_image(mlp(concat_vec_cycled)[idx], generator)
                            wandb.log(
                                {f"Train_ID_Image{idx}": [
                                    wandb.Image(id_generated_image * 255, caption=f"Train_ID_Image{idx}")]},
                                step=Global_Config.step)
                            if config['use_cycle']:
                                wandb.log(
                                    {f"Cycle_Train_ID_Image{idx}": [
                                        wandb.Image(cycled_generated_image * 255,
                                                    caption=f"Cycle_Train_ID_Image{idx}")]}, step=Global_Config.step)
                
                        
                            
            if (idx % 10000) == 0:
                torch.save(mlp.state_dict(), f'{MODELS_DIR}maper_{run_name}_{time.time()}_{int(total_error)}.pt')
                



























