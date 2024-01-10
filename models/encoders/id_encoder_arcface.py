import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import os
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch import nn
from Models.Encoders.backbone_new import Backbone
import torch.nn.functional as F
from PIL import Image
from Configs.training_config import config


model_path='/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Aravind_data/codes/Aravind_NCSN++/model_ir_se50.pth'
input_size=[112, 112]
embedding_size=512

class IDEncoder(nn.Module):
  def __init__(self):
    super(IDEncoder,self).__init__()
    self.model_path=model_path
  
  def get_embeddings(self, image, embedding_size):
    
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image=F.interpolate(image,input_size,mode='bilinear',align_corners=True)
    backbone=Backbone(50,0.6,'ir_se')
    backbone.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    backbone.to(device)
    backbone.eval()
    embeddings = np.zeros([config['batchSize'],embedding_size])
    with torch.no_grad():
      embeddings[:]= F.normalize(backbone(image.to(device))).cpu()
      embeddings=torch.from_numpy(embeddings).to(device)
      embeddings=embeddings.clone().to(torch.float32)
      return embeddings

  def forward(self,image):
    gather_embeddings=self.get_embeddings(image,embedding_size)
    #gather_embeddings=torch.tensor(gather_embeddings,requires_grad=True)
    return gather_embeddings
    
    
    
    
    
    

    
    
      
   
      
      
    
     
         
      
