import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import os
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch import nn
from Models.Encoders.face_encoder import ViT_face_ID_encoder
import torch.nn.functional as F
from PIL import Image
from Configs.training_config import config


model_path='/home/user1/aravind/New_ID/Face_transformer_model.pth'
input_size=[112, 112]
embedding_size=512

class IDEncoder(nn.Module):
  def __init__(self):
    super(IDEncoder,self).__init__()
    self.model_path=model_path
  
  def get_embeddings(self, image, embedding_size):
    '''
    id_transform=transforms.Compose([
          transforms.Resize(
              [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
            ), # smaller side resized
          transforms.CenterCrop([input_size[0], input_size[1]]),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
      ],
    )
    
    image=id_transform(image)
    image=image.unsqueeze(0)
    '''
    image=F.interpolate(image,size=112)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out=ViT_face_ID_encoder(image_size=112,
            patch_size=8,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1)
    #out.to('cuda')
    out.load_state_dict(torch.load(self.model_path,map_location=torch.device("cpu")))
    out.to(device)
    out.eval()
    embeddings = np.zeros([config['batchSize'],embedding_size])
    with torch.no_grad():
      embeddings[:] = F.normalize(out(image.to(device))).cpu()
      return embeddings

  def forward(self,image):
    gather_embeddings=self.get_embeddings(image,embedding_size)
    gather_embeddings=torch.tensor(gather_embeddings,requires_grad=True)
    return gather_embeddings
    
    
    
    
    
    

    
    
      
   
      
      
    
     
         
      
