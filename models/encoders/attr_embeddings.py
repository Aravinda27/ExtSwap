import sys
from PIL.ImageDraw import ImageDraw
sys.path.append("/content/drive/MyDrive/Deepfakes/model")
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import logging
from Models.Encoders.attr_encoder import effnetv2_l
import torch.nn.functional as F
input_size=[224,224]

class Attrencoder(nn.Module):
  def __init__(self):
    super().__init__()

    
    self.logger=logging.getLogger(__class__.__name__)
    self.model=effnetv2_l()
  
  def get_attr(self,image):
    '''
    attr_transforms=transforms.Compose([
          transforms.Resize(
              [int(128 * input_size[0] / 224), int(128 * input_size[0] / 224)],
            ), # smaller side resized
          transforms.CenterCrop([input_size[0], input_size[1]]),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
      ],
    )
    image=attr_transforms(image)
    image=image.unsqueeze(0)
    '''
    image=F.interpolate(image,size=224)
    activation = {}
    def get_activation(name):
      def hook(model, input, output):
        activation[name] = output.detach()
      return hook
    self.model.conv.register_forward_hook(get_activation('conv'))
    attr_embeddings=self.model(image)
    features=activation['conv']
    avgpool=nn.AdaptiveAvgPool2d((1,1))
    features_avgpool=avgpool(features)
    dropout_features=nn.Dropout(p=0.5)
    #print(features_avgpool.shape)
    features_dropout=dropout_features(features_avgpool)
    flattened_features=torch.flatten(features_dropout,1)
    return flattened_features
  


  def forward(self,image):
    attr_features=self.get_attr(image)
    return attr_features
  

  '''
  def my_save(self,state,reason=''):
    f_path=(str(self.args.weights_dir.joinpath(self.__class__.__name__+ reason)))
    torch.save(state,f_path)
  '''
  



'''
def attr(image):
  resize = transforms.Resize([224, 224])
  image = resize(image)
  to_tensor = transforms.ToTensor()
  image= to_tensor(image)
  image=image.unsqueeze(0)

  model=effnetv2_l()
  #print(list(model.parameters()))

  activation = {}
  def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
  model.conv.register_forward_hook(get_activation('conv'))
  #inputs = torch.rand(1, 3, 224, 224)
  outputs = model(image)
  features_1=activation['conv']
  #print(features_1.shape)
  return features_1
'''

  
  


