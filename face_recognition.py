#!/usr/bin/env python
# coding: utf-8

# In[1]:


from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import torch.optim as optim
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os


# In[2]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


# In[3]:


mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
batch_size = 8
dataset = datasets.ImageFolder('image') #../data/test_images
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

face_list = []
name_list = []
embedding_list = []
aligned = []

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True)
    if face is not None and prob>0.90:
        print("Face detected with probability: {:8f}".format(prob))
        aligned.append(face)
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])


# In[4]:


data = [embedding_list, name_list]
torch.save(data, 'data.pt')


# In[5]:


aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()


# In[6]:


def face_match(img_path, data_path):
    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True)
    emb = resnet(face.unsqueeze(0)).detach()
    
    saved_data = torch.load('data.pt')
    embedding_list = saved_data[0]
    name_list = saved_data[1]
    dist_list = []
    
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)
        
    idx_min = dist_list.index(min(dist_list))
    
    return name_list[idx_min]

person = face_match('suj.jpeg', 'data.pt')
print("Identity matched with: ", person)


# In[7]:


import pandas as pd
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=name_list, index=name_list))


# In[ ]:





# In[ ]:




