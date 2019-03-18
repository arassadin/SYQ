#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
import os

tini_path = '/home/stasysp/Envs/shad/SYQ/tiny-imagenet-200'
tini_meta_path = '/home/stasysp/Envs/shad/SYQ/SYQ-master/tensorpack/dataflow/dataset/tini_metadata/'
folders = ['train', 'val', 'test']


# In[55]:


train = []
path = tini_path +'/train/'

classes = []
dirs = []
for root, dirs_, files in os.walk(path):
    for i in dirs_:
        if 'n' in i:
            dirs.append(i + '/images/')
            
for d in dirs:
    for root, dirs_, files in os.walk(path + d):
        for f in files:
            train.append([d + f, d.split('/')[0]])
        
train = np.array(train)


# In[56]:


val_annot = tini_path +'/val/val_annotations.txt'
with open(val_annot) as f:
    val = f.readlines()
val = [i.split('\t')[:2] for i in val]
val = [['images/' + i[0], i[1]] for i in val]


with open(tini_meta_path + 'val.txt',"w") as f:
    f.write("\n".join(" ".join(map(str, x)) for x in val))


# In[64]:


classes = dict([[c, i] for i, c in enumerate(list(set([i[1] for i in train] + [i[1] for i in val])))])

with open(tini_meta_path + 'train.txt',"w") as f:
    f.write("\n".join(" ".join(map(str, [x[0], classes[x[1]]])) for x in train))
    
with open(tini_meta_path + 'val.txt',"w") as f:
    f.write("\n".join(" ".join(map(str, [x[0], classes[x[1]]])) for x in val))    


# In[53]:


test = []
path = tini_path +'/test/images/'

for root, dirs_, files in os.walk(path):
    for i in files:
        test.append(['images/' + i, '0'])
with open(tini_meta_path + 'test.txt',"w") as f:
    f.write("\n".join(" ".join(map(str, x)) for x in test))


# In[ ]:




