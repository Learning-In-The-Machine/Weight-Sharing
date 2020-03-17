#!/usr/bin/env python
# coding: utf-8

# # Results

# In[1]:


import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from helper import *
from tqdm import tqdm

pbar = tqdm(total=19)
# In[2]:

from IPython.core.pylabtools import figsize
figsize(20, FIG_HEIGHT)

# LATEX BACKEND
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=FONTSIZE)

# # MNIST
# ---

# In[6]:

mnist, mnist_cnn, mnist_fcn, mnist_fcn_vcp = load_results('mnist')
cifar, cifar_cnn, cifar_fcn, cifar_fcn_vcp = load_results('cifar')

# Paper Figures
# ---
### 1. Translation (left: FCN; right: CNN)
#    * Translation Augmented Training Acc
#    * Un-Augmented Validation Acc
#    * Translation Augmented Validation Acc

# In[13]:

translation_paper_fig(mnist_fcn, mnist_cnn)
pbar.update(1)

# In[14]:

translation_paper_fig(cifar_fcn, cifar_cnn, cifar=True)
pbar.update(1)

### 2. Approximate Weight Sharing (left: distance @ 1; right: distance @ 4)

# In[15]:


approx_ws(mnist_fcn, dist_type='dist', y=[.05, .12])
#approx_ws(mnist_fcn, dist_type='cos')
pbar.update(1)

# In[16]:


#approx_ws(cifar_fcn, dist_type='dist')
approx_ws(cifar_fcn, dist_type='cos',y=[.85, 1],cifar=True)
pbar.update(1)

# ### 3. Swap (left: MNIST; right: CIFAR)

# In[17]:

df = pd.read_csv('../FinalResults/mnist_1.csv')
cnn = df[df['model'] == 'cnn']
fcn = df[df['model'] == 'fcn']

plt.clf(); _=plt.figure(figsize=(10, FIG_HEIGHT));

p = sns.color_palette('coolwarm', 11)
sns.lineplot(x='epoch', y='swap', data=cnn, label='CNN',color=p[0]); sns.lineplot(x='epoch', y='swap', data=fcn, label='FCN',color=p[-1])
plt.title('Feature Swap MNIST'); plt.xlabel('Epochs', fontsize=FONTSIZE); plt.ylabel('Accuracy', fontsize=FONTSIZE); plt.legend(shadow=True, title='Model', prop=legend_settings, fontsize=FONTSIZE);

plt.tight_layout(); plt.savefig('Figures/Paper/swap.{save_type}'.format(
    save_type=SAVE_TYPE
))
pbar.update(1)

# Supplementary Material Figures
# ---
### 1. Augmentation Type
#    * (left: aug_type training, right: un-augmented val acc)
#    * (left: translation val, right: rotation val)
#    * (left: noise val, right: edge noise val)
#    * swap val

# In[19]:


translation_sm_fig(mnist_cnn, aug_type='rotation', save=True)
pbar.update(1)

translation_sm_fig(mnist_cnn, aug_type='noise', save=True)
pbar.update(1)

translation_sm_fig(mnist_cnn, aug_type='edge_noise', save=True)
pbar.update(1)

translation_sm_fig(mnist_fcn, aug_type='rotation', save=True)
pbar.update(1)

translation_sm_fig(mnist_fcn, aug_type='noise', save=True)
pbar.update(1)

translation_sm_fig(mnist_fcn, aug_type='edge_noise', save=True)
pbar.update(1)

translation_sm_fig(cifar_cnn, aug_type='rotation', cifar=True, save=True)
pbar.update(1)

translation_sm_fig(cifar_cnn, aug_type='noise', cifar=True, save=True)
pbar.update(1)

translation_sm_fig(cifar_cnn, aug_type='edge_noise', cifar=True, save=True)
pbar.update(1)

translation_sm_fig(cifar_fcn, aug_type='rotation', cifar=True, save=True)
pbar.update(1)

translation_sm_fig(cifar_fcn, aug_type='noise', cifar=True, save=True)
pbar.update(1)

translation_sm_fig(cifar_fcn, aug_type='edge_noise', cifar=True, save=True)
pbar.update(1)

# ### 2. Variable Connection Pattersn

translation_sm_fig(
    mnist_fcn_vcp.reset_index(),
    aug_type='vcp'
)
pbar.update(1)

# In[24]:


translation_sm_fig(
    cifar_fcn_vcp.reset_index(),
    aug_type='vcp',
    cifar=True
)
pbar.update(1)
