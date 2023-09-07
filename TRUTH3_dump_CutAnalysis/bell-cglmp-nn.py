#!/usr/bin/env python
# coding: utf-8

# ## CGLMP NN
# 
# This is the truth-level NN codes gor Bell Study. Modify from Siang-Yuan's notebook.
# 
# > Author : Y.Y. Gary Peng, S.Y. Lin  
# > Istitute: National Tsing Hua University, Taiwan and ATLAS experiment, CERN  
# > License : MIT  
# > Data : Aug. 03, 2023  
# > Email : yuan-yen.peng@cern.ch
# 
# Version : 5.0
# 

# In[1]:


get_ipython().run_line_magic('reset', '-f')


# In[2]:


import os
import gc
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from numba import njit
import matplotlib.pyplot as plt


# In[3]:


def get_files_names(path):
    files_name = glob.glob(path)
    return files_name

path = "./truth/*/*.npz"
files_name = get_files_names(path)
print(files_name)

def get_data(path):
    try:
        with np.load(path, allow_pickle=True) as f:
            data_dict = {name: f[name] for name in f.files}
            return pd.DataFrame(data_dict)
    except FileNotFoundError:
        print("File not found!")
        return pd.DataFrame()
    
files = []
files_name.sort()
for f in files_name:
    files.append(get_data(f))
    
# need to check the order of data name.
print(files_name)

# need to follow the order of data name.
(
    CGLMP,
    Higgs,
    LeadLep,
    LepM,
    LepP,
    NuM,
    NuP,
    MET,
    Wm,
    Wp,
    diLep,
    SubLep,
    Xi,
) = files
GEV=1e3

del files_name, files
gc.collect()


# In[4]:


# Some constants
GEV = 1e3
# RMV_EVT = [638488, 835579, 2168342] # escape some mathmetical errors.
RMV_EVT = [] # escape some mathmetical errors.


# In[5]:


# # DiNu info.
# DUNI_M = 34.141
# dinu_kin = pd.DataFrame({
#     'lep_p_M' : DUNI_M,
#     'lep_p_px': MET['px'],
#     'lep_p_py': MET['py'],
#     'lep_p_pz': diNu_pz,
# })

# # check format l+ -> (E, px, py, pz); then, append l- with the same format of l+.
# print(dinu_kin.shape)
# dinu_kin.drop(RMV_EVT, inplace=True)
# print(dinu_kin.shape)
# dinu_kin.head(5)


# In[6]:


# Kinemetic info of leptons.
lep_kin = pd.DataFrame({
    'lep_p_E' : LepP['E'],
    'lep_p_px': LepP['px'],
    'lep_p_py': LepP['py'],
    'lep_p_pz': LepP['pz'],
    'lep_m_E' : LepM['E'],
    'lep_m_px': LepM['px'],
    'lep_m_py': LepM['py'],
    'lep_m_pz': LepM['pz'],
})/GEV

# check format l+ -> (E, px, py, pz); then, append l- with the same format of l+.
print(lep_kin.shape)
lep_kin.drop(RMV_EVT, inplace=True)
print(lep_kin.shape)
lep_kin.head(5)


# In[7]:


# Kinemetic info of neutirnos.
nu_kin = pd.DataFrame({
    'nu_p_E' : NuP['E'],
    'nu_p_px': NuP['px'],
    'nu_p_py': NuP['py'],
    'nu_p_pz': NuP['pz'],
    'nu_m_E' : NuM['E'],
    'nu_m_px': NuM['px'],
    'nu_m_py': NuM['py'],
    'nu_m_pz': NuM['pz'],
})/GEV

# check format nu+ -> (E, px, py, pz); then, append nu- with the same format of l+.
print(nu_kin.shape)
nu_kin.drop(RMV_EVT, inplace=True)
print(nu_kin.shape)
nu_kin.head(5)


# In[8]:


# CGLMP.
CGLMP = pd.DataFrame({
    'Bxy' : CGLMP['Bxy'],
    'Byz' : CGLMP['Byz'],
    'Bzx' : CGLMP['Bzx'],
})

# check
print(CGLMP.shape)
CGLMP.drop(RMV_EVT, inplace=True)
print(CGLMP.shape)
CGLMP.head(5)


# ## The CGLMP expectation value:

# \begin{align*}
# \mathrm{tr}(\rho \mathcal{B}_\text{CGLMP}^{xy})
# = \frac{8}{ \sqrt{3} } & \langle \xi_x^+\xi_x^- + \xi_x^+\xi_x^- \rangle \\
# + 25 & \langle [{\xi_x^+}^2 - {\xi_y^+}^2][{\xi_x^-}^2 - {\xi_y^-}^2] \rangle \\
# + 100 & \langle \xi_x^+ \xi_y^+ \xi_x^- \xi_y^- \rangle
# \end{align*}
# 
# 
# \begin{equation*}
# \mathcal{I}_3 = \mathrm{max}(
# \langle \mathcal{B}_\text{CGLMP}^{xy} \rangle,
# \langle \mathcal{B}_\text{CGLMP}^{yz} \rangle,
# \langle \mathcal{B}_\text{CGLMP}^{zx} \rangle )
# \end{equation*}
# 
# 
# The ensemble of decays has rotational symmetry in the Higgs boson rest frame around the direction of the W boson momenta.  
# Hence in this Letter, rather than testing every possible pair of axes 
# we **choose a set of Cartesian coordinates**  
# in which **one axis is aligned with this privileged direction**,  
# **construct expectation values for each** of the $(x, y)$, $(y, z)$ and $(z, x)$ **pairs of axes**,  
# and **compare the largest of them**.

# ## Baseline model: $\mathcal{B}_{*z}$ Regression
# Baseline model uses only *low level features*
# * Features:
#     * four-vector component ($E, p_x, p_y, p_z$)
#         * alternate four-vector ($M, p_T, \eta, \phi$)
# * Input particles (available four-vectors):
#     * Positive/Negative charged leptons
#         * leading/sub-leading lepton
#     * Di-Lepton system
#     * Missing Transverse momentum
#         * $p_x$, $p_y$
#         * $p_T$, $\phi$
# 
# [Visualize neural network](http://alexlenail.me/NN-SVG/index.html)

# In[9]:


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
print(tf.config.list_physical_devices())
from tensorflow import keras
from tensorflow.keras import layers
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID' # If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
print(tf.__version__)

# install lbn
# !pip install lbn --upgrade
from lbn import LBN, LBNLayer

# # install auto ml (auto keras)
# !pip install git+https://github.com/keras-team/keras-tuner.git
# !pip install autokeras
# import autokeras as ak

# install keras tuner
# !pip install keras-tuner --upgrade
import keras_tuner as kt


# In[10]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.preprocessing import StandardScaler\n\n# Bxy\n# BXY_MAX = CGLMP[\'Bxy\'].max(axis=None)\n# BXY_MIN = CGLMP[\'Bxy\'].min(axis=None)                     \n# Bxy_norm = (CGLMP[\'Bxy\'] - BXY_MIN)/(BXY_MAX - BXY_MIN)\nSCALAR = StandardScaler()\nBxy_std = SCALAR.fit_transform([[x] for x in CGLMP[\'Bxy\']]).flatten() # zero mean and unit variance\nplt.hist(Bxy_std, bins=50)\nplt.title("Before reweighting Bxy_std")\nplt.show()\nprint(Bxy_std.max(), Bxy_std.min())\nprint("std:", Bxy_std.std())\nprint(len(Bxy_std))\n')


# In[11]:


# Undersampling to an uniform distribution

n = int(1e2)
step = (Bxy_std.max() - Bxy_std.min())/n
intervals = [Bxy_std.min() + i*step for i in range(n)]

sampling = int(1e3)
indices_arr = np.empty((n-1, sampling), dtype=int)
for i in range(n-1):
    try:
        indices_arr[i] = np.random.choice(np.where((intervals[i]<Bxy_std)*(Bxy_std<=intervals[i+1]))[0], size=sampling, replace=False)
    except ValueError:
        print("Cannot take a larger sample than population when 'replace=False")


# In[12]:


from sklearn.model_selection import train_test_split

lepton_features   = ['lep_p_E','lep_p_px','lep_p_py','lep_p_pz',
                     'lep_m_E','lep_m_px','lep_m_py','lep_m_pz']
neutrino_features = ['nu_p_E','nu_p_px','nu_p_py','nu_p_pz',
                     'nu_m_E','nu_m_px','nu_m_py','nu_m_pz']

def reshape_features(inputs, features):
    outputs = np.stack([inputs[features[0:4]],
                        inputs[features[4:8]]], axis=1)
    return outputs

indices = np.arange(int(1e5)) # sampling size
train_indices, temp_indices = train_test_split(indices_arr.flatten(), train_size=0.8, test_size=0.2, random_state=42)
valid_indices, test_indices = train_test_split(temp_indices, train_size=0.5, test_size=0.5, random_state=42)

lep_train = reshape_features(lep_kin.iloc[train_indices], lepton_features)
lep_valid = reshape_features(lep_kin.iloc[valid_indices], lepton_features)
lep_test  = reshape_features(lep_kin.iloc[test_indices],  lepton_features)

nu_train = reshape_features(nu_kin.iloc[train_indices], neutrino_features)
nu_valid = reshape_features(nu_kin.iloc[valid_indices], neutrino_features)
nu_test  = reshape_features(nu_kin.iloc[test_indices],  neutrino_features)

Bxy_train = Bxy_std[train_indices]
Bxy_valid = Bxy_std[valid_indices]
Bxy_test  = Bxy_std[test_indices]
plt.hist(Bxy_train, bins=50)
plt.title("Train (Bxy)")
plt.show()
plt.hist(Bxy_valid, bins=50)
plt.title("Valid (Bxy)")
plt.show()
plt.hist(Bxy_test, bins=50)
plt.title("Test (Bxy)")
plt.show()

def stack_parts(input1, input2):
    outputs = np.concatenate([input1, input2], axis=1)
    return outputs

if (lep_train.shape == nu_train.shape and 
    lep_test.shape  == nu_test.shape and 
    lep_valid.shape == nu_valid.shape):
    print('With the same shapes...\n')
    # print(f'Training data shape: {lep_train.shape}\nTesting data shape: {lep_test.shape}\nValidation data shape: {lep_valid.shape}')
    train = stack_parts(lep_train, nu_train)
    valid = stack_parts(lep_valid, nu_valid)
    test  = stack_parts(lep_test , nu_test)
    FEA_MAX = np.max([train.max(), valid.max(), test.max()])
    FEA_MIN = np.min([train.min(), valid.min(), test.min()])
    train = (train - FEA_MIN)/(FEA_MAX - FEA_MIN)
    train = np.tile(
        np.repeat(np.expand_dims(train, axis=-1), 3, axis=-1), 
        (1,8,8,1),
    )
    valid = (valid - FEA_MIN)/(FEA_MAX - FEA_MIN)
    valid = np.tile(
        np.repeat(np.expand_dims(valid, axis=-1), 3, axis=-1), 
        (1,8,8,1),
    )
    test = (test - FEA_MIN)/(FEA_MAX - FEA_MIN)
    test = np.tile(
        np.repeat(np.expand_dims(test, axis=-1), 3, axis=-1), 
        (1,8,8,1),
    )
    print(print(f'Training data shape: {train.shape}\nTesting data shape: {test.shape}\nValidation data shape: {valid.shape}'))
    
    del (indices, train_indices, temp_indices, valid_indices, test_indices,
        lep_train, lep_valid, lep_test,
        nu_train, nu_valid, nu_test)
    gc.collect()
    
else:
    print('The shape of leptons are NOT the same with neutrinos shape...\n')
    print(f'Training data shape: {lep_train.shape}, {nu_train.shape}\nTesting data shape: {lep_test.shape}, {nu_test.shape}\nValidation data shape: {lep_valid.shape}, {nu_valid.shape}')


# In[13]:


# Create random data for 50 figures (4x4 matrices)
num_figures = 30
figure_size = 4

# Calculate the grid dimensions (rows x columns)
num_rows = 5
num_cols = 6

# Create a subplot grid
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through the figures and plot them on subplots
for i, ax in enumerate(axes):
    ax.imshow(train[i,:,:,0], cmap='viridis', origin='lower', vmin=train[0:num_figures+1,:,:,0].min(), vmax=train[0:num_figures+1,:,:,0].max())
    ax.set_title(f'(0){Bxy_train[i]:2f}')
    ax.axis('off')  # Turn off axis labels
plt.tight_layout()
plt.show()


# ### Model define

# In[14]:


train = np.concatenate((train, valid), axis=0)
Bxy_train = np.concatenate((Bxy_train, Bxy_valid), axis=0)
del valid, Bxy_valid


# In[17]:


from sklearn.model_selection import GridSearchCV, KFold
from tensorflow.keras import models
from scikeras.wrappers import KerasRegressor

def build_model(node=16, drop_rate=0.3, learning_rate=0.001):
    base_model = tf.keras.applications.ConvNeXtSmall(include_top=False, input_shape=train.shape[1:])
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(drop_rate),
        layers.Dense(node, activation='elu', kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(drop_rate),
        layers.Dense(node, activation='elu', kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)), 
        layers.Dropout(drop_rate),
        layers.Dense(1, activation='linear')  # Regression output layer
    ])
    # model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse')
    return model           

model = KerasRegressor(model=build_model, epochs=10, node=16, drop_rate=0.3, learning_rate=0.001)

folds = KFold(n_splits=10, shuffle=True, random_state=42)
param_grid = {
    "epochs": [10, 20, 30],
    'node': [8, 16, 32, 64],
    'drop_rate': [0.2, 0.3, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1]
}
model_cv = GridSearchCV(estimator=model, 
                        param_grid=param_grid, 
                        cv=folds, 
                        return_train_score=True)

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0.001,
                                            patience=10,
                                            mode='auto',
                                            baseline=None,
                                            restore_best_weights=False)

history = model_cv.fit(
    train, 
    Bxy_train,
    # validation_data=(valid, Bxy_valid), 
    # epochs=50, 
    batch_size=512, 
    callbacks=stop_early
    )


# In[ ]:


Bxy_pred = model_cv.predict(test)


# In[ ]:


from sklearn.metrics import mean_squared_error

print(f'''RMSE: {mean_squared_error(Bxy_test, Bxy_pred)}
MAX of pred: {Bxy_pred.max()}; MIN of pred: {Bxy_pred.min()}
MAX of test: {Bxy_test.max()}; MIN of test: {Bxy_test.min()}''')

# Plot the results
fig = plt.figure(figsize=(5,5), dpi=120)
ax = fig.add_subplot()
plt.plot(Bxy_pred, Bxy_test, '.', color="tab:blue", alpha=.3, markersize=3)
plt.xlim([-2.5,2.5])
plt.ylim([-2.5,2.5])
plt.xlabel('pred')
plt.ylabel('true')
ax.set_aspect('equal', adjustable='box')
plt.show() 
plt.close()

fig = plt.figure(figsize=(8,5), dpi=120)
plt.plot(history.history['loss'], lw=2.5, label='Train', alpha=.8)
plt.plot(history.history['val_loss'], lw=2.5, label='Validation', alpha=.8)
plt.semilogy()
plt.title('Epoch vs MSE')
plt.xlabel('epoch')
plt.ylabel('Loss (MSE)')
plt.legend(loc='best')
plt.show()
plt.close()


# In[ ]:




