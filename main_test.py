from __future__ import print_function
import os
from os import path as osp
from os.path import basename as osbn
from time import time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import torch
from logger import BrainNetImageLogger_contrastive_tractography_labeled, BrainNetImageLogger
from pytorch_lightning.loggers import TensorBoardLogger

from tools import utils
import vtk           

from Data_Loaders.data_module_contrastive_tractography_labeled import Bundles_DataModule_tractography_labeled_fibers    # used for clustering with labeled and tractography fibers
from Data_Loaders.data_module_classification_or_contrastive_labeled import Bundles_Dataset_contrastive_labeled  # used for classification and contrastive learning with Simclr

from Nets.brain_module_cnn_contrastive_tractography_labeled import Fly_by_CNN_contrastive_tractography_labeled  # used for clustering with labeled and tractography fibers
from Nets.brain_module_cnn_contrastive_labeled import Fly_by_CNN_contrastive_labeled    # used for contrastive learning with Simclr
from Nets.brain_module_cnn import Fly_by_CNN    # used for classification

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import plotly.express as px
from Transformations.transformations import *

### Initalization ###

num_classes = 57
nb_epochs = 500
batch_size= 10
dropout_lvl=0.1
radius=1
ico_lvl=1
min_delta_early_stopping = 0.00
patience_early_stopping= 10
num_workers=12
path_data="/CMF/data/timtey/tracts/archives"
path_tractography_train = "/CMF/data/timtey/datasets/tractography_3_train.csv"
path_tractography_valid = "/CMF/data/timtey/datasets/tractography_3_valid.csv"
path_tractography_test = "/CMF/data/timtey/datasets/tractography_3_test.csv"
path_train_final = "/CMF/data/timtey/datasets/tracts_filtered_train_train_label_to_number_without_missing.csv"
path_valid_final = "/CMF/data/timtey/datasets/tracts_filtered_train_valid_label_to_number_without_missing.csv"
path_test_final = "/CMF/data/timtey/datasets/tracts_filtered_train_test_label_to_number_nb_cells_without_missing_2_part.csv"

path_tract_dataset = "/CMF/data/timtey/datasets/tractography_3.csv"
df_tract_dataset = pd.read_csv(path_tract_dataset)
tractography_list_vtk = []
for i in range(len(df_tract_dataset)):
    tractography_list_vtk.append(utils.ReadSurf(df_tract_dataset["surf"][i]))
    print("ligne ", i+1, "/", len(df_tract_dataset)," done")
print("Number of tracts: ", len(tractography_list_vtk))

df = pd.read_csv(path_test_final)
logger = TensorBoardLogger(save_dir="/home/timtey/Documents/Models_tensorboard/tensorboard_photos", name='Resnet') # to change the folder for tensorboardlogger

#used for clustering with labeled and tractography fibers
image_logger = BrainNetImageLogger_contrastive_tractography_labeled(num_features = 3,num_images = 24,mean = 0)
#used for classification and contrastive learning with Simclr
# image_logger = BrainNetImageLogger(num_features = 3,num_images = 24,mean = 0)

# These four lists are used to compute the accuracy during the test when you want to use the classification
# Y_TRUE = []
# Y_PRED = []
# Acc = []
# Acc_details = []

trainer = Trainer(log_every_n_steps=5, logger = logger, callbacks = [image_logger],accelerator="gpu")
#Data_Module used for clustering with labeled and tractography fibers
brain_data=Bundles_DataModule_tractography_labeled_fibers(0,0,0,path_data, batch_size, path_train_final, path_valid_final, path_test_final, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)
#Data_Module used for classification and contrastive learning with Simclr
# brain_data=Bundles_Dataset_contrastive_labeled(0,0,0, batch_size, path_train_final, path_valid_final, path_test_final, num_workers=num_workers)

weights = brain_data.get_weights()
model_path ="/home/timtey/Documents/Models_tensorboard/models/Loss_combine/071823/epoch=74-val_loss=-0.69.ckpt" # to change the path for the trained model
#model used for clustering with labeled and tractography fibers
model= Fly_by_CNN_contrastive_tractography_labeled(radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, learning_rate=0.0001)
#model used for classification
# model= Fly_by_CNN(radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, learning_rate=0.0001)
#model used for contrastive learning with Simclr
# model= Fly_by_CNN_contrastive_labeled(radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, learning_rate=0.0001)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
for index_csv in range(len(df)):
    path = f"/CMF/data/timtey/tracts/archives/{df['id'][index_csv]}_tracts/{df['class'][index_csv]}_DTI.vtk"    # path used for clustering with labeled and tractography fibers
    # path = f"/CMF/data/timtey/tracts/archives/{df['id'][index_csv]}_tracts/{df['class'][index_csv]}.vtp"    # path used for classification and contrastive learning with Simclr
    # create a dataloader for this part to load and create the list of all the fibers as tubes
    bundle = utils.ReadSurf(path)
    L = []
    for i in range(bundle.GetNumberOfCells()):
        fiber = utils.ExtractFiber(bundle,i)
        cc1_tf=vtk.vtkTriangleFilter()
        cc1_tf.SetInputData(fiber)
        cc1_tf.Update()
        fiber_tf = cc1_tf.GetOutput()
        L.append(fiber_tf)
    fibers = min(df['num_cells'])

    #Network used for clustering with labeled and tractography fibers
    brain_data=Bundles_DataModule_tractography_labeled_fibers(L, fibers, index_csv, path_data, batch_size, path_train_final, path_valid_final, path_test_final, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)
    #Network used for classification and contrastive learning with Simclr
    # brain_data=Bundles_Dataset_contrastive_labeled(L, fibers, index_csv, batch_size, path_train_final, path_valid_final, path_test_final, num_workers=num_workers)
    
    trainer.test(model, brain_data)

# This part is to add if you want to do a classification, it will give you the accuracy of the classification test for each bundle and also the global accuracy, it also creates the confusion matrix
'''    
    label_true = model.get_y_true()
    label_pred = model.get_y_pred()
    Y_TRUE.append(label_true)
    Y_PRED.append(label_pred)
    Acc.append(model.get_accuracy())
    Acc_details.append([label_true[0],model.get_accuracy()])
    tensor_pred = torch.tensor(label_pred)
    print(tensor_pred.unique(return_counts=True))
    print(Acc_details)
    print(len(Acc_details))
    Acc_tot_iteration = np.mean(Acc)
    print(Acc_tot_iteration)

Y_TRUE = [item for sublist in Y_TRUE for item in sublist]
print(len(Y_TRUE))
Y_PRED = [item for sublist in Y_PRED for item in sublist]
print(len(Y_PRED))
print(classification_report(Y_TRUE, Y_PRED))
list_class = list(range(0,57))#########3
confmat = confusion_matrix(y_true=Y_TRUE, y_pred=Y_PRED)
cmn = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
fig = px.imshow(cmn,labels=dict(x="Predicted condition", y="Actual condition"),x=list_class,y=list_class)
fig.update_xaxes(side="top")
fig.write_image("/home/timtey/Documents/Projet/confusion_matrix/confusion_matrix.png")
print(Acc)
Acc_total = np.mean(Acc)
print(Acc_total)
'''