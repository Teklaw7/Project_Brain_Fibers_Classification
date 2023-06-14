from __future__ import print_function
import argparse
import configparser
import glob
import json
import os
from os import path as osp
from os.path import basename as osbn
from time import time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import ants
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from logger import BrainNetImageLogger_contrastive_tractography_labeled, BrainNetImageLogger
from pytorch_lightning.loggers import TensorBoardLogger
#from torch_geometric.data import Batch as gBatch
#from torch_geometric.data import DataListLoader as gDataLoader
from vtkmodules.vtkFiltersGeneral import (
    vtkCurvatures,
    vtkTransformFilter
)
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkFiltersCore import (
    vtkFeatureEdges,
    vtkIdFilter
)
from vtk.util import numpy_support
from vtkmodules.vtkCommonCore import (
    VTK_DOUBLE,
    vtkIdList,
    vtkVersion
)
#from sDEC import DECSeq
#import datasets as ds
#from utils import ReadSurf , PolyDataToNumpy
from tools import utils
import pytorch3d
import pytorch3d.renderer as pyr
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.vis.plotly_vis import plot_scene
import vtk
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

from Data_Loaders.data_module_contrastive_tractography_labeled import Bundles_DataModule_tractography_labeled_fibers
from Data_Loaders.data_module_contrastive_labeled import Bundles_Dataset_contrastive_labeled  #same for classification

from Nets.brain_module_cnn_contrastive_tractography_labeled import Fly_by_CNN_contrastive_tractography_labeled
from Nets.brain_module_cnn_contrastive_labeled import Fly_by_CNN_contrastive_labeled
from Nets.brain_module_cnn import Fly_by_CNN

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
patience_early_stopping= 15
num_workers=12
path_data="/CMF/data/timtey/tracts/archives"
# path_ico = "/NIRAL/tools/atlas/Surface/Sphere_Template/sphere_f327680_v163842.vtk"
path_tractography_train = "/home/timtey/Documents/datasets/dataset4/tractography_2_train.csv"
path_tractography_valid = "/home/timtey/Documents/datasets/dataset4/tractography_2_valid.csv"
path_tractography_test = "/home/timtey/Documents/datasets/dataset4/tractography_2_test.csv"
path_train_final = "/home/timtey/Documents/datasets/dataset4/tracts_filtered_train_train_label_to_number_without_missing.csv"
path_valid_final = "/home/timtey/Documents/datasets/dataset4/tracts_filtered_train_valid_label_to_number_without_missing.csv"
path_test_final = "/home/timtey/Documents/datasets/dataset4/tracts_filtered_train_test_label_to_number_nb_cells_without_missing_2_part.csv"

checkpoint_callback = ModelCheckpoint(
    dirpath='/home/timtey/Documents/Models_tensorboard/models/Loss_combine/061423',
    filename='{epoch}-{val_loss:.2f}',
    monitor='val_loss',
    save_top_k=3
)

tractography_list_vtk = []
# tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg.vtp"))
# tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_103515_dg.vtp"))
# tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_108525_dg.vtp"))
# tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_113215_dg.vtp"))
# tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_119833_dg.vtp"))
# tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_121618_dg.vtp"))
# tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_124220_dg.vtp"))
# tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_124826_dg.vtp"))
# tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_139233_dg.vtp"))

# contrastive = True

df = pd.read_csv(path_test_final)
logger = TensorBoardLogger(save_dir="/home/timtey/Documents/Models_tensorboard/tensorboard_photos", name='Resnet')
image_logger = BrainNetImageLogger_contrastive_tractography_labeled(num_features = 3,num_images = 24,mean = 0)

early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=min_delta_early_stopping, patience=patience_early_stopping, verbose=True, mode='min')
trainer=Trainer(log_every_n_steps=5, max_epochs=nb_epochs, logger = logger, callbacks=[early_stop_callback, checkpoint_callback, image_logger], accelerator="gpu")
# Y_TRUE = []
# Y_PRED = []
# Acc = []
# Acc_details = []

# brain_data=Bundles_DataModule_tractography_labeled_fibers(contrastive, 0,0,0,0,0,path_data, path_ico, batch_size, path_train_final, path_valid_final, path_test_final, verts_brain, faces_brain, face_features_brain, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)
brain_data=Bundles_DataModule_tractography_labeled_fibers(0,0,0,path_data, batch_size, path_train_final, path_valid_final, path_test_final, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)

weights = brain_data.get_weights()
# model= Fly_by_CNN_contrastive_tractography_labeled(contrastive, radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, verts_left, faces_left, verts_right, faces_right, learning_rate=0.001)
model= Fly_by_CNN_contrastive_tractography_labeled(radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, learning_rate=0.0001)

trainer.fit(model, brain_data)
# trainer.test(model, brain_data)
for index_csv in range(len(df)):
    # path = f"/CMF/data/timtey/tracts/{df['surf'][index_csv]}"
    path = f"/CMF/data/timtey/tracts/archives/{df['id'][index_csv]}_tracts/{df['class'][index_csv]}_DTI.vtk"
    bundle = utils.ReadSurf(path)
    L = []
    for i in range(bundle.GetNumberOfCells()):
        fiber = utils.ExtractFiber(bundle,i)
        cc1_tf=vtk.vtkTriangleFilter()
        cc1_tf.SetInputData(fiber)
        cc1_tf.Update()
        fiber_tf = cc1_tf.GetOutput()
        L.append(fiber_tf)
    # fibers = bundle.GetNumberOfCells()
    fibers = min(df['num_cells'])

    # brain_data=Bundles_DataModule_tractography_labeled_fibers(contrastive, bundle, L, fibers, 0, index_csv, path_data, path_ico, batch_size, path_train_final, path_valid_final, path_test_final, verts_brain, faces_brain, face_features_brain, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)
    brain_data=Bundles_DataModule_tractography_labeled_fibers(L, fibers, index_csv, path_data, batch_size, path_train_final, path_valid_final, path_test_final, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)

    trainer.test(model, brain_data)
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

