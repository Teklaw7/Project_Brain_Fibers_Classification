from __future__ import print_function
import argparse
import configparser
import glob
import json
import os
from os import path as osp
from os.path import basename as osbn
from time import time

import ants
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#from torch_geometric.data import Batch as gBatch
#from torch_geometric.data import DataListLoader as gDataLoader

#from sDEC import DECSeq
#import datasets as ds
#from utils import ReadSurf , PolyDataToNumpy
from tools import utils
import pytorch3d
import pytorch3d.renderer as pyr
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
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
from Nets.brain_module_cnn_contrastive_labeled import Fly_by_CN

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




def csv_changes(csv_path_train, csv_path_valid, csv_path_test):
    df_train = pd.read_csv(csv_path_train)
    df_valid = pd.read_csv(csv_path_valid)
    df_test = pd.read_csv(csv_path_test)
    
    sample_class_csv = df_test['class'].unique()
    df_train['label'] = df_train['class']
    df_valid['label'] = df_valid['class']
    df_test['label'] = df_test['class']


    for i in range(len(sample_class_csv)):
        df_train['label'] = df_train['label'].replace(sample_class_csv[i], i+1)
        df_valid['label'] = df_valid['label'].replace(sample_class_csv[i], i+1)
        df_test['label'] = df_test['label'].replace(sample_class_csv[i], i+1)

    path_train_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_train_label_to_number.csv"
    path_valid_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_valid_label_to_number.csv"
    path_test_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_test_label_to_number.csv"

    df_train.to_csv(path_train_final)
    df_valid.to_csv(path_valid_final)
    df_test.to_csv(path_test_final)
    
    return path_train_final, path_valid_final, path_test_final

def bounding_box(csv_path, final_path):
    csv_path_train = pd.read_csv(csv_path)
    #print("id",csv_path_train['id'][0])
    csv_path_train['x_min']=0
    csv_path_train['x_max']=0
    csv_path_train['y_min']=0
    csv_path_train['y_max']=0
    csv_path_train['z_min']=0
    csv_path_train['z_max']=0
    for i in range(len(csv_path_train)):
        atlas_path = f"/CMF/data/timtey/UKF/{csv_path_train['id'][i]}_ukf.vtk"
        print(atlas_path)
        atlas = utils.ReadSurf(atlas_path)
        min_max = atlas.GetBounds()
        #print(type(min_max[0]))

        csv_path_train['x_min'][i] = min_max[0]
        csv_path_train['x_max'][i] = min_max[1]
        csv_path_train['y_min'][i] = min_max[2]
        csv_path_train['y_max'][i] = min_max[3]
        csv_path_train['z_min'][i] = min_max[4]
        csv_path_train['z_max'][i] = min_max[5]
        #print(atlas['bounds'])

    csv_path_train.to_csv(final_path)
    
    return final_path
### Part Training ###

### Training call ###

num_classes = 57
nb_epochs = 500
batch_size=10
dropout_lvl=0.1
radius=1
ico_lvl=1
min_delta_early_stopping = 0.00
patience_early_stopping= 5
num_workers=1
path_data="/CMF/data/timtey/tracts/archives"
path_ico = "/NIRAL/tools/atlas/Surface/Sphere_Template/sphere_f327680_v163842.vtk"
train_path="/CMF/data/timtey/tracts/tracts_filtered_train_train.csv"
val_path="/CMF/data/timtey/tracts/tracts_filtered_train_valid.csv"
test_path="/CMF/data/timtey/tracts/tracts_filtered_test.csv"

#path_train_final, path_valid_final, path_test_final = csv_changes(train_path, val_path, test_path)
#path_train_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_test_label_to_number_copy.csv"
#path_valid_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_test_label_to_number_copy.csv"

# path_train_final_b = "/home/timtey/Documents/Projet/dataset2/tracts_filtered_train_train_label_to_number.csv"
# path_valid_final_b = "/home/timtey/Documents/Projet/dataset2/tracts_filtered_train_valid_label_to_number.csv"
# path_test_final_b = "/home/timtey/Documents/Projet/dataset2/tracts_filtered_train_test_label_to_number.csv"

path_train_final = "/home/timtey/Documents/Projet/dataset4/tracts_filtered_train_train_label_to_number.csv"
path_valid_final = "/home/timtey/Documents/Projet/dataset4/tracts_filtered_train_valid_label_to_number.csv"
path_test_final = "/home/timtey/Documents/Projet/dataset4/tracts_filtered_train_test_test_small2.csv"
# path_test_final = "/home/timtey/Documents/Projet/dataset3/tracts_filtered_train_test_test_small.csv"
# path_test_final = "/home/timtey/Documents/Projet/dataset3/tracts_filtered_train_test_label_to_number_divide_by_2.csv"
checkpoint_callback = ModelCheckpoint(
    dirpath='/home/timtey/Documents/Projet/models',
    filename='{epoch}-{val_loss:.2f}',
    monitor='val_loss',
    save_top_k=3
)
#path_valid_final_final = bounding_box(path_valid_final, path_valid_final_b)
#path_test_final_final = bounding_box(path_test_final, path_test_final_b)
#
#path_train_final_final = bounding_box(path_train_final, path_train_final_b)
#path_train_final_b2 = pd.read_csv(path_train_final_b)
#path_train_final_b3 = path_train_final_b2.iloc[:,1:]
#path_train_final_b3.to_csv("/home/timtey/Documents/Projet/dataset3/tracts_filtered_train_train_label_to_number.csv")
#
#path_valid_final_b2 = pd.read_csv(path_valid_final_b)
#path_valid_final_b3 = path_valid_final_b2.iloc[:,1:]
#path_valid_final_b3.to_csv("/home/timtey/Documents/Projet/dataset3/tracts_filtered_train_valid_label_to_number.csv")
#
#path_test_final_b2 = pd.read_csv(path_test_final_b)
#path_test_final_b3 = path_test_final_b2.iloc[:,1:]
#path_test_final_b3.to_csv("/home/timtey/Documents/Projet/dataset3/tracts_filtered_train_test_label_to_number.csv")


contrastive = False

standart_right_brain_path = "/tools/atlas/Surface/CIVET_160K/icbm_surface/icbm_avg_mid_sym_mc_right_hires.vtk"
standart_left_brain_path = "/tools/atlas/Surface/CIVET_160K/icbm_surface/icbm_avg_mid_sym_mc_left_hires.vtk"
standart_right_brain = utils.ReadSurf(standart_right_brain_path)
standart_left_brain = utils.ReadSurf(standart_left_brain_path)
bundle_tf=vtk.vtkTriangleFilter()
bundle_tf.SetInputData(standart_right_brain)
bundle_tf.Update()
bundle_extract_tf_right = bundle_tf.GetOutput()
# print(bundle_extract_tf)
# verts, faces, edges = utils.PolyDataToTensors(bundle_extract_tf)

bundle_tf=vtk.vtkTriangleFilter()
bundle_tf.SetInputData(standart_left_brain)
bundle_tf.Update()
bundle_extract_tf_left = bundle_tf.GetOutput()


verts1, faces1, edges1 = utils.PolyDataToTensors(bundle_extract_tf_left)
verts2, faces2, edges2 = utils.PolyDataToTensors(bundle_extract_tf_right)
# print(bundle_extract_tf_left.GetBounds())
# print(bundle_extract_tf_right.GetBounds())
# print(faces1.shape)
# print(faces2.shape)
def normalize(verts, bounds):
    verts[:,0] = (2*(verts[:,0] - bounds[0])/(bounds[1] - bounds[0])) - 1
    verts[:,1] = (2*(verts[:,1] - bounds[2])/(bounds[3] - bounds[2])) - 1
    verts[:,2] = (2*(verts[:,2] - bounds[4])/(bounds[5] - bounds[4])) - 1

left_bounds = bundle_extract_tf_left.GetBounds()
right_bounds = bundle_extract_tf_right.GetBounds()
Bounds = [left_bounds[0], right_bounds[1], left_bounds[2], right_bounds[3], left_bounds[4], right_bounds[5]]
# print(Bounds)
normalize(verts1, Bounds)
normalize(verts2, Bounds)
# print(verts1)
# print(verts2)
mesh_left = Meshes(verts=[verts1], faces=[faces1])
mesh_right = Meshes(verts=[verts2], faces=[faces2])

df = pd.read_csv(path_test_final)
# for i in range(len(df)):


early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=min_delta_early_stopping, patience=patience_early_stopping, verbose=True, mode='min')
trainer=Trainer(max_epochs=nb_epochs, callbacks=[early_stop_callback, checkpoint_callback], accelerator="gpu")

Y_TRUE = []
Y_PRED = []
Acc = []
brain_data=Bundles_DataModule(contrastive, 0,0,0,0,path_data, path_ico, batch_size, path_train_final, path_valid_final, path_test_final, num_workers=num_workers)
# trainer.fit(model, brain_data)
weights = brain_data.get_weights()
model= Fly_by_CNN(contrastive, radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, verts1, faces1, verts2, faces2, learning_rate=0.001)
# print(model)
model.load_from_checkpoint("/home/timtey/Documents/Projet/models/Resnet/All/epoch=38-val_loss=0.52.ckpt")### marche pas pour models efficientnet que pour resnet18 car le modele est pa enregistre avec l argument contrastive
# print(model)
for index_csv in range(len(df)):
    path = f"/CMF/data/timtey/tracts/{df['surf'][index_csv]}"
    # print(path)
    bundle = utils.ReadSurf(path)

    fibers = bundle.GetNumberOfCells()
    # print(fibers)

    brain_data=Bundles_DataModule(contrastive, bundle, fibers, 0, index_csv, path_data, path_ico, batch_size, path_train_final, path_valid_final, path_test_final, num_workers=num_workers)

    

    trainer.test(model, brain_data)
    
    label_true = model.get_y_true()
    label_pred = model.get_y_pred()
    Y_TRUE.append(label_true)
    Y_PRED.append(label_pred)
    print(Y_TRUE)
    print(Y_PRED)
    Acc.append(model.get_accuracy())
    print(Acc)
    Acc_tot_iteration = np.mean(Acc)
    print(Acc_tot_iteration)
    # label_true_t = torch.tensor(label_true)
    # label_pred_t = torch.tensor(label_pred)
    # torch.save(label_true_t, f"/home/timtey/Documents/Projet/label_true/label_true_{index_csv}.pt")
    # torch.save(label_pred_t, f"/home/timtey/Documents/Projet/label_pred/label_pred_{index_csv}.pt")
Y_TRUE = [item for sublist in Y_TRUE for item in sublist]
# print("Y_TRUE", Y_TRUE)
Y_PRED = [item for sublist in Y_PRED for item in sublist]
# print("Y_PRED", Y_PRED)

print(classification_report(Y_TRUE, Y_PRED))
list_class = list(range(0,56))
confmat = confusion_matrix(y_true=Y_TRUE, y_pred=Y_PRED)
cmn = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
fig = px.imshow(cmn,labels=dict(x="Predicted condition", y="Actual condition"),x=list_class,y=list_class)
fig.update_xaxes(side="top")
fig.write_image("/home/timtey/Documents/Projet/confusion_matrix/confusion_matrix.png")
print(Acc)
Acc_total = np.mean(Acc)
print(Acc_total)
