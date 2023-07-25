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
from Data_Loaders.data_module_contrastive_tractography_labeled import Bundles_DataModule_tractography_labeled_fibers, Bundles_Dataset_test_contrastive_tractography_labeled, Bundles_Dataset_tractography
from Data_Loaders.data_module_contrastive_labeled import Bundles_Dataset_contrastive_labeled  #same for classification
from tools.pad import pad_verts_faces, pad_verts_faces_simple
from Nets.brain_module_cnn_contrastive_tractography_labeled import Fly_by_CNN_contrastive_tractography_labeled
from Nets.brain_module_cnn_contrastive_labeled import Fly_by_CNN_contrastive_labeled
from Nets.brain_module_cnn import Fly_by_CNN
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
path_ico = "/NIRAL/tools/atlas/Surface/Sphere_Template/sphere_f327680_v163842.vtk"
path_tractography_train = "/home/timtey/Documents/datasets/dataset4/tractography_3_train.csv"
path_tractography_valid = "/home/timtey/Documents/datasets/dataset4/tractography_3_valid.csv"
path_tractography_test = "/home/timtey/Documents/datasets/dataset4/tractography_3_test.csv"
path_train_final = "/home/timtey/Documents/datasets/dataset4/tracts_filtered_train_train_label_to_number_without_missing.csv"
path_valid_final = "/home/timtey/Documents/datasets/dataset4/tracts_filtered_train_valid_label_to_number_without_missing.csv"
path_test_final = "/home/timtey/Documents/datasets/dataset4/tracts_filtered_train_test_label_to_number_nb_cells_without_missing_2_part.csv"

# checkpoint_callback = ModelCheckpoint(
#     dirpath='/home/timtey/Documents/Models_tensorboard/models/Loss_combine',
#     filename='{epoch}-{val_loss:.2f}',
#     monitor='val_loss',
#     save_top_k=3
# )

path_tract_dataset = "/home/timtey/Documents/datasets/dataset4/tractography_3.csv"
df_tract_dataset = pd.read_csv(path_tract_dataset)
tractography_list_vtk = []
for i in range(len(df_tract_dataset)):
    tractography_list_vtk.append(utils.ReadSurf(df_tract_dataset["surf"][i]))
    print("ligne ", i+1, "/", len(df_tract_dataset)," done")
print("Number of tracts: ", len(tractography_list_vtk))

# contrastive = True

df = pd.read_csv(path_test_final)
logger = TensorBoardLogger(save_dir="/home/timtey/Documents/Models_tensorboard/tensorboard_photos", name='Resnet')
image_logger = BrainNetImageLogger_contrastive_tractography_labeled(num_features = 3,num_images = 24,mean = 0)

# early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=min_delta_early_stopping, patience=patience_early_stopping, verbose=True, mode='min')
# trainer=Trainer(log_every_n_steps=5, max_epochs=nb_epochs, logger = logger, callbacks=[image_logger], accelerator="gpu")
# Y_TRUE = []
# Y_PRED = []
# Acc = []
# Acc_details = []
trainer = Trainer(log_every_n_steps=5, logger = logger, callbacks = [image_logger],accelerator="gpu")
# brain_data=Bundles_DataModule_tractography_labeled_fibers(contrastive, 0,0,0,0,0,path_data, path_ico, batch_size, path_train_final, path_valid_final, path_test_final, verts_brain, faces_brain, face_features_brain, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)
brain_data=Bundles_DataModule_tractography_labeled_fibers(0,0,0,path_data, batch_size, path_train_final, path_valid_final, path_test_final, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)

weights = brain_data.get_weights()
# model= Fly_by_CNN_contrastive_tractography_labeled(contrastive, radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, verts_left, faces_left, verts_right, faces_right, learning_rate=0.001)
# model= Fly_by_CNN_contrastive_tractography_labeled(contrastive, radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, learning_rate=0.001)
model_path ="/home/timtey/Documents/Models_tensorboard/models/Loss_combine/071823/epoch=74-val_loss=-0.69.ckpt"

model= Fly_by_CNN_contrastive_tractography_labeled(radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, learning_rate=0.0001)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
# model = Fly_by_CNN_contrastive_tractography_labeled.load_from_checkpoint(model_path)
# model.cuda()
# model.eval()

# trainer.fit(model, brain_data)
# trainer.test(model, brain_data)
list_test_data = pd.read_csv(path_test_final)
list_test_tractography_data = pd.read_csv(path_tractography_test)
for index_csv in range(len(df)):
    path = f"/CMF/data/timtey/tracts/archives/{df['id'][index_csv]}_tracts/{df['class'][index_csv]}_DTI.vtk"
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
    # fibers = bundle.GetNumberOfCells()
    fibers = min(df['num_cells'])

    # brain_data=Bundles_DataModule_tractography_labeled_fibers(contrastive, bundle, L, fibers, 0, index_csv, path_data, path_ico, batch_size, path_train_final, path_valid_final, path_test_final, verts_brain, faces_brain, face_features_brain, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)
    # brain_data=Bundles_DataModule_tractography_labeled_fibers(contrastive, bundle, L, fibers, 0, index_csv, path_data, path_ico, batch_size, path_train_final, path_valid_final, path_test_final, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)
    # test_dataset = Bundles_Dataset_test_contrastive_tractography_labeled(list_test_data, L, fibers, index_csv)
    # test_tractography_dataset = Bundles_Dataset_tractography(list_test_tractography_data, tractography_list_vtk)
    brain_data=Bundles_DataModule_tractography_labeled_fibers(L, fibers, index_csv, path_data, batch_size, path_train_final, path_valid_final, path_test_final, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)
    
    # test_data = DataLoader(test_dataset, batch_size=1, collate_fn=pad_verts_faces_simple, shuffle=False, num_workers=8)
    # print(test_data, type(test_data))
    trainer.test(model, brain_data)
    # for idx, batch in enumerate(test_data):
        # print("batch",batch)
        # print("idx",idx)
        # proj = model(batch)



    # trainer.test(model, brain_data)
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

