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



# def csv_changes(csv_path_train, csv_path_valid, csv_path_test):
#     df_train = pd.read_csv(csv_path_train)
#     df_valid = pd.read_csv(csv_path_valid)
#     df_test = pd.read_csv(csv_path_test)
    
#     sample_class_csv = df_test['class'].unique()
#     df_train['label'] = df_train['class']
#     df_valid['label'] = df_valid['class']
#     df_test['label'] = df_test['class']


#     for i in range(len(sample_class_csv)):
#         df_train['label'] = df_train['label'].replace(sample_class_csv[i], i+1)
#         df_valid['label'] = df_valid['label'].replace(sample_class_csv[i], i+1)
#         df_test['label'] = df_test['label'].replace(sample_class_csv[i], i+1)

#     path_train_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_train_label_to_number.csv"
#     path_valid_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_valid_label_to_number.csv"
#     path_test_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_test_label_to_number.csv"

#     df_train.to_csv(path_train_final)
#     df_valid.to_csv(path_valid_final)
#     df_test.to_csv(path_test_final)
    
#     return path_train_final, path_valid_final, path_test_final

# def bounding_box(csv_path, final_path):
#     csv_path_train = pd.read_csv(csv_path)
#     #print("id",csv_path_train['id'][0])
#     csv_path_train['x_min']=0
#     csv_path_train['x_max']=0
#     csv_path_train['y_min']=0
#     csv_path_train['y_max']=0
#     csv_path_train['z_min']=0
#     csv_path_train['z_max']=0
#     for i in range(len(csv_path_train)):
#         atlas_path = f"/CMF/data/timtey/UKF/{csv_path_train['id'][i]}_ukf.vtk"
#         print(atlas_path)
#         atlas = utils.ReadSurf(atlas_path)
#         min_max = atlas.GetBounds()
#         #print(type(min_max[0]))

#         csv_path_train['x_min'][i] = min_max[0]
#         csv_path_train['x_max'][i] = min_max[1]
#         csv_path_train['y_min'][i] = min_max[2]
#         csv_path_train['y_max'][i] = min_max[3]
#         csv_path_train['z_min'][i] = min_max[4]
#         csv_path_train['z_max'][i] = min_max[5]
#         #print(atlas['bounds'])

#     csv_path_train.to_csv(final_path)
    
#     return final_path
### Part Training ###


def adjust_edge_curvatures(source, curvature_name, epsilon=1.0e-08):
    """
    This function adjusts curvatures along the edges of the surface by replacing
     the value with the average value of the curvatures of points in the neighborhood.

    Remember to update the vtkCurvatures object before calling this.

    :param source: A vtkPolyData object corresponding to the vtkCurvatures object.
    :param curvature_name: The name of the curvature, 'Gauss_Curvature' or 'Mean_Curvature'.
    :param epsilon: Absolute curvature values less than this will be set to zero.
    :return:
    """

    def point_neighbourhood(pt_id):
        """
        Find the ids of the neighbours of pt_id.

        :param pt_id: The point id.
        :return: The neighbour ids.
        """
        """
        Extract the topological neighbors for point pId. In two steps:
        1) source.GetPointCells(pt_id, cell_ids)
        2) source.GetCellPoints(cell_id, cell_point_ids) for all cell_id in cell_ids
        """
        cell_ids = vtkIdList()
        source.GetPointCells(pt_id, cell_ids)
        neighbour = set()
        for cell_idx in range(0, cell_ids.GetNumberOfIds()):
            cell_id = cell_ids.GetId(cell_idx)
            cell_point_ids = vtkIdList()
            source.GetCellPoints(cell_id, cell_point_ids)
            for cell_pt_idx in range(0, cell_point_ids.GetNumberOfIds()):
                neighbour.add(cell_point_ids.GetId(cell_pt_idx))
        return neighbour

    def compute_distance(pt_id_a, pt_id_b):
        """
        Compute the distance between two points given their ids.

        :param pt_id_a:
        :param pt_id_b:
        :return:
        """
        pt_a = np.array(source.GetPoint(pt_id_a))
        pt_b = np.array(source.GetPoint(pt_id_b))
        return np.linalg.norm(pt_a - pt_b)

    # Get the active scalars
    source.GetPointData().SetActiveScalars(curvature_name)
    np_source = dsa.WrapDataObject(source)
    curvatures = np_source.PointData[curvature_name]

    #  Get the boundary point IDs.
    array_name = 'ids'
    id_filter = vtkIdFilter()
    id_filter.SetInputData(source)
    id_filter.SetPointIds(True)
    id_filter.SetCellIds(False)
    id_filter.SetPointIdsArrayName(array_name)
    id_filter.SetCellIdsArrayName(array_name)
    id_filter.Update()

    edges = vtkFeatureEdges()
    edges.SetInputConnection(id_filter.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.FeatureEdgesOff()
    edges.Update()

    edge_array = edges.GetOutput().GetPointData().GetArray(array_name)
    boundary_ids = []
    for i in range(edges.GetOutput().GetNumberOfPoints()):
        boundary_ids.append(edge_array.GetValue(i))
    # Remove duplicate Ids.
    p_ids_set = set(boundary_ids)

    # Iterate over the edge points and compute the curvature as the weighted
    # average of the neighbours.
    count_invalid = 0
    for p_id in boundary_ids:
        p_ids_neighbors = point_neighbourhood(p_id)
        # Keep only interior points.
        p_ids_neighbors -= p_ids_set
        # Compute distances and extract curvature values.
        curvs = [curvatures[p_id_n] for p_id_n in p_ids_neighbors]
        dists = [compute_distance(p_id_n, p_id) for p_id_n in p_ids_neighbors]
        curvs = np.array(curvs)
        dists = np.array(dists)
        curvs = curvs[dists > 0]
        dists = dists[dists > 0]
        if len(curvs) > 0:
            weights = 1 / np.array(dists)
            weights /= weights.sum()
            new_curv = np.dot(curvs, weights)
        else:
            # Corner case.
            count_invalid += 1
            # Assuming the curvature of the point is planar.
            new_curv = 0.0
        # Set the new curvature value.
        curvatures[p_id] = new_curv

    #  Set small values to zero.
    if epsilon != 0.0:
        curvatures = np.where(abs(curvatures) < epsilon, 0, curvatures)
        # Curvatures is now an ndarray
        curv = numpy_support.numpy_to_vtk(num_array=curvatures.ravel(),
                                          deep=True,
                                          array_type=VTK_DOUBLE)
        curv.SetName(curvature_name)
        source.GetPointData().RemoveArray(curvature_name)
        source.GetPointData().AddArray(curv)
        source.GetPointData().SetActiveScalars(curvature_name)



### Training call ###

num_classes = 57
nb_epochs = 500
batch_size= 5
dropout_lvl=0.1
radius=1
ico_lvl=1
min_delta_early_stopping = 0.00
patience_early_stopping= 10
num_workers=12
path_data="/CMF/data/timtey/tracts/archives"
path_ico = "/NIRAL/tools/atlas/Surface/Sphere_Template/sphere_f327680_v163842.vtk"
# train_path="/CMF/data/timtey/tracts/tracts_filtered_train_train.csv"
# val_path="/CMF/data/timtey/tracts/tracts_filtered_train_valid.csv"
# test_path="/CMF/data/timtey/tracts/tracts_filtered_test.csv"

#path_train_final, path_valid_final, path_test_final = csv_changes(train_path, val_path, test_path)
#path_train_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_test_label_to_number_copy.csv"
#path_valid_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_test_label_to_number_copy.csv"

# path_train_final_b = "/home/timtey/Documents/Projet/dataset2/tracts_filtered_train_train_label_to_number.csv"
# path_valid_final_b = "/home/timtey/Documents/Projet/dataset2/tracts_filtered_train_valid_label_to_number.csv"
# path_test_final_b = "/home/timtey/Documents/Projet/dataset2/tracts_filtered_train_test_label_to_number.csv"
path_tractography_train = "/home/timtey/Documents/datasets/dataset4/tractography_2_train.csv"
path_tractography_valid = "/home/timtey/Documents/datasets/dataset4/tractography_2_valid.csv"
path_tractography_test = "/home/timtey/Documents/datasets/dataset4/tractography_2_test.csv"
path_train_final = "/home/timtey/Documents/datasets/dataset4/tracts_filtered_train_train_label_to_number_without_missing.csv"
path_valid_final = "/home/timtey/Documents/datasets/dataset4/tracts_filtered_train_valid_label_to_number_without_missing.csv"
path_test_final = "/home/timtey/Documents/datasets/dataset4/tracts_filtered_train_test_label_to_number_nb_cells_without_missing_2_part.csv"

checkpoint_callback = ModelCheckpoint(
    dirpath='/home/timtey/Documents/Models_tensorboard/models/Resnet/Contrastive',
    filename='{epoch}-{val_loss:.2f}',
    monitor='val_loss',
    save_top_k=3
)

tractography_list_vtk = []
tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg.vtp"))
tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_103515_dg.vtp"))
tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_108525_dg.vtp"))
tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_113215_dg.vtp"))
tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_119833_dg.vtp"))
tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_121618_dg.vtp"))
tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_124220_dg.vtp"))
tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_124826_dg.vtp"))
tractography_list_vtk.append(utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_139233_dg.vtp"))
print("tractography_list_vtk", len(tractography_list_vtk))
print("tractography_list_vtk", tractography_list_vtk)
contrastive = True

brain_standart = False
if brain_standart:
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
    # print(bundle_extract_tf_left)
    # print(bundle_extract_tf_left.GetBounds())
    # print(bundle_extract_tf_right.GetBounds())
    bundle_extract_tf_left, mean, f = utils.ScaleSurf(bundle_extract_tf_left, mean_arr=[-0.3,0,0], scale_factor=0.005)
    bundle_extract_tf_right, mean, f = utils.ScaleSurf(bundle_extract_tf_right, mean_arr=[0.3,0,0], scale_factor=0.005)
    # print(bundle_extract_tf_left.GetBounds())
    # print(bundle_extract_tf_right.GetBounds())
    # print(kjdshf)
    normals_left = utils.ComputeNormals(bundle_extract_tf_left)
    normals_right = utils.ComputeNormals(bundle_extract_tf_right)
    cc = vtkCurvatures()
    cc_r = vtkCurvatures()
    cc.SetInputData(normals_left)
    cc_r.SetInputData(normals_right)
    # print(cc)
    cc.SetCurvatureTypeToGaussian()
    # cc.SetCurvatureTypeToMean()
    cc_r.SetCurvatureTypeToGaussian()
    # cc_r.SetCurvatureTypeToMean()
    cc.Update()
    cc_r.Update()
    # print(cc)
    # adjust_edge_curvatures(cc.GetOutput(), 'Gauss_Curvature')
    # adjust_edge_curvatures(cc.GetOutput(), 'Mean_Curvature')
    # adjust_edge_curvatures(cc_r.GetOutput(), 'Gauss_Curvature')
    # adjust_edge_curvatures(cc_r.GetOutput(), 'Mean_Curvature')
    # normals_left.GetPointData().AddArray(cc.GetOutput().GetPointData().GetAbstractArray('Gauss_Curvature'))
    # normals_left.GetPointData().AddArray(cc.GetOutput().GetPointData().GetAbstractArray('Mean_Curvature'))
    # normals_right.GetPointData().AddArray(cc_r.GetOutput().GetPointData().GetAbstractArray('Gauss_Curvature'))
    # normals_right.GetPointData().AddArray(cc_r.GetOutput().GetPointData().GetAbstractArray('Mean_Curvature'))
    # print(normals_left)
    # print(kjhfdskjg)
    verts_left, faces_left, edges_left = utils.PolyDataToTensors(normals_left)
    verts_right, faces_right, edges_right = utils.PolyDataToTensors(normals_right)
    # normals_left = utils.ComputeNormals(bundle_extract_tf_left)
    # print(bundle_extract_tf_left)
    # print(normals_left)
    # print(normals_left.shape)
    # print(bundle_extract_tf_left.GetBounds())
    # print(bundle_extract_tf_right.GetBounds())
    # print(faces1.shape)
    # print(faces2.shape)
    def normalize(verts, bounds):
        verts[:,0] = (0.8*(verts[:,0] - bounds[0])/(bounds[1] - bounds[0])) - 0.4
        verts[:,1] = (0.8*(verts[:,1] - bounds[2])/(bounds[3] - bounds[2])) - 0.4
        verts[:,2] = (0.8*(verts[:,2] - bounds[4])/(bounds[5] - bounds[4])) - 0.4

    # left_bounds = bundle_extract_tf_left.GetBounds()
    # right_bounds = bundle_extract_tf_right.GetBounds()
    # Bounds = [left_bounds[0], right_bounds[1], left_bounds[2], right_bounds[3], left_bounds[4], right_bounds[5]]
    # print(Bounds)
    # normalize(verts_left, Bounds)
    # normalize(verts_right, Bounds)
    mesh_left = Meshes(verts=[verts_left], faces=[faces_left]) # with brain
    mesh_right = Meshes(verts=[verts_right], faces=[faces_right]) # with brain
    # mesh_left = mesh_left.to(self.device) # with brain
    # mesh_right = mesh_right.to(self.device) # with brain
    # self.mesh_left.textures = textures # with brain
    # self.mesh_right.textures = textures # with brain
    meshes_brain = join_meshes_as_scene([mesh_left, mesh_right]) # with brain
    # meshes_brain = mesh_right
    verts_brain = meshes_brain.verts_padded()
    faces_brain = meshes_brain.faces_padded()
    verts_brain = verts_brain[0]
    faces_brain = faces_brain[0]
    # print(normals_left)
    normals_l = torch.tensor(vtk_to_numpy(normals_left.GetPointData().GetScalars("Normals")))
    normals_r = torch.tensor(vtk_to_numpy(normals_right.GetPointData().GetScalars("Normals")))
    # normals_l = torch.tensor(vtk_to_numpy(normals_left.GetPointData().GetScalars("Gauss_Curvature"))).unsqueeze(1) #Gauss_Curvature,
    # normals_l = torch.tensor(vtk_to_numpy(normals_left.GetPointData().GetScalars("Mean_Curvature"))).unsqueeze(1) #Gauss_Curvature, 
    # normals_r = torch.tensor(vtk_to_numpy(normals_right.GetPointData().GetScalars("Gauss_Curvature"))).unsqueeze(1) #Normals
    # normals_r = torch.tensor(vtk_to_numpy(normals_right.GetPointData().GetScalars("Mean_Curvature"))).unsqueeze(1) #Normals
    # normals_l = torch.cat((normals_l, normals_l, normals_l),1) #curvature
    min_l = torch.min(normals_l) #curvature
    max_l = torch.max(normals_l) #curvature
    # normals_r = torch.cat((normals_r, normals_r, normals_r),1) #curvature
    min_r = torch.min(normals_r) #curvature
    max_r = torch.max(normals_r) #curvature


    # def transf(normals, max, min):                          #curvature
    #     normals = ((2*(normals - min))/(max - min)) - 1 #curvature
    #     return normals #curvature
    # normals_l = transf(normals_l, max_l, min_l) #curvature
    # normals_r = transf(normals_r, max_r, min_r) #curvature
    # print(normals_l.shape)
    # print(torch.sum(normals_l==1))
    # print(torch.max(normals_l))
    # print(torch.min(normals_r))
    # print(torch.max(normals_r))
    # print(kusgdku)
    normals_brain = torch.cat((normals_l, normals_r), 0)
    vertex_features_brain = normals_brain #torch.cat((normals_brain), 1)
    # print(vertex_features_brain.shape)
    faces_pid0_brain = faces_brain[:,0:1]
    nb_faces_brain = faces_brain.shape[0]
    offset_brain = torch.zeros((nb_faces_brain,vertex_features_brain.shape[1]), dtype=int) + torch.arange(vertex_features_brain.shape[1]).to(torch.int64)
    faces_pid0_offset_brain = offset_brain + torch.multiply(faces_pid0_brain, vertex_features_brain.shape[1])
    face_features_brain = torch.take(vertex_features_brain, faces_pid0_offset_brain)
    complete = torch.ones((faces_brain.shape[0],6))
    # print(complete.shape)
    face_features_brain = torch.cat((face_features_brain, complete), 1)
else:
    verts_brain = []
    faces_brain = []
    face_features_brain = []
    verts_left = []
    faces_left = []
    verts_right = []
    faces_right = []
# print(face_features_brain.shape)
# print(normals_l.shape)
# print(normals_r.shape)
# print(normals_brain.shape)
# print(verts_brain.shape)
# print(faces_brain.shape)
# self.meshes_brain = self.meshes_brain.to(self.device)
# print(verts1)
# print(verts2)
# mesh_left = Meshes(verts=[verts_left], faces=[faces_left])
# mesh_right = Meshes(verts=[verts_right], faces=[faces_right])


# print(bundle_extract_tf_left)
# print(scale)
# print(ksdjhg)

df = pd.read_csv(path_test_final)
# for i in range(len(df)):
logger = TensorBoardLogger(save_dir="/home/timtey/Documents/Models_tensorboard/tensorboard_photos", name='Resnet')
image_logger = BrainNetImageLogger_contrastive_tractography_labeled(num_features = 3,num_images = 24,mean = 0)

early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=min_delta_early_stopping, patience=patience_early_stopping, verbose=True, mode='min')
# trainer=Trainer(max_epochs=nb_epochs, callbacks=[early_stop_callback, checkpoint_callback], accelerator="gpu")
trainer=Trainer(log_every_n_steps=5, max_epochs=nb_epochs, logger = logger, callbacks=[early_stop_callback, checkpoint_callback, image_logger], accelerator="gpu")
Y_TRUE = []
Y_PRED = []
Acc = []
Acc_details = []



brain_data=Bundles_DataModule_tractography_labeled_fibers(contrastive, 0,0,0,0,0,path_data, path_ico, batch_size, path_train_final, path_valid_final, path_test_final, verts_brain, faces_brain, face_features_brain, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)

weights = brain_data.get_weights()
model= Fly_by_CNN_contrastive_tractography_labeled(contrastive, radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, verts_left, faces_left, verts_right, faces_right, learning_rate=0.001)
trainer.fit(model, brain_data)
# trainer.test(model, brain_data)
for index_csv in range(len(df)):
    path = f"/CMF/data/timtey/tracts/{df['surf'][index_csv]}"
    # print(path)
    bundle = utils.ReadSurf(path)
    # print(bundle)
    L = []
    for i in range(bundle.GetNumberOfCells()):
        fiber = utils.ExtractFiber(bundle,i)
        cc1_tf=vtk.vtkTriangleFilter()
        cc1_tf.SetInputData(fiber)
        cc1_tf.Update()
        fiber_tf = cc1_tf.GetOutput()
        L.append(fiber_tf)
    fibers = bundle.GetNumberOfCells()
    # print(fibers)
    fibers = min(df['num_cells'])
    # fibers_s = 500
    # if fibers < fibers_s:
        # print(fibers)
    # else:
        # fibers = fibers_s
    # print(fibers)

    brain_data=Bundles_DataModule_tractography_labeled_fibers(contrastive, bundle, L, fibers, 0, index_csv, path_data, path_ico, batch_size, path_train_final, path_valid_final, path_test_final, verts_brain, faces_brain, face_features_brain, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=num_workers)

    

    trainer.test(model, brain_data)
    
    label_true = model.get_y_true()
    label_pred = model.get_y_pred()
    # print("label_true", label_true)
    # print("label_pred", label_pred)
    # print("label_true len", len(label_true))
    # print("label_pred len", len(label_pred))
    Y_TRUE.append(label_true)
    Y_PRED.append(label_pred)
    Acc.append(model.get_accuracy())
    Acc_details.append([label_true[0],model.get_accuracy()])
    # print(label_pred)
    tensor_pred = torch.tensor(label_pred)
    print(tensor_pred.unique(return_counts=True))
    print(Acc_details)
    print(len(Acc_details))
    Acc_tot_iteration = np.mean(Acc)
    print(Acc_tot_iteration)

Y_TRUE = [item for sublist in Y_TRUE for item in sublist]
# print("Y_TRUE", Y_TRUE)
print(len(Y_TRUE))
Y_PRED = [item for sublist in Y_PRED for item in sublist]
# print("Y_PRED", Y_PRED)
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


