from torch.utils.data import Dataset, DataLoader#, ConcatDataset
import torch
#import split_train_eval
from tools import utils
import pytorch_lightning as pl 
import vtk
from pytorch3d.structures import Meshes
from random import *
from pytorch3d.vis.plotly_vis import plot_scene
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from pytorch3d.renderer import TexturesVertex
from torch.utils.data._utils.collate import default_collate
import pandas as pd
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from itertools import cycle
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
from sklearn.utils.class_weight import compute_class_weight


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)



class Bundles_Dataset_contrastive_tractography_labeled(Dataset):
    def __init__(self, data, path_data, path_ico, verts_brain, faces_brain, face_features_brain, transform=True, column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
        self.data = data
        self.transform = transform
        self.path_data = path_data
        self.path_ico = path_ico
        self.verts_brain = verts_brain
        self.faces_brain = faces_brain
        self.face_features_brain = face_features_brain
        self.column_class = column_class
        self.column_id = column_id
        self.column_label = column_label
        self.column_x_min = column_x_min
        self.column_x_max = column_x_max
        self.column_y_min = column_y_min
        self.column_y_max = column_y_max
        self.column_z_min = column_z_min
        self.column_z_max = column_z_max
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     
    def __len__(self):
        return len(self.data)
        # return 10

    def __getitem__(self, idx):
        # print("data", self.data)
        # print("path_data", self.path_data)
        # print("idx", idx)
        sample_row = self.data.loc[idx]
        
        sample_id, sample_class, sample_label = sample_row[self.column_id], sample_row[self.column_class], sample_row[self.column_label]
        sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max = sample_row[self.column_x_min], sample_row[self.column_x_max], sample_row[self.column_y_min], sample_row[self.column_y_max], sample_row[self.column_z_min], sample_row[self.column_z_max]
        path_cc1 = f"/CMF/data/timtey/tracts/archives/{sample_id}_tracts/{sample_class}.vtp"
        
        cc1 = utils.ReadSurf(path_cc1)
        n = randint(0,cc1.GetNumberOfCells()-1)
        name = [sample_id, sample_label, n]
        cc1_extract = utils.ExtractFiber(cc1,n)
        cc1_tf=vtk.vtkTriangleFilter()
        cc1_tf.SetInputData(cc1_extract)
        cc1_tf.Update()
        cc1_extract_tf = cc1_tf.GetOutput()
        
        verts, faces, edges = utils.PolyDataToTensors(cc1_extract_tf)
    
        verts_fiber = torch.clone(verts)
        faces_fiber = torch.clone(faces)
        edges_fiber = torch.clone(edges)

        verts_fiber_bounds = cc1_extract_tf.GetBounds()
        verts_fiber_bounds = list(verts_fiber_bounds)
        max_bounds = max(verts_fiber_bounds)
        min_bounds = min(verts_fiber_bounds)
        verts_fiber_bounds = [min_bounds,max_bounds,min_bounds,max_bounds,min_bounds,max_bounds]
        sample_min_max = [sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max]

        # EstimatedUncertainty = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("EstimatedUncertainty"))).unsqueeze(1)
        # FA1 = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("FA1"))).unsqueeze(1)
        # FA2 = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("FA2"))).unsqueeze(1)
        # HemisphereLocataion = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("HemisphereLocataion"))).unsqueeze(1)
        # cluster_idx = vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("cluster_idx"))
        # trace1 = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("trace1"))).unsqueeze(1)
        # trace2 = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("trace2"))).unsqueeze(1)
        # vtkOriginalPointIds = vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("vtkOriginalPointIds"))
        TubeNormals = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("TubeNormals")))


        # vertex_features = torch.cat([EstimatedUncertainty, FA1, FA2, HemisphereLocataion, trace1, trace2, TubeNormals], dim=1)
        vertex_features = torch.cat([TubeNormals], dim=1)
        faces_pid0 = faces[:,0:1]
        faces_pid0_fiber = faces_fiber[:,0:1]
        nb_faces = len(faces)
        nb_faces_fiber = len(faces_fiber)
        # offset = torch.zeros((nb_faces,vertex_features.shape[1]), dtype=int) + torch.Tensor([i for i in range(vertex_features.shape[1])]).to(torch.int64)
        offset = torch.zeros((nb_faces, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)
        offset_fiber = torch.zeros((nb_faces_fiber, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])
        faces_pid0_offset_fiber = offset_fiber + torch.multiply(faces_pid0_fiber, vertex_features.shape[1])
        face_features = torch.take(vertex_features, faces_pid0_offset)
        face_features_fiber = torch.take(vertex_features, faces_pid0_offset_fiber)
        

        ### labels ###
        labels = torch.tensor([sample_label])
        labels_fiber = torch.tensor([sample_label])
        # #Load  Icosahedron
        # reader = utils.ReadSurf(self.path_ico)
        # verts_ico, faces_ico, edges_ico = utils.PolyDataToTensors(reader)
        # nb_faces = len(faces_ico)

        # FF_brain = torch.ones(self.faces_brain.shape[0],8)

        # list_id = [120515, 102816, 111413, 134324, 136227, 137633, 142828, 143325]
        # if sample_id in list_id:
        #     return verts,faces,face_features,labels,verts_fiber,faces_fiber,face_features_fiber,labels_fiber, self.verts_brain, self.faces_brain, self.face_features_brain
        # else:
        ###
        # path_brain = f"/CMF/data/timtey/tractography/all/brain_mask_{sample_id}.vtk"
        # brain_mask = utils.ReadSurf(path_brain)
        # brain_tf = vtk.vtkTriangleFilter()
        # brain_tf.SetInputData(brain_mask)
        # brain_tf.Update()
        # brain_mask_f = brain_tf.GetOutput()
        # brain_mask_f, mean, std = utils.ScaleSurf(brain_mask_f, scale_factor=0.005)
        # normal_brain = utils.ComputeNormals(brain_mask_f)
        # verts_brain, faces_brain, edges_brain = utils.PolyDataToTensors(brain_mask_f)
        # normals_brain = torch.tensor(vtk_to_numpy(normal_brain.GetPointData().GetScalars("Normals")))
        # vertex_features_brain = torch.cat([normals_brain], dim=1)
        # faces_pid0_brain = faces_brain[:,0:1]
        # nb_faces_brain = faces_brain.shape[0]
        # offset_brain = torch.zeros((nb_faces_brain,vertex_features_brain.shape[1]), dtype=int) + torch.arange(vertex_features_brain.shape[1]).to(torch.int64)
        # faces_pid0_offset_brain = offset_brain + torch.multiply(faces_pid0_brain, vertex_features_brain.shape[1])
        # face_features_brain = torch.take(vertex_features_brain, faces_pid0_offset_brain)
        # complete = torch.ones((faces_brain.shape[0],6))
        # face_features_brain = torch.cat((face_features_brain, complete), 1)
        ###

        verts_brain = torch.load(f"brain_structures/verts_brain_{sample_id}.pt")
        faces_brain = torch.load(f"brain_structures/faces_brain_{sample_id}.pt")
        face_features_brain = torch.load(f"brain_structures/face_features_brain_{sample_id}.pt")
        data_lab = torch.tensor([0])
        name_l = torch.tensor([name])

        return verts,faces,face_features,labels,verts_fiber,faces_fiber,face_features_fiber,labels_fiber, verts_brain, faces_brain, face_features_brain, verts_fiber_bounds, sample_min_max, data_lab, name_l
        # return verts,faces,face_features,labels,verts_fiber,faces_fiber,face_features_fiber,labels_fiber, self.verts_brain, self.faces_brain, self.face_features_brain
        # return verts, faces, face_features, labels, verts_fiber     #24 images
        # return verts, faces, face_features, labels    #12images



class Bundles_Dataset_tractography(Dataset):
    def __init__(self, data, path_data, path_ico, verts_brain, faces_brain, face_features_brain, length, transform=True, column_surf='surf',column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
        self.data = data
        self.transform = transform
        self.path_data = path_data
        self.path_ico = path_ico
        self.verts_brain = verts_brain
        self.faces_brain = faces_brain
        self.face_features_brain = face_features_brain
        self.length = length
        self.column_surf = column_surf
        self.column_class = column_class
        self.column_id = column_id
        self.column_label = column_label
        self.column_x_min = column_x_min
        self.column_x_max = column_x_max
        self.column_y_min = column_y_min
        self.column_y_max = column_y_max
        self.column_z_min = column_z_min
        self.column_z_max = column_z_max

    
    def __len__(self):
        # print("len", self.fibers)
        # return len(self.data)
        # print("self.length", self.length)
        return self.length
        # return 10

    def __getitem__(self, idx):
        sample_row = self.data.loc[idx]
        sample_surf = sample_row[self.column_surf]
        sample_id, sample_class, sample_label = sample_row[self.column_id], sample_row[self.column_class], sample_row[self.column_label]
        sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max = sample_row[self.column_x_min], sample_row[self.column_x_max], sample_row[self.column_y_min], sample_row[self.column_y_max], sample_row[self.column_z_min], sample_row[self.column_z_max]
        
        path_tracts = f"{sample_surf}"
        tracts = utils.ReadSurf(path_tracts)
        n = randint(0,tracts.GetNumberOfCells()-1)

        tracts_extract = utils.ExtractFiber(tracts,n)
        name = [sample_id, sample_label, n]
        tracts_tf = vtk.vtkTriangleFilter()
        tracts_tf.SetInputData(tracts_extract)
        tracts_tf.Update()
        tracts_f = tracts_tf.GetOutput()
        verts, faces, edges = utils.PolyDataToTensors(tracts_f)

        """
        # verts = torch.tensor([])
        while verts.shape[0] < 100:
            tracts_extract = utils.ExtractFiber(tracts,n)
            name = [sample_id, sample_label, n]
            tracts_tf = vtk.vtkTriangleFilter()
            tracts_tf.SetInputData(tracts_extract)
            tracts_tf.Update()
            tracts_f = tracts_tf.GetOutput()
            verts, faces, edges = utils.PolyDataToTensors(tracts_f)
        """

        verts_fiber = torch.clone(verts)
        faces_fiber = torch.clone(faces)
        edges_fiber = torch.clone(edges)

        verts_fiber_bounds = tracts_f.GetBounds()
        verts_fiber_bounds = list(verts_fiber_bounds)
        max_bounds = max(verts_fiber_bounds)
        min_bounds = min(verts_fiber_bounds)
        verts_fiber_bounds = [min_bounds,max_bounds,min_bounds,max_bounds,min_bounds,max_bounds]
        sample_min_max = [sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max]

        TubeNormals = torch.tensor(vtk_to_numpy(tracts_f.GetPointData().GetScalars("TubeNormals")))
        vertex_features = torch.cat([TubeNormals], dim=1)
        faces_pid0 = faces[:,0:1]
        faces_pid0_fiber = faces_fiber[:,0:1]
        nb_faces = len(faces)
        nb_faces_fiber = len(faces_fiber)

        offset = torch.zeros((nb_faces, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)
        offset_fiber = torch.zeros((nb_faces_fiber, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)

        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])
        faces_pid0_offset_fiber = offset_fiber + torch.multiply(faces_pid0_fiber, vertex_features.shape[1])
        face_features = torch.take(vertex_features, faces_pid0_offset)
        face_features_fiber = torch.take(vertex_features, faces_pid0_offset_fiber)

        verts_brain = torch.load(f"brain_structures/verts_brain_{sample_id}.pt")
        faces_brain = torch.load(f"brain_structures/faces_brain_{sample_id}.pt")
        face_features_brain = torch.load(f"brain_structures/face_features_brain_{sample_id}.pt")

        labels = torch.tensor([sample_label])
        labels_fiber = torch.tensor([sample_label])
        data_lab = torch.tensor([1])
        name_l = torch.tensor([name])

        return verts,faces,face_features,labels,verts_fiber,faces_fiber,face_features_fiber,labels_fiber, verts_brain, faces_brain, face_features_brain, verts_fiber_bounds, sample_min_max, data_lab, name_l

        






class Bundles_Dataset_test_contrastive_tractography_labeled(Dataset):
    def __init__(self, contrastive, data, bundle, L, fibers, index_csv, path_data, path_ico, verts_brain, faces_brain, face_features_brain, transform=True, column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
        self.contrastive = contrastive
        self.data = data
        self.bundle = bundle
        self.L = L
        self.fibers = fibers
        self.index_csv = index_csv
        self.transform = transform
        self.path_data = path_data
        self.path_ico = path_ico
        self.verts_brain = verts_brain
        self.faces_brain = faces_brain
        self.face_features_brain = face_features_brain
        self.column_class = column_class
        self.column_id = column_id
        self.column_label = column_label
        self.column_x_min = column_x_min
        self.column_x_max = column_x_max
        self.column_y_min = column_y_min
        self.column_y_max = column_y_max
        self.column_z_min = column_z_min
        self.column_z_max = column_z_max

     
    def __len__(self):
        print("len", self.fibers)
        return self.fibers

    def __getitem__(self, idx):

        sample_row = self.data.loc[self.index_csv]
        sample_label = sample_row[self.column_label]
        sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max = sample_row[self.column_x_min], sample_row[self.column_x_max], sample_row[self.column_y_min], sample_row[self.column_y_max], sample_row[self.column_z_min], sample_row[self.column_z_max]
        sample_id = sample_row[self.column_id]


        ###
        bundle_extract_tf = self.L[idx]
        name = [sample_id, sample_label, idx]
        ###
        # bundle_extract = utils.ExtractFiber(self.bundle,idx)
        # bundle_tf=vtk.vtkTriangleFilter()
        # bundle_tf.SetInputData(bundle_extract)
        # bundle_tf.Update()
        # bundle_extract_tf = bundle_tf.GetOutput()
        ###
        verts, faces, edges = utils.PolyDataToTensors(bundle_extract_tf)        
        verts_fiber = torch.clone(verts)
        faces_fiber = torch.clone(faces)
        edges_fiber = torch.clone(edges)

        verts_fiber_bounds = bundle_extract_tf.GetBounds()
        verts_fiber_bounds = list(verts_fiber_bounds)
        max_bounds = max(verts_fiber_bounds)
        min_bounds = min(verts_fiber_bounds)
        verts_fiber_bounds = [min_bounds,max_bounds,min_bounds,max_bounds,min_bounds,max_bounds]
        sample_min_max = [sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max]


        # ldjs = bundle_extract_tf.GetPointData().GetScalars("colors")
        # EstimatedUncertainty = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("EstimatedUncertainty"))).unsqueeze(1)
        # FA1 = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("FA1"))).unsqueeze(1)
        # FA2 = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("FA2"))).unsqueeze(1)
        # HemisphereLocataion = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("HemisphereLocataion"))).unsqueeze(1)
        # cluster_idx = vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("cluster_idx"))
        # trace1 = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("trace1"))).unsqueeze(1)
        # trace2 = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("trace2"))).unsqueeze(1)
        # vtkOriginalPointIds = vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("vtkOriginalPointIds"))
        TubeNormals = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("TubeNormals")))
        # vertex_features = torch.cat([EstimatedUncertainty, FA1, FA2, HemisphereLocataion, trace1, trace2, TubeNormals], dim=1)
        vertex_features = torch.cat([TubeNormals], dim=1)

        faces_pid0 = faces[:,0:1]
        faces_pid0_fiber = faces_fiber[:,0:1]
        nb_faces = len(faces)
        nb_faces_fiber = len(faces_fiber)
        # offset = torch.zeros((nb_faces,vertex_features.shape[1]), dtype=int) + torch.Tensor([i for i in range(vertex_features.shape[1])]).to(torch.int64)
        offset = torch.zeros((nb_faces, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)
        offset_fiber = torch.zeros((nb_faces_fiber, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])
        faces_pid0_offset_fiber = offset_fiber + torch.multiply(faces_pid0_fiber, vertex_features.shape[1])
        face_features = torch.take(vertex_features, faces_pid0_offset)
        face_features_fiber = torch.take(vertex_features, faces_pid0_offset_fiber)

        ### labels ###
        labels = torch.tensor([sample_label])
        labels_fiber = torch.tensor([sample_label])
        # if self.contrastive:
            # labels = torch.tensor([1])
        #Load  Icosahedron
        # reader = utils.ReadSurf(self.path_ico)
        # verts_ico, faces_ico, edges_ico = utils.PolyDataToTensors(reader)
        # nb_faces = len(faces_ico)
        # print(verts_fiber)
        # print("FF_brain")
        # FF_brain = torch.ones(self.faces_brain.shape[0],8)
        # print("FF_brain", FF_brain.shape)
        # print(self.face_features_brain.shape)
        # list_id = [120515, 102816, 111413, 134324, 136227, 137633, 142828, 143325]
        # if sample_id in list_id:
        #     return verts,faces,face_features,labels,verts_fiber,faces_fiber,face_features_fiber,labels_fiber, self.verts_brain, self.faces_brain, self.face_features_brain
        # else:
        ###
        # path_brain = f"/CMF/data/timtey/tractography/all/brain_mask_{sample_id}.vtk"
        # # print("path_brain", path_brain)
        # brain_mask = utils.ReadSurf(path_brain)
        # # print("brain_mask", brain_mask)
        # brain_tf = vtk.vtkTriangleFilter()
        # brain_tf.SetInputData(brain_mask)
        # brain_tf.Update()
        # brain_mask_f = brain_tf.GetOutput()
        # # print("brain_mask_f", brain_mask_f)
        # brain_mask_f, mean, std = utils.ScaleSurf(brain_mask_f, scale_factor=0.005)
        # normal_brain = utils.ComputeNormals(brain_mask_f)
        # # print("brain_mask_f", normal_brain)
        # # print("brain_mask_f", brain_mask_f)
        # # print("sample_id", sample_id)
        # verts_brain, faces_brain, edges_brain = utils.PolyDataToTensors(brain_mask_f)
        # normals_brain = torch.tensor(vtk_to_numpy(normal_brain.GetPointData().GetScalars("Normals")))
        # vertex_features_brain = torch.cat([normals_brain], dim=1)
        # # print("vertex_features_brain", vertex_features_brain.shape)
        # faces_pid0_brain = faces_brain[:,0:1]
        # nb_faces_brain = faces_brain.shape[0]
        # offset_brain = torch.zeros((nb_faces_brain,vertex_features_brain.shape[1]), dtype=int) + torch.arange(vertex_features_brain.shape[1]).to(torch.int64)
        # faces_pid0_offset_brain = offset_brain + torch.multiply(faces_pid0_brain, vertex_features_brain.shape[1])
        # face_features_brain = torch.take(vertex_features_brain, faces_pid0_offset_brain)
        # complete = torch.ones((faces_brain.shape[0],6))
        # face_features_brain = torch.cat((face_features_brain, complete), 1)
        ###

        verts_brain = torch.load(f"brain_structures/verts_brain_{sample_id}.pt")
        faces_brain = torch.load(f"brain_structures/faces_brain_{sample_id}.pt")
        face_features_brain = torch.load(f"brain_structures/face_features_brain_{sample_id}.pt")

        data_lab = torch.tensor([2])
        name_l = torch.tensor([name])
        return verts,faces,face_features,labels,verts_fiber,faces_fiber,face_features_fiber,labels_fiber, verts_brain, faces_brain, face_features_brain, verts_fiber_bounds, sample_min_max, data_lab, name_l

        # return verts, faces, face_features, labels, verts_fiber, faces_fiber, face_features_fiber, labels_fiber, self.verts_brain, self.faces_brain, self.face_features_brain   #24 images
        # return verts, faces, face_features, labels, verts_fiber   #24 images
        # return verts, faces, face_features, labels    #12images



class Bundles_DataModule_tractography_labeled_fibers(pl.LightningDataModule):
    def __init__(self, contrastive, bundle, L, fibers, fibers_valid, index_csv, path_data, path_ico, batch_size, train_path, val_path, test_path, verts_brain, faces_brain, face_features_brain, path_tractography_train, path_tractography_valid, path_tractography_test, num_workers=12, transform=True, persistent_workers=False):
        super().__init__()
        self.contrastive = contrastive
        self.bundle = bundle
        self.L = L
        self.fibers = fibers
        self.fibers_valid = fibers_valid
        self.index_csv = index_csv
        self.path_data = path_data
        self.path_ico = path_ico
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.verts_brain = verts_brain
        self.faces_brain = faces_brain
        self.face_features_brain = face_features_brain
        self.path_tractography_train = path_tractography_train
        self.path_tractography_valid = path_tractography_valid
        self.path_tractography_test = path_tractography_test
        self.num_workers = num_workers
        self.transform = transform
        self.persistent_workers = persistent_workers
        self.weights = []
        self.df_train = pd.read_csv(self.train_path)
        self.df_val = pd.read_csv(self.val_path)
        self.df_test = pd.read_csv(self.test_path)
        y_train = np.array(self.df_train.loc[:,'label'])

        labels = np.unique(y_train)
        weights_train = torch.tensor(compute_class_weight('balanced', classes = labels, y = y_train)).to(torch.float32)
        self.weights.append(weights_train)

        y_val = np.array(self.df_val.loc[:,'label'])

        labels = np.unique(y_val)

        weights_val = torch.tensor(compute_class_weight('balanced', classes = labels, y = y_val)).to(torch.float32)
        self.weights.append(weights_val)


        y_test2 = []
        for i in range(len(self.df_test)):
            nb_cells = self.df_test.loc[i,'num_cells']
            for j in range(nb_cells):
                y_test2.append(self.df_test.loc[i,'label'])

        y_test = np.array(self.df_test.loc[:,'label'])

        labels = np.unique(y_test)
        weights_test = torch.tensor(compute_class_weight('balanced', classes = labels, y = y_test2)).to(torch.float32)
        self.weights.append(weights_test)


    def setup(self, stage=None):

        #list_train_data = open(self.train_path, "r").read().splitlines()
        #list_val_data = open(self.val_path, "r").read().splitlines()
        #list_test_data = open(self.test_path, "r").read().splitlines()
        list_train_data = pd.read_csv(self.train_path)
        list_val_data = pd.read_csv(self.val_path)
        list_test_data = pd.read_csv(self.test_path)
        list_train_tractography_data = pd.read_csv(self.path_tractography_train)
        list_val_tractography_data = pd.read_csv(self.path_tractography_valid)
        list_test_tractography_data = pd.read_csv(self.path_tractography_test)
        len_train = len(list_train_data)
        len_val = len(list_val_data)
        len_test = len(list_test_data)
        
        
        self.train_dataset = Bundles_Dataset_contrastive_tractography_labeled(list_train_data, self.path_data, self.path_ico, self.verts_brain, self.faces_brain, self.face_features_brain, self.transform)
        self.val_dataset = Bundles_Dataset_contrastive_tractography_labeled(list_val_data, self.path_data, self.path_ico, self.verts_brain, self.faces_brain, self.face_features_brain, self.transform)
        self.test_dataset = Bundles_Dataset_test_contrastive_tractography_labeled(self.contrastive, list_test_data, self.bundle, self.L, self.fibers, self.index_csv, self.path_data, self.path_ico, self.verts_brain, self.faces_brain, self.face_features_brain, self.transform)
        self.train_tractography_dataset = Bundles_Dataset_tractography(list_train_tractography_data, self.path_data, self.path_ico, self.verts_brain, self.faces_brain, self.face_features_brain, len_train, self.transform)
        self.val_tractography_dataset = Bundles_Dataset_tractography(list_val_tractography_data, self.path_data, self.path_ico, self.verts_brain, self.faces_brain, self.face_features_brain, len_val, self.transform)
        self.test_tractography_dataset = Bundles_Dataset_tractography(list_test_tractography_data, self.path_data, self.path_ico, self.verts_brain, self.faces_brain, self.face_features_brain, len_test, self.transform)

        self.concatenated_train_dataset = ConcatDataset(self.train_dataset, self.train_tractography_dataset)
        self.concatenated_val_dataset = ConcatDataset(self.val_dataset, self.val_tractography_dataset)
        self.concatenated_test_dataset = ConcatDataset(self.test_dataset, self.test_tractography_dataset)

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        return DataLoader(self.concatenated_train_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        return DataLoader(self.concatenated_val_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        # return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        return DataLoader(self.concatenated_test_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def get_weights(self):
        return self.weights

    def pad_verts_faces(self, batch):
        print("len batch", len(batch))
        print("len batch[0]", len(batch[0]))
        print("batch[0]", len(batch[0][0]))
        labeled_fibers = ()
        tractography_fibers = ()
        for i in range(len(batch)):
            labeled_fibers += (batch[i][0],)
            tractography_fibers += (batch[i][1],)

        print("len labeled_fibers", len(labeled_fibers))
        print("len labeled_fibers", len(labeled_fibers[0]))
        print("len tractography_fibers", len(tractography_fibers))
        print("len tractography_fibers", len(tractography_fibers[0]))
        
        print("len batch i ", len(batch[i]))
        print("len labeled_fibers", len(labeled_fibers))
        print("len tractography_fibers", len(tractography_fibers))

        verts_lf = [v for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        faces_lf = [f for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        verts_data_faces_lf = [vdf for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        labels_lf = [l for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        verts_fiber_lf = [vfi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        faces_fiber_lf = [ffi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        verts_data_faces_fiber_lf = [vdffi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        labels_fiber_lf = [lfi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        verts_brain_lf = [vb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        faces_brain_lf = [fb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        verts_data_faces_brain_lf = [ffb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        verts_fiber_bounds_lf = [vfb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in labeled_fibers]
        sample_min_max_lf = [smm for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in tractography_fibers]
        data_lab_lf = [dl for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm,dl, nl in labeled_fibers]
        name_labels_lf = [nl for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm,dl, nl in labeled_fibers]



        verts_tf = [v for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in tractography_fibers]
        faces_tf = [f for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in tractography_fibers]
        verts_data_faces_tf = [vdf for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm,dl, nl in tractography_fibers]
        labels_tf = [l for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm, dl, nl in tractography_fibers]
        verts_fiber_tf = [vfi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm, dl, nl in tractography_fibers]
        faces_fiber_tf = [ffi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm, dl, nl in tractography_fibers]
        verts_data_faces_fiber_tf = [vdffi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm, dl, nl in tractography_fibers]
        labels_fiber_tf = [lfi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm, dl, nl in tractography_fibers]
        verts_brain_tf = [vb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm, dl, nl in tractography_fibers]
        faces_brain_tf = [fb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm, dl, nl in tractography_fibers]
        verts_data_faces_brain_tf = [ffb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm, dl, nl in tractography_fibers]
        verts_fiber_bounds_tf = [vfb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm, dl, nl in tractography_fibers]
        sample_min_max_tf = [smm for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm, dl, nl in tractography_fibers]
        data_lab_tf = [dl for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm,dl, nl in tractography_fibers]
        name_labels_tf = [nl for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb,smm,dl, nl in tractography_fibers]

        print("verts", len(verts_lf))
        print("verts", len(verts_tf))
        print("faces", len(faces_lf))
        print("faces", len(faces_tf))
        print("verts_data_faces", len(verts_data_faces_lf))
        print("verts_data_faces", len(verts_data_faces_tf))
        print("labels", len(labels_lf))
        print("labels", len(labels_tf))
        print("verts_fiber", len(verts_fiber_lf))
        print("verts_fiber", len(verts_fiber_tf))
        print("faces_fiber", len(faces_fiber_lf))
        print("faces_fiber", len(faces_fiber_tf))
        print("verts_data_faces_fiber", len(verts_data_faces_fiber_lf))
        print("verts_data_faces_fiber", len(verts_data_faces_fiber_tf))
        print("labels_fiber", len(labels_fiber_lf))
        print("labels_fiber", len(labels_fiber_tf))
        print("verts_brain", len(verts_brain_lf))
        print("verts_brain", len(verts_brain_tf))
        print("faces_brain", len(faces_brain_lf))
        print("faces_brain", len(faces_brain_tf))
        print("verts_data_faces_brain", len(verts_data_faces_brain_lf))
        print("verts_data_faces_brain", len(verts_data_faces_brain_tf))
        print("verts_fiber_bounds", len(verts_fiber_bounds_lf))
        print("verts_fiber_bounds", len(verts_fiber_bounds_tf))
        print("sample_min_max", len(sample_min_max_lf))
        print("sample_min_max", len(sample_min_max_tf))
        print("data_lab", len(data_lab_lf))
        print("data_lab", len(data_lab_tf))
        print("name_labels", len(name_labels_lf))
        print("name_labels", len(name_labels_tf))


        # print(akjhdfkdsh)
        # verts = [v for v, f, vdf, l, vfi in labeled_fibers]
        # labeled_fibers = [lf for lf, tf in labeled_fibers]
        # verts = [v for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm,dl, nl in labeled_fibers]
        # verts = [v for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm,dl, nl in tractography_fibers]

        # print("verts deb", len(verts))
        # faces = [f for v, f, vdf, l, vfi in batch]
        # faces = [f for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm,dl, nl in batch]        
        # verts_data_vertex = [vdv for v, f, vdv, vdf, l in batch]        
        # verts_data_faces = [vdf for v, f, vdf, l, vfi in batch]
        # verts_data_faces = [vdf for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm,dl, nl in batch]        
        # labels = [l for v, f, vdf, l, vfi in batch]  
        # labels = [l for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm,dl, nl in batch]      
        # verts_fiber = [vfi for v, f, vdf, l, vfi in batch]
        # verts_fiber = [vfi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm,dl, nl in batch]
        # faces_fiber = [ffi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm,dl, nl in batch]
        # verts_data_faces_fiber = [vdffi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in batch]
        # labels_fiber = [lfi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in batch]
        # verts_brain = [vb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in batch]
        # faces_brain = [fb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in batch]
        # verts_data_faces_brain = [ffb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in batch]
        # verts_fiber_bounds = [vfb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in batch]
        # sample_min_max = [smm for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in batch]
        # data_lab = [dl for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in batch]
        # name_labels = [nl for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm, dl, nl in batch]




        verts_lf = pad_sequence(verts_lf, batch_first=True, padding_value=0.0)
        faces_lf = pad_sequence(faces_lf, batch_first=True, padding_value=-1)
        # verts_data_faces_lf = torch.cat(verts_data_vertex_lf)
        verts_data_faces_lf = torch.cat(verts_data_faces_lf)
        labels_lf = torch.cat(labels_lf)
        verts_fiber_lf = pad_sequence(verts_fiber_lf, batch_first=True, padding_value=0.0)
        faces_fiber_lf = pad_sequence(faces_fiber_lf, batch_first=True, padding_value=-1)
        verts_data_faces_fiber_lf = torch.cat(verts_data_faces_fiber_lf)
        labels_fiber_lf = torch.cat(labels_fiber_lf)
        verts_brain_lf = pad_sequence(verts_brain_lf, batch_first=True, padding_value=0.0)
        faces_brain_lf = pad_sequence(faces_brain_lf, batch_first=True, padding_value=-1)
        verts_data_faces_brain_lf = torch.cat(verts_data_faces_brain_lf)
        # verts_fiber_bounds_lf = torch.cat(verts_fiber_bounds_lf)
        # sample_min_max_lf = torch.cat(sample_min_max_lf)
        # data_lab_lf = torch.cat(data_lab_lf)
        # name_labels_lf = torch.cat(name_labels_lf)

        verts_tf = pad_sequence(verts_tf, batch_first=True, padding_value=0.0)
        faces_tf = pad_sequence(faces_tf, batch_first=True, padding_value=-1)
        # verts_data_vertex_tf = torch.cat(verts_data_vertex_tf)
        verts_data_faces_tf = torch.cat(verts_data_faces_tf)
        labels_tf = torch.cat(labels_tf)
        verts_fiber_tf = pad_sequence(verts_fiber_tf, batch_first=True, padding_value=0.0)
        faces_fiber_tf = pad_sequence(faces_fiber_tf, batch_first=True, padding_value=-1)
        verts_data_faces_fiber_tf = torch.cat(verts_data_faces_fiber_tf)
        labels_fiber_tf = torch.cat(labels_fiber_tf)
        verts_brain_tf = pad_sequence(verts_brain_tf, batch_first=True, padding_value=0.0)
        faces_brain_tf = pad_sequence(faces_brain_tf, batch_first=True, padding_value=-1)
        verts_data_faces_brain_tf = torch.cat(verts_data_faces_brain_tf)
        # verts_fiber_bounds_tf = torch.cat(verts_fiber_bounds_tf)
        # sample_min_max_tf = torch.cat(sample_min_max_tf)
        # data_lab_tf = torch.cat(data_lab_tf)
        # name_labels_tf = torch.cat(name_labels_tf)

        print("verts_lf", verts_lf.shape)
        print("faces_lf", faces_lf.shape)
        print("verts_data_faces_lf", verts_data_faces_lf.shape)
        print("labels_lf", labels_lf.shape)
        print("verts_fiber_lf", verts_fiber_lf.shape)
        print("faces_fiber_lf", faces_fiber_lf.shape)
        print("verts_data_faces_fiber_lf", verts_data_faces_fiber_lf.shape)
        print("labels_fiber_lf", labels_fiber_lf.shape)
        print("verts_brain_lf", verts_brain_lf.shape)
        print("faces_brain_lf", faces_brain_lf.shape)
        print("verts_data_faces_brain_lf", verts_data_faces_brain_lf.shape)
        # print("verts_fiber_bounds_lf", verts_fiber_bounds_lf.shape)
        # print("sample_min_max_lf", sample_min_max_lf.shape)
        # print("data_lab_lf", data_lab_lf.shape)
        # print("name_labels_lf", name_labels_lf.shape)
        print("verts_fiber_bounds_lf", verts_fiber_bounds_lf)
        print("sample_min_max_lf", sample_min_max_lf)
        print("data_lab_lf", data_lab_lf)
        print("name_labels_lf", name_labels_lf)

        print("verts_tf", verts_tf.shape)
        print("faces_tf", faces_tf.shape)
        print("verts_data_faces_tf", verts_data_faces_tf.shape)
        print("labels_tf", labels_tf.shape)
        print("verts_fiber_tf", verts_fiber_tf.shape)
        print("faces_fiber_tf", faces_fiber_tf.shape)
        print("verts_data_faces_fiber_tf", verts_data_faces_fiber_tf.shape)
        print("labels_fiber_tf", labels_fiber_tf.shape)
        print("verts_brain_tf", verts_brain_tf.shape)
        print("faces_brain_tf", faces_brain_tf.shape)
        print("verts_data_faces_brain_tf", verts_data_faces_brain_tf.shape)
        print("verts_fiber_bounds_tf", verts_fiber_bounds_tf)
        print("sample_min_max_tf", sample_min_max_tf)
        print("data_lab_tf", data_lab_tf)
        print("name_labels_tf", name_labels_tf)
        # print("verts_fiber_bounds_tf", verts_fiber_bounds_tf.shape)
        # print("sample_min_max_tf", sample_min_max_tf.shape)
        # print("data_lab_tf", data_lab_tf.shape)
        # print("name_labels_tf", name_labels_tf.shape)
        
        print(akjshksfajh)
        verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
        faces = pad_sequence(faces, batch_first=True, padding_value=-1)        
        verts_data_faces = torch.cat(verts_data_faces)
        labels = torch.cat(labels)
        verts_fiber = pad_sequence(verts_fiber, batch_first=True, padding_value=0.0)
        faces_fiber = pad_sequence(faces_fiber, batch_first=True, padding_value=-1)
        verts_data_faces_fiber = torch.cat(verts_data_faces_fiber)
        labels_fiber = torch.cat(labels_fiber)
        verts_brain = pad_sequence(verts_brain, batch_first=True, padding_value=0.0)
        faces_brain = pad_sequence(faces_brain, batch_first=True, padding_value=-1)
        verts_data_faces_brain = torch.cat(verts_data_faces_brain)
        data_lab = torch.cat(data_lab)
        name_labels = torch.cat(name_labels)


        return verts, faces, verts_data_faces, labels, verts_fiber, faces_fiber, verts_data_faces_fiber, labels_fiber, verts_brain, faces_brain, verts_data_faces_brain, verts_fiber_bounds, sample_min_max, data_lab, name_labels


