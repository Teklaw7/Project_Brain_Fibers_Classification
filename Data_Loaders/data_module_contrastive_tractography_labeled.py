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
    # def __init__(self, data, path_data, path_ico, verts_brain, faces_brain, face_features_brain, transform=True, column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
    def __init__(self, data, path_data, path_ico, transform=True, column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
        self.data = data
        self.transform = transform
        self.path_data = path_data
        self.path_ico = path_ico
        # self.verts_brain = verts_brain
        # self.faces_brain = faces_brain
        # self.face_features_brain = face_features_brain
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
        return len(self.data)

    def __getitem__(self, idx):

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
        nb_faces = len(faces)
        offset = torch.zeros((nb_faces, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])
        face_features = torch.take(vertex_features, faces_pid0_offset)

        ### labels ###
        labels = torch.tensor([sample_label])
        data_lab = [0]
        name_l = [name]
        # Fiber_infos = [verts_fiber_bounds, sample_min_max, data_lab, name_l]
        Fiber_infos = [sample_min_max, data_lab, name_l]

        return verts,faces,face_features,labels, Fiber_infos
    

class Bundles_Dataset_tractography(Dataset):
    # def __init__(self, data, path_data, path_ico, verts_brain, faces_brain, face_features_brain, length, tractography_list_vtk, transform=True, column_surf='surf',column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
    def __init__(self, data, path_data, path_ico, length, tractography_list_vtk, transform=True, column_surf='surf',column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
        self.data = data
        self.transform = transform
        self.path_data = path_data
        self.path_ico = path_ico
        # self.verts_brain = verts_brain
        # self.faces_brain = faces_brain
        # self.face_features_brain = face_features_brain
        self.length = length
        self.tractography_list_vtk = tractography_list_vtk
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
        return len(self.data)

    def __getitem__(self, idx):
        sample_row = self.data.loc[idx]
        sample_surf = sample_row[self.column_surf]
        sample_id, sample_class, sample_label = sample_row[self.column_id], sample_row[self.column_class], sample_row[self.column_label]
        sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max = sample_row[self.column_x_min], sample_row[self.column_x_max], sample_row[self.column_y_min], sample_row[self.column_y_max], sample_row[self.column_z_min], sample_row[self.column_z_max]
        list_sample_id = [102008, 103515, 108525, 113215, 119833, 121618, 124220, 124826, 139233]
        tracts_idx = list_sample_id.index(sample_id)
        tracts = self.tractography_list_vtk[tracts_idx]
        n = randint(0,tracts.GetNumberOfCells()-1)

        tracts_extract = utils.ExtractFiber(tracts,n)
        name = [sample_id, sample_label, n]
        tracts_tf = vtk.vtkTriangleFilter()
        tracts_tf.SetInputData(tracts_extract)
        tracts_tf.Update()
        tracts_f = tracts_tf.GetOutput()
        verts, faces, edges = utils.PolyDataToTensors(tracts_f)
        
        while verts.shape[0] < 100:
            n = randint(0,tracts.GetNumberOfCells()-1)
            tracts_extract = utils.ExtractFiber(tracts,n)
            name = [sample_id, sample_label, n]
            tracts_tf = vtk.vtkTriangleFilter()
            tracts_tf.SetInputData(tracts_extract)
            tracts_tf.Update()
            tracts_f = tracts_tf.GetOutput()
            verts, faces, edges = utils.PolyDataToTensors(tracts_f)

        sample_min_max = [sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max]

        TubeNormals = torch.tensor(vtk_to_numpy(tracts_f.GetPointData().GetScalars("TubeNormals")))
        vertex_features = torch.cat([TubeNormals], dim=1)
        faces_pid0 = faces[:,0:1]
        nb_faces = len(faces)
        offset = torch.zeros((nb_faces, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])
        face_features = torch.take(vertex_features, faces_pid0_offset)

        labels = torch.tensor([sample_label])
        data_lab = [1]
        name_l = [name]
        # Fiber_infos = [verts_fiber_bounds, sample_min_max, data_lab, name_l]
        Fiber_infos = [sample_min_max, data_lab, name_l]

        return verts,faces,face_features,labels, Fiber_infos


class Bundles_Dataset_test_contrastive_tractography_labeled(Dataset):
    # def __init__(self, contrastive, data, bundle, L, fibers, index_csv, path_data, path_ico, verts_brain, faces_brain, face_features_brain, transform=True, column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
    def __init__(self, contrastive, data, bundle, L, fibers, index_csv, path_data, path_ico, transform=True, column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
        self.contrastive = contrastive
        self.data = data
        self.bundle = bundle
        self.L = L
        self.fibers = fibers
        self.index_csv = index_csv
        self.transform = transform
        self.path_data = path_data
        self.path_ico = path_ico
        # self.verts_brain = verts_brain
        # self.faces_brain = faces_brain
        # self.face_features_brain = face_features_brain
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
        return self.fibers

    def __getitem__(self, idx):

        sample_row = self.data.loc[self.index_csv]
        sample_label = sample_row[self.column_label]
        sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max = sample_row[self.column_x_min], sample_row[self.column_x_max], sample_row[self.column_y_min], sample_row[self.column_y_max], sample_row[self.column_z_min], sample_row[self.column_z_max]
        sample_id = sample_row[self.column_id]
        bundle_extract_tf = self.L[idx]
        name = [sample_id, sample_label, idx]
        verts, faces, edges = utils.PolyDataToTensors(bundle_extract_tf)
        sample_min_max = [sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max]

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
        nb_faces = len(faces)
        offset = torch.zeros((nb_faces, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])
        face_features = torch.take(vertex_features, faces_pid0_offset)

        labels = torch.tensor([sample_label])
        data_lab = [2]
        name_l = [name]
        # Fiber_infos = [verts_fiber_bounds, sample_min_max, data_lab, name_l]
        Fiber_infos = [sample_min_max, data_lab, name_l]

        return verts,faces,face_features,labels,Fiber_infos



class Bundles_DataModule_tractography_labeled_fibers(pl.LightningDataModule):
    # def __init__(self, contrastive, bundle, L, fibers, fibers_valid, index_csv, path_data, path_ico, batch_size, train_path, val_path, test_path, verts_brain, faces_brain, face_features_brain, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=12, transform=True, persistent_workers=False):
    def __init__(self, contrastive, bundle, L, fibers, fibers_valid, index_csv, path_data, path_ico, batch_size, train_path, val_path, test_path, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=12, transform=True, persistent_workers=False):
    
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
        # self.verts_brain = verts_brain
        # self.faces_brain = faces_brain
        # self.face_features_brain = face_features_brain
        self.path_tractography_train = path_tractography_train
        self.path_tractography_valid = path_tractography_valid
        self.path_tractography_test = path_tractography_test
        self.tractography_list_vtk = tractography_list_vtk
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

        list_train_data = pd.read_csv(self.train_path)
        list_val_data = pd.read_csv(self.val_path)
        list_test_data = pd.read_csv(self.test_path)
        list_train_tractography_data = pd.read_csv(self.path_tractography_train)
        list_val_tractography_data = pd.read_csv(self.path_tractography_valid)
        list_test_tractography_data = pd.read_csv(self.path_tractography_test)
        len_train = len(list_train_data)
        len_val = len(list_val_data)
        len_test = len(list_test_data)
        
        self.train_dataset = Bundles_Dataset_contrastive_tractography_labeled(list_train_data, self.path_data, self.path_ico, self.transform)
        self.val_dataset = Bundles_Dataset_contrastive_tractography_labeled(list_val_data, self.path_data, self.path_ico, self.transform)
        self.test_dataset = Bundles_Dataset_test_contrastive_tractography_labeled(self.contrastive, list_test_data, self.bundle, self.L, self.fibers, self.index_csv, self.path_data, self.path_ico, self.transform)
        self.train_tractography_dataset = Bundles_Dataset_tractography(list_train_tractography_data, self.path_data, self.path_ico, len_train, self.tractography_list_vtk, self.transform)
        self.val_tractography_dataset = Bundles_Dataset_tractography(list_val_tractography_data, self.path_data, self.path_ico, len_val, self.tractography_list_vtk, self.transform)
        self.test_tractography_dataset = Bundles_Dataset_tractography(list_test_tractography_data, self.path_data, self.path_ico, len_test, self.tractography_list_vtk, self.transform)

        self.concatenated_train_dataset = ConcatDataset(self.train_dataset, self.train_tractography_dataset)
        self.concatenated_val_dataset = ConcatDataset(self.val_dataset, self.val_tractography_dataset)
        self.concatenated_test_dataset = ConcatDataset(self.test_dataset, self.test_tractography_dataset)

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces_simple, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        # return DataLoader(self.train_tractography_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces_simple, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers) 
        return DataLoader(self.concatenated_train_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces_simple, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        # return DataLoader(self.val_tractography_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces_simple, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        return DataLoader(self.concatenated_val_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        # return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces_simple, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        # return DataLoader(self.test_tractography_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces_simple, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        return DataLoader(self.concatenated_test_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def get_weights(self):
        return self.weights



    def pad_verts_faces_simple(self, batch):
        verts = [v for v, f, vdf, l ,f_infos in batch]
        faces = [f for v, f, vdf, l ,f_infos in batch]
        verts_data_faces = [vdf for v, f, vdf, l ,f_infos in batch]
        labels = [l for v, f, vdf, l ,f_infos in batch]
        faces_infos = [f_infos for v, f, vdf, l ,f_infos in batch]

        verts = pad_sequence(verts, batch_first=True, padding_value=0)
        faces = pad_sequence(faces, batch_first=True, padding_value=-1)
        verts_data_faces = torch.cat(verts_data_faces)
        labels = torch.cat(labels)
        return verts, faces, verts_data_faces, labels, faces_infos


    def pad_verts_faces(self, batch):
        labeled_fibers = ()
        tractography_fibers = ()
        for i in range(len(batch)):
            labeled_fibers += (batch[i][0],)
            tractography_fibers += (batch[i][1],)

        verts_lf = [v for v, f, vdf, l, f_infos in labeled_fibers]
        faces_lf = [f for v, f, vdf, l, f_infos in labeled_fibers]
        verts_data_faces_lf = [vdf for v, f, vdf, l, f_infos in labeled_fibers]
        labels_lf = [l for v, f, vdf, l, f_infos in labeled_fibers]
        f_infos_lf = [f_infos for v, f, vdf, l, f_infos in labeled_fibers]

        verts_tf = [v for v, f, vdf, l, f_infos in tractography_fibers]
        faces_tf = [f for v, f, vdf, l, f_infos in tractography_fibers]
        verts_data_faces_tf = [vdf for v, f, vdf, l, f_infos in tractography_fibers]
        labels_tf = [l for v, f, vdf, l, f_infos in tractography_fibers]
        f_infos_tf = [f_infos for v, f, vdf, l, f_infos in tractography_fibers]

        verts = verts_lf + verts_tf
        faces = faces_lf + faces_tf
        verts_data_faces = verts_data_faces_lf + verts_data_faces_tf
        labels = labels_lf + labels_tf
        f_infos = f_infos_lf + f_infos_tf

        verts = pad_sequence(verts, batch_first=True, padding_value=0.0)
        faces = pad_sequence(faces, batch_first=True, padding_value=-1)

        verts_data_faces = torch.cat(verts_data_faces)
        labels = torch.cat(labels)

        return verts, faces, verts_data_faces, labels, f_infos
