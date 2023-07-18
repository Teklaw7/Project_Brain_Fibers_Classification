from torch.utils.data import Dataset, DataLoader#, ConcatDataset
import torch
from tools import utils
import pytorch_lightning as pl 
import vtk
from pytorch3d.structures import Meshes
from random import *
from pytorch3d.vis.plotly_vis import plot_scene
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
# from pytorch3d.renderer import TexturesVertex
# from torch.utils.data._utils.collate import default_collate
import pandas as pd
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from itertools import cycle
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
from sklearn.utils.class_weight import compute_class_weight
from Transformations.transformations import *

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class Bundles_Dataset_contrastive_tractography_labeled(Dataset):
    def __init__(self, data, column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
        self.data = data    #csv file with the data
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
        # path_cc1 = f"/CMF/data/timtey/tracts/archives/{sample_id}_tracts/{sample_class}.vtp"
        path_cc1 = f"/CMF/data/timtey/tracts/archives/{sample_id}_tracts/{sample_class}_DTI.vtk"
        cc1 = utils.ReadSurf(path_cc1)
        n = randint(0,cc1.GetNumberOfCells()-1)
        name = [sample_id, sample_label, n]
        cc1_extract = utils.ExtractFiber(cc1,n)
        cc1_tf=vtk.vtkTriangleFilter()
        cc1_tf.SetInputData(cc1_extract)
        cc1_tf.Update()
        cc1_extract_tf = cc1_tf.GetOutput()
        
        verts, faces, edges = utils.PolyDataToTensors(cc1_extract_tf)
        verts_fiber_bounds = cc1_extract_tf.GetBounds()
        mean_v, scale_factor_v = get_mean_scale_factor(verts_fiber_bounds)
        sample_min_max = [sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max]
        mean_s, scale_factor_s = get_mean_scale_factor(sample_min_max)

        TubeNormals = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("TubeNormals")))
        FA = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("FA"))).unsqueeze(1)
        MD = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("MD"))).unsqueeze(1)
        AD = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("AD"))).unsqueeze(1)
        RD = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("RD"))).unsqueeze(1)

        vertex_features = torch.cat([TubeNormals, FA, MD, AD, RD], dim=1)
        faces_pid0 = faces[:,0:1]
        nb_faces = len(faces)
        offset = torch.zeros((nb_faces, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])
        face_features = torch.take(vertex_features, faces_pid0_offset)

        ### labels ###
        labels = torch.tensor([sample_label])
        data_lab = torch.tensor([0]).unsqueeze(0) #1x1
        name_l = torch.tensor(name).unsqueeze(0) #1x3
        Fiber_infos = torch.cat((data_lab, name_l), dim=1) #1x4
        Mean = torch.cat((torch.tensor(mean_v).unsqueeze(0), torch.tensor(mean_s).unsqueeze(0)), dim=1) # 1 x 6
        Scale = torch.cat((torch.tensor(scale_factor_v).unsqueeze(0), torch.tensor(scale_factor_s).unsqueeze(0)), dim=0).unsqueeze(0) # 1 x 2
        return verts,faces,face_features,labels, Fiber_infos, Mean, Scale
    

class Bundles_Dataset_tractography(Dataset):
    def __init__(self, data, tractography_list_vtk, column_surf='surf',column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
        self.data = data # csv file
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
        list_sample_id = ['102008_1', '102008_2', '102008_3', '102008_4',
                          '103515_1', '103515_2', '103515_3', '103515_4',
                          '108525_1', '108525_2', '108525_3', '108525_4',
                          '113215_1', '113215_2', '113215_3', '113215_4',
                          '119833_1', '119833_2', '119833_3', '119833_4',
                          '121618_1', '121618_2', '121618_3', '121618_4',
                          '124220_1', '124220_2', '124220_3', '124220_4',
                          '124826_1', '124826_2', '124826_3', '124826_4',
                          '139233_1', '139233_2', '139233_3', '139233_4']
        tracts_idx = list_sample_id.index(str(sample_id))
        tracts = self.tractography_list_vtk[tracts_idx]
        n = randint(0,tracts.GetNumberOfCells()-1)

        tracts_extract = utils.ExtractFiber(tracts,n)
        name = [int(sample_id), sample_label, n]
        tracts_tf = vtk.vtkTriangleFilter()
        tracts_tf.SetInputData(tracts_extract)
        tracts_tf.Update()
        tracts_f = tracts_tf.GetOutput()
        verts, faces, edges = utils.PolyDataToTensors(tracts_f)
        
        while verts.shape[0] < 100:
            n = randint(0,tracts.GetNumberOfCells()-1)
            tracts_extract = utils.ExtractFiber(tracts,n)
            name = [int(sample_id), sample_label, n]
            tracts_tf = vtk.vtkTriangleFilter()
            tracts_tf.SetInputData(tracts_extract)
            tracts_tf.Update()
            tracts_f = tracts_tf.GetOutput()
            verts, faces, edges = utils.PolyDataToTensors(tracts_f)

        verts_fiber_bounds = tracts_f.GetBounds()
        mean_v, scale_factor_v = get_mean_scale_factor(verts_fiber_bounds)
        sample_min_max = [sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max]
        mean_s, scale_factor_s = get_mean_scale_factor(sample_min_max)
        TubeNormals = torch.tensor(vtk_to_numpy(tracts_f.GetPointData().GetScalars("TubeNormals")))
        FA = torch.tensor(vtk_to_numpy(tracts_f.GetPointData().GetScalars("FA"))).unsqueeze(1)
        MD = torch.tensor(vtk_to_numpy(tracts_f.GetPointData().GetScalars("MD"))).unsqueeze(1)
        AD = torch.tensor(vtk_to_numpy(tracts_f.GetPointData().GetScalars("AD"))).unsqueeze(1)
        RD = torch.tensor(vtk_to_numpy(tracts_f.GetPointData().GetScalars("RD"))).unsqueeze(1)
        vertex_features = torch.cat([TubeNormals, FA, MD, AD, RD], dim=1)
        faces_pid0 = faces[:,0:1]
        nb_faces = len(faces)
        offset = torch.zeros((nb_faces, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])
        face_features = torch.take(vertex_features, faces_pid0_offset)

        labels = torch.tensor([sample_label])
        data_lab = torch.tensor([1]).unsqueeze(0) #1x1
        name_l = torch.tensor(name).unsqueeze(0) #1x3
        Fiber_infos = torch.cat((data_lab, name_l), dim=1) #1x4
        mean_v = torch.tensor(mean_v).unsqueeze(0) #1x3
        mean_s = torch.tensor(mean_s).unsqueeze(0) #1x3
        Mean = torch.cat((mean_v, mean_s), dim=1) # 1x6
        Scale = torch.cat((torch.tensor(scale_factor_v).unsqueeze(0), torch.tensor(scale_factor_s).unsqueeze(0)), dim=0).unsqueeze(0) #1x2

        return verts,faces,face_features,labels, Fiber_infos, Mean, Scale


class Bundles_Dataset_test_contrastive_tractography_labeled(Dataset):
    def __init__(self, data, L, fibers, index_csv, column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
        self.data = data
        self.L = L
        self.fibers = fibers
        self.index_csv = index_csv
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
        n = randint(0,len(self.L)-1)
        bundle_extract_tf = self.L[n]
        name = [sample_id, sample_label, n]
        verts, faces, edges = utils.PolyDataToTensors(bundle_extract_tf)
        verts_fiber_bounds = bundle_extract_tf.GetBounds()
        mean_v, scale_factor_v = get_mean_scale_factor(verts_fiber_bounds)
        sample_min_max = [sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max]
        mean_s, scale_factor_s = get_mean_scale_factor(sample_min_max)

        TubeNormals = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("TubeNormals")))
        FA = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("FA"))).unsqueeze(1)
        MD = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("MD"))).unsqueeze(1)
        AD = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("AD"))).unsqueeze(1)
        RD = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("RD"))).unsqueeze(1)

        vertex_features = torch.cat([TubeNormals, FA, MD, AD, RD], dim=1)
        faces_pid0 = faces[:,0:1]
        nb_faces = len(faces)
        offset = torch.zeros((nb_faces, vertex_features.shape[1]), dtype=int) + torch.arange(vertex_features.shape[1]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])
        face_features = torch.take(vertex_features, faces_pid0_offset)

        labels = torch.tensor([sample_label])
        data_lab = torch.tensor([2]).unsqueeze(0) # 1 x 1
        name_l = torch.tensor(name).unsqueeze(0) # 1 x 3
        Fiber_infos = torch.cat((data_lab, name_l), dim=1) # 1 x 4
        mean_v = torch.tensor(mean_v).unsqueeze(0) # 1 x 3
        mean_s = torch.tensor(mean_s).unsqueeze(0) # 1 x 3
        Mean = torch.cat((mean_v, mean_s), dim=1) # 1 x 6
        scale_factor_v = torch.tensor(scale_factor_v).unsqueeze(0) # 1 
        scale_factor_s = torch.tensor(scale_factor_s).unsqueeze(0) # 1 
        Scale = torch.cat((scale_factor_v, scale_factor_s), dim=0).unsqueeze(0) # 1 x 2

        return verts,faces,face_features,labels,Fiber_infos, Mean, Scale



class Bundles_DataModule_tractography_labeled_fibers(pl.LightningDataModule):
    def __init__(self, L, fibers, index_csv, path_data, batch_size, train_path, val_path, test_path, path_tractography_train, path_tractography_valid, path_tractography_test, tractography_list_vtk, num_workers=12, persistent_workers=False):
    
        super().__init__()
        self.L = L
        self.fibers = fibers
        self.index_csv = index_csv
        self.path_data = path_data
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.path_tractography_train = path_tractography_train
        self.path_tractography_valid = path_tractography_valid
        self.path_tractography_test = path_tractography_test
        self.tractography_list_vtk = tractography_list_vtk
        self.num_workers = num_workers
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
        
        self.train_dataset = Bundles_Dataset_contrastive_tractography_labeled(list_train_data)
        self.val_dataset = Bundles_Dataset_contrastive_tractography_labeled(list_val_data)
        self.test_dataset = Bundles_Dataset_test_contrastive_tractography_labeled(list_test_data, self.L, self.fibers, self.index_csv)
        self.train_tractography_dataset = Bundles_Dataset_tractography(list_train_tractography_data, self.tractography_list_vtk)
        self.val_tractography_dataset = Bundles_Dataset_tractography(list_val_tractography_data, self.tractography_list_vtk)
        self.test_tractography_dataset = Bundles_Dataset_tractography(list_test_tractography_data, self.tractography_list_vtk)

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
        return DataLoader(self.concatenated_test_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def get_weights(self):
        return self.weights

    def pad_verts_faces_simple(self, batch):
        verts = [v for v, f, vdf, l ,f_infos, m, s in batch]
        faces = [f for v, f, vdf, l ,f_infos, m, s  in batch]
        verts_data_faces = [vdf for v, f, vdf, l ,f_infos, m, s in batch]
        labels = [l for v, f, vdf, l ,f_infos, m, s in batch]
        f_infos = [f_infos for v, f, vdf, l ,f_infos, m, s in batch]
        mean = [m for v, f, vdf, l ,f_infos, m, s in batch]
        scale = [s for v, f, vdf, l ,f_infos, m, s in batch]

        verts = pad_sequence(verts, batch_first=True, padding_value=0)
        faces = pad_sequence(faces, batch_first=True, padding_value=-1)
        verts_data_faces = torch.cat(verts_data_faces)
        labels = torch.cat(labels)
        f_infos = torch.cat(f_infos)
        mean = torch.cat(mean)
        scale = torch.cat(scale)

        return verts, faces, verts_data_faces, labels, f_infos, mean, scale

    def pad_verts_faces(self, batch):
        labeled_fibers = ()
        tractography_fibers = ()
        for i in range(len(batch)):
            labeled_fibers += (batch[i][0],)
            tractography_fibers += (batch[i][1],)

        verts_lf = [v for v, f, vdf, l, f_infos, m, s in labeled_fibers]
        faces_lf = [f for v, f, vdf, l, f_infos, m, s in labeled_fibers]
        verts_data_faces_lf = [vdf for v, f, vdf, l, f_infos, m, s in labeled_fibers]
        labels_lf = [l for v, f, vdf, l, f_infos, m, s in labeled_fibers]
        f_infos_lf = [f_infos for v, f, vdf, l, f_infos, m, s in labeled_fibers]
        mean_lf = [m for v, f, vdf, l, f_infos, m, s in labeled_fibers]
        scale_lf = [s for v, f, vdf, l, f_infos, m, s in labeled_fibers]

        verts_tf = [v for v, f, vdf, l, f_infos, m, s in tractography_fibers]
        faces_tf = [f for v, f, vdf, l, f_infos, m, s in tractography_fibers]
        verts_data_faces_tf = [vdf for v, f, vdf, l, f_infos, m, s in tractography_fibers]
        labels_tf = [l for v, f, vdf, l, f_infos, m, s in tractography_fibers]
        f_infos_tf = [f_infos for v, f, vdf, l, f_infos, m, s in tractography_fibers]
        mean_tf = [m for v, f, vdf, l, f_infos, m, s in tractography_fibers]
        scale_tf = [s for v, f, vdf, l, f_infos, m, s in tractography_fibers]

        verts = verts_lf + verts_tf
        faces = faces_lf + faces_tf
        verts_data_faces = verts_data_faces_lf + verts_data_faces_tf
        labels = labels_lf + labels_tf
        f_infos = f_infos_lf + f_infos_tf
        mean = mean_lf + mean_tf
        scale = scale_lf + scale_tf
        verts = pad_sequence(verts, batch_first=True, padding_value=0.0)
        faces = pad_sequence(faces, batch_first=True, padding_value=-1)

        verts_data_faces = torch.cat(verts_data_faces)
        labels = torch.cat(labels)
        f_infos = torch.cat(f_infos)
        mean = torch.cat(mean)
        scale = torch.cat(scale)

        return verts, faces, verts_data_faces, labels, f_infos, mean, scale
