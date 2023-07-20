from torch.utils.data import Dataset, DataLoader
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

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
from sklearn.utils.class_weight import compute_class_weight

class Bundles_Dataset_contrastive_labeled(Dataset):
    def __init__(self, data, column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
        self.data = data
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
        cc1_extract = utils.ExtractFiber(cc1,n)
        cc1_tf=vtk.vtkTriangleFilter()
        cc1_tf.SetInputData(cc1_extract)
        cc1_tf.Update()
        cc1_extract_tf = cc1_tf.GetOutput()
        
        verts, faces, edges = utils.PolyDataToTensors(cc1_extract_tf)    
        verts_fiber = torch.clone(verts)
        faces_fiber = torch.clone(faces)
        verts_fiber_bounds = cc1_extract_tf.GetBounds()
        verts_fiber_bounds = list(verts_fiber_bounds)
        max_bounds = max(verts_fiber_bounds)
        min_bounds = min(verts_fiber_bounds)
        verts_fiber_bounds = [min_bounds,max_bounds,min_bounds,max_bounds,min_bounds,max_bounds]
        sample_min_max = [sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max]

        EstimatedUncertainty = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("EstimatedUncertainty"))).unsqueeze(1)
        FA1 = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("FA1"))).unsqueeze(1)
        FA2 = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("FA2"))).unsqueeze(1)
        HemisphereLocataion = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("HemisphereLocataion"))).unsqueeze(1)
        # cluster_idx = vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("cluster_idx"))
        trace1 = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("trace1"))).unsqueeze(1)
        trace2 = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("trace2"))).unsqueeze(1)
        # vtkOriginalPointIds = vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("vtkOriginalPointIds"))
        TubeNormals = torch.tensor(vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("TubeNormals")))


        vertex_features = torch.cat([EstimatedUncertainty, FA1, FA2, HemisphereLocataion, trace1, trace2, TubeNormals], dim=1)
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
        ### labels ###
        labels = torch.tensor([sample_label])
        labels_fiber = torch.tensor([sample_label])
        verts_brain = torch.load(f"brain_structures/verts_brain_{sample_id}.pt")
        faces_brain = torch.load(f"brain_structures/faces_brain_{sample_id}.pt")
        face_features_brain = torch.load(f"brain_structures/face_features_brain_{sample_id}.pt")
        return verts,faces,face_features,labels,verts_fiber,faces_fiber,face_features_fiber,labels_fiber, verts_brain, faces_brain, face_features_brain, verts_fiber_bounds, sample_min_max



class Bundles_Dataset_test_contrastive_labeled(Dataset):
    def __init__(self, data, bundle, L, fibers, index_csv, column_class='class',column_id='id', column_label='label', column_x_min = 'x_min', column_x_max = 'x_max', column_y_min = 'y_min', column_y_max = 'y_max', column_z_min = 'z_min', column_z_max = 'z_max'):
        self.data = data
        self.bundle = bundle
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
        bundle_extract_tf = self.L[idx]
        verts, faces, edges = utils.PolyDataToTensors(bundle_extract_tf)  
        verts_fiber = torch.clone(verts)
        faces_fiber = torch.clone(faces)

        verts_fiber_bounds = bundle_extract_tf.GetBounds()
        verts_fiber_bounds = list(verts_fiber_bounds)
        max_bounds = max(verts_fiber_bounds)
        min_bounds = min(verts_fiber_bounds)
        verts_fiber_bounds = [min_bounds,max_bounds,min_bounds,max_bounds,min_bounds,max_bounds]
        sample_min_max = [sample_x_min, sample_x_max, sample_y_min, sample_y_max, sample_z_min, sample_z_max]

        EstimatedUncertainty = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("EstimatedUncertainty"))).unsqueeze(1)
        FA1 = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("FA1"))).unsqueeze(1)
        FA2 = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("FA2"))).unsqueeze(1)
        HemisphereLocataion = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("HemisphereLocataion"))).unsqueeze(1)
        # cluster_idx = vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("cluster_idx"))
        trace1 = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("trace1"))).unsqueeze(1)
        trace2 = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("trace2"))).unsqueeze(1)
        # vtkOriginalPointIds = vtk_to_numpy(cc1_extract_tf.GetPointData().GetScalars("vtkOriginalPointIds"))
        TubeNormals = torch.tensor(vtk_to_numpy(bundle_extract_tf.GetPointData().GetScalars("TubeNormals")))
        vertex_features = torch.cat([EstimatedUncertainty, FA1, FA2, HemisphereLocataion, trace1, trace2, TubeNormals], dim=1)

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
        ### labels ###
        labels = torch.tensor([sample_label])
        labels_fiber = torch.tensor([sample_label])

        verts_brain = torch.load(f"brain_structures/verts_brain_{sample_id}.pt")
        faces_brain = torch.load(f"brain_structures/faces_brain_{sample_id}.pt")
        face_features_brain = torch.load(f"brain_structures/face_features_brain_{sample_id}.pt")
        return verts,faces,face_features,labels,verts_fiber,faces_fiber,face_features_fiber,labels_fiber, verts_brain, faces_brain, face_features_brain, verts_fiber_bounds, sample_min_max




class Bundles_Dataset_contrastive_labeled(pl.LightningDataModule):
    def __init__(self, bundle, L, fibers, index_csv, batch_size, train_path, val_path, test_path, num_workers=12, transform=True, persistent_workers=False):
        super().__init__()
        self.bundle = bundle
        self.L = L
        self.fibers = fibers
        self.index_csv = index_csv
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
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
        
        self.train_dataset = Bundles_Dataset_contrastive_labeled(list_train_data)
        self.val_dataset = Bundles_Dataset_contrastive_labeled(list_val_data)
        self.test_dataset = Bundles_Dataset_test_contrastive_labeled(list_test_data, self.bundle, self.L, self.fibers, self.index_csv)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.pad_verts_faces, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def get_weights(self):
        return self.weights

    def pad_verts_faces(self, batch):
        verts = [v for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]
        faces = [f for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]        
        verts_data_faces = [vdf for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]        
        labels = [l for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]      
        verts_fiber = [vfi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]
        faces_fiber = [ffi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]
        verts_data_faces_fiber = [vdffi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]
        labels_fiber = [lfi for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]
        verts_brain = [vb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]
        faces_brain = [fb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]
        verts_data_faces_brain = [ffb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]
        verts_fiber_bounds = [vfb for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]
        sample_min_max = [smm for v, f, vdf, l, vfi, ffi, vdffi, lfi, vb, fb, ffb, vfb, smm in batch]
        
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
        
        



        return verts, faces, verts_data_faces, labels, verts_fiber, faces_fiber, verts_data_faces_fiber, labels_fiber, verts_brain, faces_brain, verts_data_faces_brain, verts_fiber_bounds, sample_min_max


