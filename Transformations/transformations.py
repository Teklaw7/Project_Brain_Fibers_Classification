import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl 
import torchvision.models as models
from torch.nn.functional import softmax
import torchmetrics
from tools import utils
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)
from pytorch3d.renderer.blending import sigmoid_alpha_blend, hard_rgb_blend
from pytorch3d.structures import Meshes, join_meshes_as_scene
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pytorch3d.vis.plotly_vis import plot_scene
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from sklearn.utils.class_weight import compute_class_weight
import random
import pytorch3d.transforms as T3d
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# import MLP
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
import pandas as pd

def transformation_verts_by_fiber(verts, mean_f, scale_f):
    va = verts - mean_f
    scale_f = scale_f*0.6
    for i in range(va.shape[0]):
        va[i,:,:] = va[i,:,:]*scale_f[i]
    return va

def transformation_verts(verts, mean_v, scale_v):
    va = verts - mean_v
    for i in range(va.shape[0]):
        va[i,:,:] = va[i,:,:]*scale_v[i]
    return va

class RotationTransform:
    def __call__(self, verts, rotation_matrix):
        b = torch.transpose(verts,0,1)
        a= torch.mm(rotation_matrix,torch.transpose(verts,0,1))
        verts = torch.transpose(torch.mm(rotation_matrix,torch.transpose(verts,0,1)),0,1)
        return verts
    
def randomrotation(verts):
    verts_device = verts.get_device()
    rotation_matrix = T3d.random_rotation().to(verts_device)
    rotation_transform = RotationTransform()
    verts = rotation_transform(verts,rotation_matrix)
    return verts


def randomrot(verts):
    verts_i = verts.clone()
    lim = 5*np.pi/180
    gauss_law = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([lim/3]))
    x_sample = gauss_law.sample().item()
    Rx = torch.tensor([[1,0,0],[0,torch.cos(x_sample),-torch.sin(x_sample)],[0,torch.sin(x_sample),torch.cos(x_sample)]]).to(verts.get_device())
    y_sample = gauss_law.sample().item()
    Ry = torch.tensor([[torch.cos(y_sample),0,torch.sin(y_sample)],[0,1,0],[-torch.sin(y_sample),0,torch.cos(y_sample)]]).to(verts.get_device())
    z_sample = gauss_law.sample().item()
    Rz = torch.tensor([[torch.cos(z_sample),-torch.sin(z_sample),0],[torch.sin(z_sample),torch.cos(z_sample),0],[0,0,1]]).to(verts.get_device())
    verts_i[:,:,0] = verts[:,:,0]@Rx #multiplication btw the 2 matrix
    verts_i[:,:,1] = verts[:,:,1]@Ry
    verts_i[:,:,2] = verts[:,:,2]@Rz
    return verts_i


def randomstretching(verts):
    verts_i = verts.clone()
    gauss_law = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([0.1]))
    for i in range(verts_i.shape[0]):
        M = torch.tensor([[gauss_law.sample().item(),0,0],[0,gauss_law.sample().item(),0],[0,0,gauss_law.sample().item()]]).to(verts.get_device())
        M = M.to(torch.float32)
        verts_i[i,:,:] = verts[i,:,:]@M #multiplication btw the 2 matrix
    return verts_i

def get_mean_scale_factor(bounds):
    bounds = np.array(bounds)
    mean_f = [0.0]*3
    bounds_max_f = [0.0]*3
    mean_f[0] = (bounds[0]+bounds[1])/2.0
    mean_f[1] = (bounds[2]+bounds[3])/2.0
    mean_f[2] = (bounds[4]+bounds[5])/2.0
    mean_f = np.array(mean_f)
    bounds_max_f[0] = max(bounds[0],bounds[1])
    bounds_max_f[1] = max(bounds[2],bounds[3])
    bounds_max_f[2] = max(bounds[4],bounds[5])
    bounds_max_f = np.array(bounds_max_f)
    scale_f = 1/np.linalg.norm(bounds_max_f-mean_f)

    return mean_f, scale_f

# def adjust_edge_curvatures(source, curvature_name, epsilon=1.0e-08):
#     """
#     This function adjusts curvatures along the edges of the surface by replacing
#      the value with the average value of the curvatures of points in the neighborhood.

#     Remember to update the vtkCurvatures object before calling this.

#     :param source: A vtkPolyData object corresponding to the vtkCurvatures object.
#     :param curvature_name: The name of the curvature, 'Gauss_Curvature' or 'Mean_Curvature'.
#     :param epsilon: Absolute curvature values less than this will be set to zero.
#     :return:
#     """

#     def point_neighbourhood(pt_id):
#         """
#         Find the ids of the neighbours of pt_id.

#         :param pt_id: The point id.
#         :return: The neighbour ids.
#         """
#         """
#         Extract the topological neighbors for point pId. In two steps:
#         1) source.GetPointCells(pt_id, cell_ids)
#         2) source.GetCellPoints(cell_id, cell_point_ids) for all cell_id in cell_ids
#         """
#         cell_ids = vtkIdList()
#         source.GetPointCells(pt_id, cell_ids)
#         neighbour = set()
#         for cell_idx in range(0, cell_ids.GetNumberOfIds()):
#             cell_id = cell_ids.GetId(cell_idx)
#             cell_point_ids = vtkIdList()
#             source.GetCellPoints(cell_id, cell_point_ids)
#             for cell_pt_idx in range(0, cell_point_ids.GetNumberOfIds()):
#                 neighbour.add(cell_point_ids.GetId(cell_pt_idx))
#         return neighbour

#     def compute_distance(pt_id_a, pt_id_b):
#         """
#         Compute the distance between two points given their ids.

#         :param pt_id_a:
#         :param pt_id_b:
#         :return:
#         """
#         pt_a = np.array(source.GetPoint(pt_id_a))
#         pt_b = np.array(source.GetPoint(pt_id_b))
#         return np.linalg.norm(pt_a - pt_b)

#     # Get the active scalars
#     source.GetPointData().SetActiveScalars(curvature_name)
#     np_source = dsa.WrapDataObject(source)
#     curvatures = np_source.PointData[curvature_name]

#     #  Get the boundary point IDs.
#     array_name = 'ids'
#     id_filter = vtkIdFilter()
#     id_filter.SetInputData(source)
#     id_filter.SetPointIds(True)
#     id_filter.SetCellIds(False)
#     id_filter.SetPointIdsArrayName(array_name)
#     id_filter.SetCellIdsArrayName(array_name)
#     id_filter.Update()

#     edges = vtkFeatureEdges()
#     edges.SetInputConnection(id_filter.GetOutputPort())
#     edges.BoundaryEdgesOn()
#     edges.ManifoldEdgesOff()
#     edges.NonManifoldEdgesOff()
#     edges.FeatureEdgesOff()
#     edges.Update()

#     edge_array = edges.GetOutput().GetPointData().GetArray(array_name)
#     boundary_ids = []
#     for i in range(edges.GetOutput().GetNumberOfPoints()):
#         boundary_ids.append(edge_array.GetValue(i))
#     # Remove duplicate Ids.
#     p_ids_set = set(boundary_ids)

#     # Iterate over the edge points and compute the curvature as the weighted
#     # average of the neighbours.
#     count_invalid = 0
#     for p_id in boundary_ids:
#         p_ids_neighbors = point_neighbourhood(p_id)
#         # Keep only interior points.
#         p_ids_neighbors -= p_ids_set
#         # Compute distances and extract curvature values.
#         curvs = [curvatures[p_id_n] for p_id_n in p_ids_neighbors]
#         dists = [compute_distance(p_id_n, p_id) for p_id_n in p_ids_neighbors]
#         curvs = np.array(curvs)
#         dists = np.array(dists)
#         curvs = curvs[dists > 0]
#         dists = dists[dists > 0]
#         if len(curvs) > 0:
#             weights = 1 / np.array(dists)
#             weights /= weights.sum()
#             new_curv = np.dot(curvs, weights)
#         else:
#             # Corner case.
#             count_invalid += 1
#             # Assuming the curvature of the point is planar.
#             new_curv = 0.0
#         # Set the new curvature value.
#         curvatures[p_id] = new_curv

#     #  Set small values to zero.
#     if epsilon != 0.0:
#         curvatures = np.where(abs(curvatures) < epsilon, 0, curvatures)
#         # Curvatures is now an ndarray
#         curv = numpy_support.numpy_to_vtk(num_array=curvatures.ravel(),
#                                           deep=True,
#                                           array_type=VTK_DOUBLE)
#         curv.SetName(curvature_name)
#         source.GetPointData().RemoveArray(curvature_name)
#         source.GetPointData().AddArray(curv)
#         source.GetPointData().SetActiveScalars(curvature_name)