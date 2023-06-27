from tools import utils
import vtk
import numpy as np
import torch
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter

obj = utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_139233_dg_ex.vtp")
# verts, faces, edges = utils.PolyDataToTensors(obj)
# print(verts.shape)
# print(verts)
# print(faces.shape)
# print(edges.shape)
# print(torch.min(verts[:, 0]))
# print(torch.max(verts[:, 0]))

# thetax = torch.tensor(0)
# rotx = torch.tensor([[1, 0, 0], [0, torch.cos(thetax), -torch.sin(thetax)], [0, torch.sin(thetax), torch.cos(thetax)]])
# thetay = torch.tensor(0)
# roty = torch.tensor([[torch.cos(thetay), 0, torch.sin(thetay)], [0, 1, 0], [-torch.sin(thetay), 0, torch.cos(thetay)]])
# thetaz = torch.tensor(180)
# rotz = torch.tensor([[torch.cos(thetaz), -torch.sin(thetaz), 0], [torch.sin(thetaz), torch.cos(thetaz), 0], [0, 0, 1]])

# # verts[:,0] = verts[:,0]@rotx
# # verts[:,1] = verts[:,1]@roty
# # verts[:,2] = verts[:,2]@rotz
# verts = verts@rotz
# print(verts.shape)
# print(faces.shape)
# print(edges.shape)
# print(verts)
# print(torch.min(verts[:, 0]))
# print(torch.max(verts[:, 0]))
# print(obj.GetPoints().GetData())
# obj.GetPoints().SetData(utils.TensorsToVTKPoints(verts))
# f = obj.GetPoints().GetData()
# print(f)
# print(verts)
# verts = numpy_to_vtk(verts)
# print(verts)
# obj.GetPoints().SetData(verts)
# print(obj.GetPoints().GetData())
transform = vtkTransform()
transform.RotateZ(180)
transformFilter = vtkTransformPolyDataFilter()
transformFilter.SetInputData(obj)
transformFilter.SetTransform(transform)
transformFilter.Update()
obj = transformFilter.GetOutput()
# print(obj.GetPoints().GetData())
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("/CMF/data/timtey/tractography/all/tractogram_deterministic_139233_dg_ex_flip.vtp")
writer.SetInputData(obj)
writer.Write()
# writer = vtk.vtkPolyDataWriter()
# writer.SetFileName("/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg_flip.vtk")
# writer.SetInputData(obj)
# writer.Write()

# x_sample = gauss_law.sample().item()
#     Rx = torch.tensor([[1,0,0],[0,torch.cos(x_sample),-torch.sin(x_sample)],[0,torch.sin(x_sample),torch.cos(x_sample)]]).to(verts.get_device())
#     y_sample = gauss_law.sample().item()
#     Ry = torch.tensor([[torch.cos(y_sample),0,torch.sin(y_sample)],[0,1,0],[-torch.sin(y_sample),0,torch.cos(y_sample)]]).to(verts.get_device())
#     z_sample = gauss_law.sample().item()
#     Rz = torch.tensor([[torch.cos(z_sample),-torch.sin(z_sample),0],[torch.sin(z_sample),torch.cos(z_sample),0],[0,0,1]]).to(verts.get_device())
#     verts_i[:,:,0] = verts[:,:,0]@Rx #multiplication btw the 2 matrix
#     verts_i[:,:,1] = verts[:,:,1]@Ry
#     verts_i[:,:,2] = verts[:,:,2]@Rz

