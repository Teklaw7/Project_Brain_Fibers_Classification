import pandas
import numpy as np
import utils
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.structures import Meshes, join_meshes_as_scene
import vtk
from pytorch3d.renderer.blending import sigmoid_alpha_blend, hard_rgb_blend
import torch
from pytorch3d.renderer import TexturesVertex
from vtk.util.numpy_support import vtk_to_numpy
path ="/home/timtey/Documents/Projet/dataset4/tracts_filtered_train_train_label_to_number.csv"

df = pandas.read_csv(path)
b = utils.ReadSurf("/tools/atlas/Surface/CIVET_160K/icbm_surface/icbm_avg_mid_sym_mc_right_hires.vtk")
b2 = utils.ReadSurf("/tools/atlas/Surface/CIVET_160K/icbm_surface/icbm_avg_mid_sym_mc_left_hires.vtk")
bundle_tf=vtk.vtkTriangleFilter()
bundle_tf.SetInputData(b)
bundle_tf.Update()
bundle_extract_tf = bundle_tf.GetOutput()
verts, faces, edges = utils.PolyDataToTensors(bundle_extract_tf)
def normalize(verts):
    verts[:,0] = (2*(verts[:,0] - df["x_min"][0])/(df["x_max"][0] - df["x_min"][0])) - 1
    verts[:,1] = (2*(verts[:,1] - df["y_min"][0])/(df["y_max"][0] - df["y_min"][0])) - 1
    verts[:,2] = (2*(verts[:,2] - df["z_min"][0])/(df["z_max"][0] - df["z_min"][0])) - 1
normalize(verts)
meshes = Meshes(verts=[verts], faces=[faces], textures=None)
bundle_tf=vtk.vtkTriangleFilter()
bundle_tf.SetInputData(b2)
bundle_tf.Update()
bundle_extract_tf = bundle_tf.GetOutput()
verts2, faces2, edges2 = utils.PolyDataToTensors(bundle_extract_tf)
normalize(verts2)
meshes2 = Meshes(verts=[verts2], faces=[faces2], textures=None)
alpha = 0.5
joined_meshes = join_meshes_as_scene([meshes, meshes2])

pathh = "/MEDUSA_STOR/timtey/tractography/test/tractogram_deterministic_139233_dg2.vtk"
fiber = utils.ReadSurf(pathh)
verts3, faces3, edges3 = utils.PolyDataToTensors(fiber)
lksd = fiber.GetPointData().GetScalars("colors")
colors = torch.tensor(vtk_to_numpy(lksd))
vertex_features = torch.cat([colors], dim=1)
textures = TexturesVertex(verts_features=torch.ones(verts.shape))
mesh = Meshes(verts=[verts3], faces=[faces3], textures=textures)


fig2 = plot_scene({
           "display of all fibers": {
               "mesh": joined_meshes,
           }
       })
# fig2.show()