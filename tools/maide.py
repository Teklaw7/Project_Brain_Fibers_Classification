import utils
import vtk
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.vis.plotly_vis import plot_scene
import torch
import numpy as np
path = "/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg.vtp"
# path = "/CMF/data/timtey/tracts/archives/101006_tracts/T_AF_left.vtp"
bundle = utils.ReadSurf(path)
# print(bundle)
V = []
# for i in range(bundle.GetNumberOfCells()):
# print(i)
fiber = utils.ExtractFiber(bundle, 0)
# print(fiber)
fiber_tf = vtk.vtkTriangleFilter()
fiber_tf.SetInputData(fiber)
fiber_tf.Update()
fiber_extract_tf = fiber_tf.GetOutput()
verts, faces, edges = utils.PolyDataToTensors(fiber_extract_tf)
print(verts.shape)
# print(verts.shape)
# V.append(verts.shape[0])


# print(len(V))
# print(np.mean(V))
# print("V",verts.shape)
# print("F",faces.shape)
# print("E",edges.shape)
meshes_fiber = Meshes(
    verts=[verts],   
    faces=[faces],
    textures=None 
)
# 
fig = plot_scene({
        "display": {
            "mesh": meshes_fiber,
        }
    })
# 
fig.show()