from tools import utils
import vtk
import torch
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.structures import Meshes
path = "/CMF/data/timtey/tracts/archives/101006_tracts/T_CC1.vtp"
bundle = utils.ReadSurf(path)


fiber = utils.ExtractFiber(bundle, 0)
print(fiber)
cc1_tf=vtk.vtkTriangleFilter()
cc1_tf.SetInputData(fiber)
cc1_tf.Update()
cc1_extract_tf = cc1_tf.GetOutput()
verts,faces,edges = utils.PolyDataToNumpy(cc1_extract_tf)
print(verts.shape)
print(faces.shape)
print(edges.shape)

verts = torch.tensor(verts)
faces = torch.tensor(faces)
mesh = Meshes(verts=[verts], faces=[faces])

radius = 1.0
ico_lvl = 1
ico_sphere = utils.CreateIcosahedron(radius, ico_lvl)
ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
ico_sphere_verts = ico_sphere_verts
ico_sphere_faces = ico_sphere_faces
# ico_sphere_edges = np.array(self.ico_sphere_edges)
mesh2 = Meshes(verts=[ico_sphere_verts], faces=torch.zeros(0,3))


fig = plot_scene({
    "subplot1": {
        "cow_mesh_batch": mesh,
        "ico": mesh2
    }
})
fig.show()