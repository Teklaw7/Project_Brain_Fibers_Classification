import utils
import vtk
import numpy as np
import torch
path = "/CMF/data/timtey/tracts/archives/101006_tracts/T_AF_left.vtp"
tracts = utils.ReadSurf(path)
verts  = torch.ones([1000,3])
while verts.shape[0] < 100:
    print("in")
    n = np.random.randint(0,tracts.GetNumberOfCells())
    tracts_extract = utils.ExtractFiber(tracts,n)
    # name = [sample_id, sample_label, n]
    tracts_tf = vtk.vtkTriangleFilter()
    tracts_tf.SetInputData(tracts_extract)
    tracts_tf.Update()
    tracts_f = tracts_tf.GetOutput()
    verts, faces, edges = utils.PolyDataToTensors(tracts_f)
    # verts =torch.ones([10,3])
    print(verts.shape)

print("good")