import dipy
import numpy as np
from dipy.io.image import load_nifti, load_nifti_data
import vtk

import utils

# print(brain_mask)
from vtk import vtkMarchingCubes, vtkDiscreteMarchingCubes
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.renderer.blending import sigmoid_alpha_blend, hard_rgb_blend
# Create the data we want to contour

# marching = vtk.vtkMarchingCubes()
# marching.SetInputData(brain_mask)
# marching.SetValue(0, 1)
# marching.Update()
# 

path_brain_mask = "/CMF/data/timtey/tractography/training/135528/whiteMatterFA_smooth_mask.nii.gz"
path_brain_mask2 = "/CMF/data/timtey/tractography/training/135528/whiteMatterFA0.2_mask.nii.gz"




brain_mask = load_nifti_data(path_brain_mask)
brain_mask2 = load_nifti_data(path_brain_mask2)
reader = vtk.vtkNIFTIImageReader()
reader.SetFileName(path_brain_mask)
reader.Update()
im = reader.GetOutput()

reader2 = vtk.vtkNIFTIImageReader()
reader2.SetFileName(path_brain_mask2)
reader2.Update()
im2 = reader2.GetOutput()

def MarchingCubes(image,threshold):
    '''
    http://www.vtk.org/Wiki/VTK/Examples/Cxx/Modelling/ExtractLargestIsosurface 
    '''
    mc = vtk.vtkMarchingCubes()
    # mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.SetValue(0, threshold)
    mc.Update()
    
    # To remain largest region
    confilter =vtk.vtkPolyDataConnectivityFilter()
    confilter.SetInputData(mc.GetOutput())
    confilter.SetExtractionModeToLargestRegion()
    confilter.Update()

    return confilter.GetOutput()


poly = MarchingCubes(im, 1)
poly2 = MarchingCubes(im2, 1)
# print("poly",poly)

# f = utils.Ex
# print("poly",poly)
# writer = vtk.vtkPolyDataWriter()
# writer.SetInputData(poly)
# writer.SetFileName('/CMF/data/timtey/tractography/training/102008/brain_mask_102008.vtp')

# # writer.Write()
# # writer.Update()
# writer.Write()

utils.WriteSurf(poly, '/CMF/data/timtey/tractography/training/135528/brain_mask_135528.vtk')
# bbb = utils.ReadSurf("/CMF/data/timtey/tractography/validation/139233/brain_mask_139233.vtk")
# print(bbb)  



verts, faces, edges = utils.PolyDataToTensors(poly)
verts2, faces2, edges2 = utils.PolyDataToTensors(poly2)
mesh = Meshes(verts=[verts], faces=[faces], textures=None)
mesh2 = Meshes(verts=[verts2], faces=[faces2], textures=None)
# print(mesh)
# mesh.update_meshes_lists(mesh, 0.5)
# mesh.transparent = False
# a = sigmoid_alpha_blend(mesh, mesh2, 0.5)
fig2 = plot_scene({
           "whiteMatterFA_smooth_mask": {
               "mesh": mesh,
           }, 

           "whiteMatterFA0.2_mask": {
               "mesh": mesh2,
           }

       },
       ncols=2)
# fig2.show()

# vertsb, facesb, edgesb = utils.PolyDataToTensors(bbb)
# mesh = Meshes(verts=[vertsb], faces=[facesb], textures=None)






