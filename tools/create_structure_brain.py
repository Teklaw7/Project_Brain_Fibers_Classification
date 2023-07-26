import numpy as np
from dipy.io.image import load_nifti, load_nifti_data
import vtk
from library import utils_lib
# Create the data we want to contour

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
utils_lib.WriteSurf(poly, '/CMF/data/timtey/tractography/training/135528/brain_mask_135528.vtk')