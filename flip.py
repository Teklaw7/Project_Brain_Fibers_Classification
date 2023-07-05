from tools import utils
import vtk
import numpy as np
import torch
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter

obj = utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_139233_dg_ex.vtp")
transform = vtkTransform()
transform.RotateZ(180)
transformFilter = vtkTransformPolyDataFilter()
transformFilter.SetInputData(obj)
transformFilter.SetTransform(transform)
transformFilter.Update()
obj = transformFilter.GetOutput()
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("/CMF/data/timtey/tractography/all/tractogram_deterministic_139233_dg_ex_flip.vtp")
writer.SetInputData(obj)
writer.Write()
# writer = vtk.vtkPolyDataWriter()
# writer.SetFileName("/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg_flip.vtk")
# writer.SetInputData(obj)
# writer.Write()
