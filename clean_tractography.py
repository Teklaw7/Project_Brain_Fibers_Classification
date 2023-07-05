from tools import utils
import vtk
import numpy as np
import torch
path = "/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg.vtp"
# path = "/CMF/data/timtey/tracts/archives/102008_tracts/T_CC1.vtp"
# path2 = "/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg_clean.vtk"

tracts = utils.ReadSurf(path)
# tracts2 = utils.ReadSurf(path2)
# nb_points = tracts.GetNumberOfPoints()
# print(nb_points)
nb_cells = tracts.GetNumberOfCells()
print(nb_cells)
# nb_cells2 = tracts2.GetNumberOfCells()
# print(nb_cells2)

for i in range(nb_cells):
    a = tracts.GetCell(i).GetNumberOfPoints()
    # print(a)
    if a  < 2:
        tracts.DeleteCell(i)
        # print(tracts.GetNumberOfCells())
tracts.RemoveDeletedCells()
print(tracts.GetNumberOfCells())
writer = vtk.vtkPolyDataWriter()
# writer.setFileVersion(1)
writer.SetFileName("/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg_clean.vtk")
writer.SetInputData(tracts)
writer.Write()
# vtk.vtkPolyDataWriter().SetFileName("/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg_clean.vtp").SetInputData(tracts).Write()
print("Final")
print(tracts.GetNumberOfCells())
