from tools import utils
import vtk
import numpy as np
import torch
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from tools import utils
obj = utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg_flip.vtp")


nb_cells = obj.GetNumberOfCells()

obj1 = vtk.vtkPolyData()
obj2 = vtk.vtkPolyData()
cell = vtk.vtkCellArray()
Points_ids = []
cell_2 = vtk.vtkCellArray()
Points_ids2 = []

list_id_1 = []
list_id_2 = []
list_id_3 = []
list_id_4 = []
# nb_cells=2000
for i in range(nb_cells):
    if i < int(nb_cells/4):
        list_id_1.append(i)
    elif int(nb_cells/4) <= i < int(nb_cells/2):
        list_id_2.append(i)
    elif int(nb_cells/2) <= i < int(nb_cells/4*3):
        list_id_3.append(i)
    else:
        list_id_4.append(i)

L_tube_1 = utils.ExtractPart(obj, list_id_1)
# m1 = utils.Merge(L_tube_1)
# print(m1)
print("extract_part_done")
print(L_tube_1)
# print(ksdjhdksjgh)
# print("L_tube_1",len(L_tube_1))
L_tube_2 = utils.ExtractPart(obj, list_id_2)
print("extract_part_done")
print(L_tube_2)
# print("L_tube_2",len(L_tube_2))
L_tube_3 = utils.ExtractPart(obj, list_id_3)
print("extract_part_done")
print(L_tube_3)
# print("L_tube_3",len(L_tube_3))
L_tube_4 = utils.ExtractPart(obj, list_id_4)
print("extract_part_done")
print(L_tube_4)

# print("L_tube_1", L_tube_1)
# print("L_tube_2", L_tube_2)
# print("L_tube_3", L_tube_3)
# print("L_tube_4", L_tube_4)


# m1 = utils.Merge(L_tube_1)
# print("Merge_done")
# m2 = utils.Merge(L_tube_2)
# print("Merge_done")
# m3 = utils.Merge(L_tube_3)
# print("Merge_done")
# m4 = utils.Merge(L_tube_4)
# print("Merge_done")

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg_flip_1.vtp")
writer.SetInputData(L_tube_1)
writer.Write()
print("done")
writer2 = vtk.vtkXMLPolyDataWriter()
writer2.SetFileName("/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg_flip_2.vtp")
writer2.SetInputData(L_tube_2)
writer2.Write()
print("done")
writer3 = vtk.vtkXMLPolyDataWriter()
writer3.SetFileName("/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg_flip_3.vtp")
writer3.SetInputData(L_tube_3)
writer3.Write()
print("done")
writer4 = vtk.vtkXMLPolyDataWriter()
writer4.SetFileName("/CMF/data/timtey/tractography/all/tractogram_deterministic_102008_dg_flip_4.vtp")
writer4.SetInputData(L_tube_4)    
writer4.Write()


