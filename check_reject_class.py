import vtk
from tools import utils
import pickle

path = "/CMF/data/timtey/tractography/all/tractogram_deterministic_139233_dg_flip_1_DTI.vtp"
bundle = utils.ReadSurf(path)
number_cells = bundle.GetNumberOfCells()
print(number_cells)

# fiber = utils.ExtractFiber(bundle,10078)
# print(fiber)

open_file = open("to_reject.pkl", "rb")
L_n_lab = pickle.load(open_file)
open_file.close()

# print(L_n_lab)


print(len(L_n_lab))
print(L_n_lab.shape)
print(L_n_lab[0], L_n_lab[1])
# Lab = L_n_lab[:,-1]
# print(Lab.shape)
# Ln_infos = L_n_lab[:,:-1]
Ln_infos = L_n_lab
print(Ln_infos.shape)
# for i in range(Ln_infos.shape[0]):
#     print(Ln_infos[i][0].item())

'''
L_lab = []
for j in range(57):
    l_j_lab = []
    for i in range(Lab.shape[0]):
        lab = Lab[i].item()
        if lab == j:
            l_j_lab.append(Ln_infos[i])
    L_lab.append(l_j_lab)
print(len(L_lab))
'''


'''
L_l_i = []
for i in range(len(L_lab)):
    l_i = []
    if len(L_lab[i]) > 0:
        for j in range(len(L_lab[i])):
            # print(int(L_lab[i][j][-1].item()))
            l_i.append(int(L_lab[i][j][-1].item()))
    print(len(l_i))
    L_l_i.append(l_i)
print(L_l_i)
print(len(L_l_i))

for a in range(len(L_l_i)):
    l_fiber = []
    for b in range(len(L_l_i[a])):
        l_fiber.append(utils.ExtractFiber(bundle,L_l_i[a][b]))
    merge = utils.Merge(l_fiber)
    # print(merge)
    if merge.GetNumberOfCells() > 0:
        vtk_writer = vtk.vtkXMLPolyDataWriter()
        vtk_writer.SetFileName(f"/CMF/data/timtey/tractography/all/test_tracts_slicer_seuil_05/tractography_fibers_test_light{a}.vtp")
        vtk_writer.SetInputData(merge)
        vtk_writer.Write()
        '''



### rejection fibers

# for i in range(Ln_infos.shape[0]):
    # print(Ln_infos[i][0].item())
    # if Ln_infos[i][0].item() == 0:
        # fiber = utils.ExtractFiber(bundle, int(Ln_infos[i][1].item()))
        # vtk_writer = vtk.vtkXMLPolyDataWriter()
        # vtk_writer.SetFileName(f"/CMF/data/timtey/tractography/all/test_tracts_slicer_seuil_05/tractography_fibers_test_light_reject{i}.vtp")

# Ln_infos = Ln_infos[::10,:]
print(Ln_infos.shape)

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
LN = chunkIt(Ln_infos, 10)
print(len(LN))
# print(LN)
list_fiber_reject = []


for j in range(len(LN)):
    l_fiber = []
    for i in range(len(LN[j])):
        fiber = utils.ExtractFiber(bundle, int(LN[j][i][-1].item()))
        l_fiber.append(fiber)
    print(len(l_fiber))
    merge = utils.Merge(l_fiber)
    # if merge.GetNumberOfCells() > 0:
    #     vtk_writer = vtk.vtkXMLPolyDataWriter()
    #     vtk_writer.SetFileName(f"/CMF/data/timtey/tractography/all/test_tracts_slicer_reject/tractography_fibers_test_light_reject{j}.vtp")
    #     vtk_writer.SetInputData(merge)
    #     vtk_writer.Write()

# for i in range(Ln_infos.shape[0]):
#     fiber = utils.ExtractFiber(bundle, int(Ln_infos[i][-1].item()))
#     list_fiber_reject.append(fiber)
# vtk_writer = vtk.vtkXMLPolyDataWriter()
# vtk_writer.SetFileName(f"/CMF/data/timtey/tractography/all/test_tracts_slicer_reject/tractography_fibers_test_light_reject.vtp")
# vtk_writer.SetInputData(utils.Merge(list_fiber_reject))
# vtk_writer.Write()