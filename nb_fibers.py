from tools import utils
import os


liste = os.listdir("/CMF/data/timtey/tracts/archives/102816_tracts")
L = []
for i in range(len(liste)):
    fiber = utils.ReadSurf("/CMF/data/timtey/tracts/archives/102816_tracts/"+liste[i])
    L.append(fiber.GetNumberOfCells())
print(sum(L))



path = "/CMF/data/timtey/tractography/all/tractogram_deterministic_"
Lb = []

list_id = [102008,103515,108525,113215,119833,121618,124220,124826,139233]
for j in range(len(list_id)):
    for i in range(1,5):
        # print(i)
        path_i = path + str(list_id[j]) + "_dg_flip_" + str(i) + ".vtp"
        print(path_i)
        bundle = utils.ReadSurf(path_i)
        Lb.append(bundle.GetNumberOfCells())

print(sum(Lb))
