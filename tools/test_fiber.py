import utils
import vtk
import torch
from dipy.io.streamline import save_trk, save_vtp
# print("A")

# import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import nrrd

# path_train = f"/MEDUSA_STOR/timtey/tractography/test/tractogram_deterministic_139233_dg.vtp"
# l=[]
# # print("B")
# bundle = utils.ReadSurf(path_train)
# # print("C")
# # bundle.to(torch.device("cuda:0"))
# print(bundle.GetNumberOfCells())
# for i in range(bundle.GetNumberOfCells()):
#     bundle_extract = utils.ExtractFiber(bundle,i)
#     # bundle_extract = utils.ExtractFiber(bundle,449)
#     # print(bundle_extract)
#     bundle_tf=vtk.vtkTriangleFilter()
#     bundle_tf.SetInputData(bundle_extract)
#     bundle_tf.Update()
#     bundle_extract_tf = bundle_tf.GetOutput()
#     # verts, faces, edges = utils.PolyDataToTensors(bundle_extract_tf)  
#     # print(len(verts))   
#     # print(f"{i}",verts)
#     # print(bundle_extract_tf.GetNumberOfCells())
#     min_max = bundle_extract_tf.GetBounds()
#     if min_max[0]>min_max[1] or min_max[2]>min_max[3] or min_max[4]>min_max[5]:
#         # print(f"{i} is empty")
#         l.append(i)
#         # print(bundle_extract_tf. GetBounds())
        
#     # print("cc",bundle.GetNumberOfCells())
#     # writer = vtk.vtkPolyDataWriter()
#     # writer.SetInputData(bundle)
#     # writer.SetFileName('/MEDUSA_STOR/timtey/tractography/training/tractogram_deterministic_102008_dg2.vtp')
#     # writer.Update()

#     # print(l)
#     # print(len(l))
#     # print(bundle.GetNumberOfCells())training
#     # bundle.DeleteCell(448)
#     # bundle.RemoveDeletedCells()
#     # print(bundle.GetNumberOfCells())
#     # bundle_extract = utils.ExtractFiber(bundle,448)
#     # bundle_tf=vtk.vtkTriangleFilter()
#     # bundle_tf.SetInputData(bundle_extract)
#     # bundle_tf.Update()
#     # bundle_tf = bundle_tf.GetOutput()
#     # verts, faces, edges = utils.PolyDataToTensors(bundle_tf)
#     # print(len(verts))

# print(l)
# for i in range(len(l)):
#     bundle.DeleteCell(l[i])
# bundle.RemoveDeletedCells()
# utils.WriteSurf(bundle, "/MEDUSA_STOR/timtey/tractography/test/tractogram_deterministic_139233_dg2.vtk")

# b = utils.ReadSurf("/MEDUSA_STOR/timtey/tractography/test/tractogram_deterministic_139233_dg2.vtk")
# print("coucou",b.GetNumberOfCells())
    # save_vtp(bundle, "/MEDUSA_STOR/timtey/tractography/training/tractogram_deterministic_102008_dg2.vtp")
    # b = utils.ReadSurf("/MEDUSA_STOR/timtey/tractography/training/tractogram_deterministic_102008_dg2.vtk")
    # print("coucou",b.GetNumberOfCells())
    # print(b)
# print(b)

model = torch.load("X2.pt")
# model = model[0,:,3,:,:]
print("jahgd",model.shape)
model = model[0,:,:,:]
print(model.shape)
# print(model[0].shape)
transform = T.ToPILImage()
print(len(model[0]))
for i in range(len(model[0])):
    img = transform(model[0][i])
    img.save(f"photo_X2_{i}_.png")
# img = transform(model[0])
# model.cpu()
# print(model)
# model = model.cpu().numpy()
# # model.numpy()
# for i in range(len(model)):
#     # model[i] = model[i].numpy()
#     nrrd.write(f'photo{i}.nrrd', model)

# img.show()

