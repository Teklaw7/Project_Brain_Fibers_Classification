
from pytorch_lightning.callbacks import Callback
import torchvision
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import pytorch3d.transforms as T3d



class RotationTransform:
    def __call__(self, verts, rotation_matrix):
        b = torch.transpose(verts,0,1)
        # print("b", b.shape)
        a= torch.mm(rotation_matrix,torch.transpose(verts,0,1))
        verts = torch.transpose(torch.mm(rotation_matrix,torch.transpose(verts,0,1)),0,1)
        return verts

class RandomRotationTransform:
    def __call__(self, verts):
        rotation_matrix = T3d.random_rotation()
        rotation_transform = RotationTransform()
        verts = rotation_transform(verts,rotation_matrix)
        return verts

def transformation_verts_by_fiber(verts, verts_fiber_bounds):
    for i in range (verts.shape[0]):
        verts[i,:,0] = (0.8*(verts[i,:,0] - verts_fiber_bounds[i][0])/(verts_fiber_bounds[i][1] - verts_fiber_bounds[i][0])) - 0.4
        verts[i,:,1] = (0.8*(verts[i,:,1] - verts_fiber_bounds[i][2])/(verts_fiber_bounds[i][3] - verts_fiber_bounds[i][2])) - 0.4
        verts[i,:,2] = (0.8*(verts[i,:,2] - verts_fiber_bounds[i][4])/(verts_fiber_bounds[i][5] - verts_fiber_bounds[i][4])) - 0.4
    return verts

def transformation_verts(verts, sample_min_max):
    for i in range (verts.shape[0]):
        verts[i,:,0] = ((verts[i,:,0] - sample_min_max[i][0])/(sample_min_max[i][1] - sample_min_max[i][0])) - 0.5
        verts[i,:,1] = ((verts[i,:,1] - sample_min_max[i][2])/(sample_min_max[i][3] - sample_min_max[i][2])) - 0.5
        verts[i,:,2] = ((verts[i,:,2] - sample_min_max[i][4])/(sample_min_max[i][5] - sample_min_max[i][4])) - 0.5
    return verts


def randomrotation(verts):
    verts_device = verts.get_device()
    rotation_matrix = T3d.random_rotation().to(verts_device)
    rotation_transform = RotationTransform()
    # print("verts", verts.shape)
    # print("rotation_matrix", rotation_matrix.shape)
    # print("rotation_matrix", type(rotation_matrix))
    verts = rotation_transform(verts,rotation_matrix)
    return verts

def pad_double_batch(V,F,FF,VFI,FFI,FFFI, V2,F2,FF2,VFI2,FFI2,FFFI2):
    # V_max = max(V.shape[1],V2.shape[1])
    # i_V_max = (V.shape[1], V2.shape[1]).index(max(V.shape[1], V2.shape[1]))
    delta_V = V.shape[1] - V2.shape[1]
    if delta_V > 0:
        V2 = torch.cat((V2, torch.zeros(V2.shape[0],delta_V,3).to(V2.device)), dim=1)
        V_c = torch.cat((V,V2), dim=0)
    elif delta_V < 0:
        V = torch.cat((V, torch.zeros(V.shape[0],-delta_V,3).to(V.device)), dim=1)
        V_c = torch.cat((V2,V), dim=0)
    elif delta_V == 0:
        V_c = torch.cat((V,V2), dim=0)
    # V_c = torch.cat((V,V2), dim=0)
    delta_F = F.shape[1] - F2.shape[1]
    if delta_F > 0:
        F2 = torch.cat((F2, -torch.ones(F2.shape[0],delta_F,3).to(F2.device)), dim=1)
        F_c = torch.cat((F,F2), dim=0)
    elif delta_F < 0:
        F = torch.cat((F, -torch.ones(F.shape[0],-delta_F,3).to(F.device)), dim=1)
        F_c = torch.cat((F2,F), dim=0)
    elif delta_F == 0:
        F_c = torch.cat((F,F2), dim=0)
    # F_c = torch.cat((F,F2), dim=0)
    delta_FF = FF.shape[1] - FF2.shape[1]
    if delta_FF > 0:
        FF2 = torch.cat((FF2, torch.zeros(FF2.shape[0],delta_FF,3).to(FF2.device)), dim=1)
        FF_c = torch.cat((FF,FF2), dim=0)
    elif delta_FF < 0:
        FF = torch.cat((FF, torch.zeros(FF.shape[0],-delta_FF,3).to(FF.device)), dim=1)
        FF_c = torch.cat((FF2,FF), dim=0)
    elif delta_FF == 0:
        FF_c = torch.cat((FF,FF2), dim=0)
    # FF_c = torch.cat((FF,FF2), dim=0)
    delta_VFI = VFI.shape[1] - VFI2.shape[1]
    if delta_VFI > 0:
        VFI2 = torch.cat((VFI2, torch.zeros(VFI2.shape[0],delta_VFI,3).to(VFI2.device)), dim=1)
        VFI_c = torch.cat((VFI,VFI2), dim=0)
    elif delta_VFI < 0:
        VFI = torch.cat((VFI, torch.zeros(VFI.shape[0],-delta_VFI,3).to(VFI.device)), dim=1)
        VFI_c = torch.cat((VFI2,VFI), dim=0)
    elif delta_VFI == 0:
        VFI_c = torch.cat((VFI,VFI2), dim=0)
    # VFI_c = torch.cat((VFI,VFI2), dim=0)
    delta_FFI = FFI.shape[1] - FFI2.shape[1]
    if delta_FFI > 0:
        FFI2 = torch.cat((FFI2, -torch.ones(FFI2.shape[0],delta_FFI,3).to(FFI2.device)), dim=1)
        FFI_c = torch.cat((FFI,FFI2), dim=0)
    elif delta_FFI < 0:
        FFI = torch.cat((FFI, -torch.ones(FFI.shape[0],-delta_FFI,3).to(FFI.device)), dim=1)
        FFI_c = torch.cat((FFI2,FFI), dim=0)
    elif delta_FFI == 0:
        FFI_c = torch.cat((FFI,FFI2), dim=0)
    # FFI_c = torch.cat((FFI,FFI2), dim=0)
    delta_FFFI = FFFI.shape[1] - FFFI2.shape[1]
    if delta_FFFI > 0:
        FFFI2 = torch.cat((FFFI2, torch.zeros(FFFI2.shape[0],delta_FFFI,3).to(FFFI2.device)), dim=1)
        FFFI_c = torch.cat((FFFI,FFFI2), dim=0)
    elif delta_FFFI < 0:
        FFFI = torch.cat((FFFI, torch.zeros(FFFI.shape[0],-delta_FFFI,3).to(FFFI.device)), dim=1)
        FFFI_c = torch.cat((FFFI2,FFFI), dim=0)
    elif delta_FFFI == 0:
        FFFI_c = torch.cat((FFFI,FFFI2), dim=0)
    # FFFI_c = torch.cat((FFFI,FFFI2), dim=0)
    # delta_VB = VB.shape[1] - VB2.shape[1]
    # if delta_VB > 0:
    #     VB2 = torch.cat((VB2, torch.zeros(VB2.shape[0],delta_VB,3).to(VB2.device)), dim=1)
    #     VB_c = torch.cat((VB,VB2), dim=0)
    # elif delta_VB < 0:
    #     VB = torch.cat((VB, torch.zeros(VB.shape[0],-delta_VB,3).to(VB.device)), dim=1)
    #     VB_c = torch.cat((VB2,VB), dim=0)
    # elif delta_VB == 0:
    #     VB_c = torch.cat((VB,VB2), dim=0)
    # # VB_c = torch.cat((VB,VB2), dim=0)
    # delta_FB = FB.shape[1] - FB2.shape[1]
    # if delta_FB > 0:
    #     FB2 = torch.cat((FB2, -torch.ones(FB2.shape[0],delta_FB,3).to(FB2.device)), dim=1)
    #     FB_c = torch.cat((FB,FB2), dim=0)
    # elif delta_FB < 0:
    #     FB = torch.cat((FB, -torch.ones(FB.shape[0],-delta_FB,3).to(FB.device)), dim=1)
    #     FB_c = torch.cat((FB2,FB), dim=0)
    # elif delta_FB == 0:
    #     FB_c = torch.cat((FB,FB2), dim=0)
    # # FB_c = torch.cat((FB,FB2), dim=0)
    # delta_FFB = FFB.shape[1] - FFB2.shape[1]
    # if delta_FFB > 0:
    #     FFB2 = torch.cat((FFB2, torch.zeros(FFB2.shape[0],delta_FFB,3).to(FFB2.device)), dim=1)
    #     FFB_c = torch.cat((FFB,FFB2), dim=0)
    # elif delta_FFB < 0:
    #     FFB = torch.cat((FFB, torch.zeros(FFB.shape[0],-delta_FFB,3).to(FFB.device)), dim=1)
    #     FFB_c = torch.cat((FFB2,FFB), dim=0)
    # elif delta_FFB == 0:
    #     FFB_c = torch.cat((FFB,FFB2), dim=0)
    # FFB_c = torch.cat((FFB,FFB2), dim=0)
    # return V_c,F_c,FF_c,VFI_c,FFI_c,FFFI_c,VB_c,FB_c,FFB_c
    return V_c,F_c,FF_c,VFI_c,FFI_c,FFFI_c


class BrainNetImageLogger_contrastive_tractography_labeled(Callback):
    def __init__(self,num_features = 3 , num_images=12, log_steps=10,mean=0,std=0.015):
        self.num_features = num_features
        self.log_steps = log_steps
        self.num_images = num_images
        self.mean = mean
        self.std = std

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

            V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max, data_lab, name_labels = batch
            # V2, F2, FF2, labels2, VFI2, FFI2, FFFI2, labelsFI2, VB2, FB2, FFB2, vfbounds2, sample_min_max2 = batch[1]
            # print("on train batch",batch.shape)
            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
            # VF = VF.to(pl_module.device,non_blocking=True)
            FF = FF.to(pl_module.device,non_blocking=True)
            VFI = VFI.to(pl_module.device,non_blocking=True)
            FFI = FFI.to(pl_module.device,non_blocking=True)
            FFFI = FFFI.to(pl_module.device,non_blocking=True)
            VB = VB.to(pl_module.device,non_blocking=True)
            FB = FB.to(pl_module.device,non_blocking=True)
            FFB = FFB.to(pl_module.device,non_blocking=True)
            # V2 = V2.to(pl_module.device,non_blocking=True)
            # F2 = F2.to(pl_module.device,non_blocking=True)
            # VF = VF.to(pl_module.device,non_blocking=True)
            # FF2 = FF2.to(pl_module.device,non_blocking=True)
            # VFI2 = VFI2.to(pl_module.device,non_blocking=True)
            # FFI2 = FFI2.to(pl_module.device,non_blocking=True)
            # FFFI2 = FFFI2.to(pl_module.device,non_blocking=True)
            # VB2 = VB2.to(pl_module.device,non_blocking=True)
            # FB2 = FB2.to(pl_module.device,non_blocking=True)
            # FFB2 = FFB2.to(pl_module.device,non_blocking=True)
            # V_c,F_c,FF_c, VFI_c, FFI_c, FFFI_c = pad_double_batch(V,F,FF,VFI,FFI,FFFI, V2,F2,FF2,VFI2,FFI2,FFFI2)
            
            # V_c1 = V_c +torch.normal(0, 0.03, size=V_c.shape).to(pl_module.device, non_blocking=True)
            # V_c2 = V_c +torch.normal(0, 0.03, size=V_c.shape).to(pl_module.device, non_blocking=True)
            # VFI_c1 = VFI_c +torch.normal(0, 0.03, size=VFI_c.shape).to(pl_module.device, non_blocking=True)
            # VFI_c2 = VFI_c +torch.normal(0, 0.03, size=VFI_c.shape).to(pl_module.device, non_blocking=True)

            # V_c1 = V_c1.to(pl_module.device,non_blocking=True)
            # V_c2 = V_c2.to(pl_module.device,non_blocking=True)
            # VFI_c1 = VFI_c1.to(pl_module.device,non_blocking=True)
            # VFI_c2 = VFI_c2.to(pl_module.device,non_blocking=True)

            # for i in range(V_c1.shape[0]):
                # V_c1[i] = randomrotation(V_c1[i])
                # V_c2[i] = randomrotation(V_c2[i])
                # VFI_c1[i] = randomrotation(VFI_c1[i])
                # VFI_c2[i] = randomrotation(VFI_c2[i])

            V1 = V +torch.normal(0, 0.03, size=V.shape).to(pl_module.device, non_blocking=True)
            V2 = V +torch.normal(0, 0.03, size=V.shape).to(pl_module.device, non_blocking=True)
            VFI1 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(pl_module.device, non_blocking=True)
            VFI2 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(pl_module.device, non_blocking=True)
            for i in range(0, V.shape[0]):
                V1[i] = randomrotation(V1[i])
                V2[i] = randomrotation(V2[i])
                VFI1[i] = randomrotation(VFI1[i])
                VFI2[i] = randomrotation(VFI2[i])

            with torch.no_grad():

                images, PF = pl_module.render(V, F, FF)  
                images1, PF1 = pl_module.render(V1, F, FF)
                images2, PF2 = pl_module.render(V2, F, FF)

                # s=1
                # color_jitter = T.ColorJitter(
                #     0.5 * s, 0.5 * s, 0.5 * s, 0.1 * s
                # )
                # blur = T.GaussianBlur((3, 3), (0.1, 2.0))
                # augmentation = T.Compose([
                #     # T.RandomResizedCrop(size=224),
                #     # T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
                #     T.RandomApply([color_jitter], p=0.8),
                #     T.CenterCrop(size=224),
                #     # T.RandomInvert(p=0.5),
                #     # T.RandomApply([blur], p=0.5),
                #     # T.Grayscale(num_output_channels=1)
                # ])
                # X1 = torch.tensor([]).to(pl_module.device)
                # X2 = torch.tensor([]).to(pl_module.device)
                # for i in range(0, images.shape[2]):
                #     x1 = augmentation(images[:,:,i:i+1,:,:])
                #     x2 = augmentation(images[:,:,i:i+1,:,:])
                #     x1 = T.RandomHorizontalFlip(p=0.5)(x1)
                #     x1 = T.RandomVerticalFlip(p=0.5)(x1)
                #     x2 = T.RandomHorizontalFlip(p=0.5)(x2)
                #     x2 = T.RandomVerticalFlip(p=0.5)(x2)
                #     # x1 = F.adjust_brightness(x1, 0.5)
                #     # x2 = F.adjust_brightness(x2, 0.5)
                #     X1 = torch.cat((X1, x1), dim=2)
                #     X2 = torch.cat((X2, x2), dim=2)   
                images_fiber, PF_fiber = pl_module.render(VFI, FFI, FFFI)
                images_fiber_1, PF_fiber_1 = pl_module.render(VFI1, FFI, FFFI)
                images_fiber_2, PF_fiber_2 = pl_module.render(VFI2, FFI, FFFI)
                # X1_fiber = torch.tensor([]).to(pl_module.device)
                # X2_fiber = torch.tensor([]).to(pl_module.device)
                # for i in range(0, images_fiber.shape[2]):
                #     x1_fiber = augmentation(images_fiber[:,:,i:i+1,:,:])
                #     x2_fiber = augmentation(images_fiber[:,:,i:i+1,:,:])
                #     x1_fiber = T.RandomHorizontalFlip(p=0.5)(x1_fiber)
                #     x1_fiber = T.RandomVerticalFlip(p=0.5)(x1_fiber)
                #     x2_fiber = T.RandomHorizontalFlip(p=0.5)(x2_fiber)
                #     x2_fiber = T.RandomVerticalFlip(p=0.5)(x2_fiber)
                #     X1_fiber = torch.cat((X1_fiber, x1_fiber), dim=2)
                #     X2_fiber = torch.cat((X2_fiber, x2_fiber), dim=2)

                # images_brain, PF_brain = pl_module.render(VB, FB, FFB)
                # X1_brain = torch.tensor([]).to(pl_module.device)
                # X2_brain = torch.tensor([]).to(pl_module.device)
                # for i in range(0, images_brain.shape[2]):
                #     x1_brain = augmentation(images_brain[:,:,i:i+1,:,:])
                #     x2_brain = augmentation(images_brain[:,:,i:i+1,:,:])
                #     x1_brain = T.RandomHorizontalFlip(p=0.5)(x1_brain)
                #     x1_brain = T.RandomVerticalFlip(p=0.5)(x1_brain)
                #     x2_brain = T.RandomHorizontalFlip(p=0.5)(x2_brain)
                #     x2_brain = T.RandomVerticalFlip(p=0.5)(x2_brain)
                #     X1_brain = torch.cat((X1_brain, x1_brain), dim=2)
                #     X2_brain = torch.cat((X2_brain, x2_brain), dim=2)
                images = torch.cat((images, images_fiber, images1, images2, images_fiber_1, images_fiber_2), dim=1)
                grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image train', grid_images, pl_module.global_step)
                grid_images_const = torchvision.utils.make_grid(images[0, self.num_images:, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image train const', grid_images_const, pl_module.global_step)
                # grid_images_const = torchvision.utils.make_grid(images[0, self.num_images:, 0:self.num_features, :, :])
                # trainer.logger.experiment.add_image('Image train const', grid_images_const, pl_module.global_step)
                # grid_images_const_fiber

                # images_noiseM = pl_module.noise(images)
# 
                # grid_images_noiseM = torchvision.utils.make_grid(images_noiseM[0, 0:self.num_images, 0:self.num_features, :, :])
                # trainer.logger.experiment.add_image('Image + noise M ', grid_images_noiseM, pl_module.global_step)

                # grid_eacsf = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:1, :, :])
                # trainer.logger.experiment.add_image('Image eacsf', grid_eacsf, pl_module.global_step)

                # grid_sa = torchvision.utils.make_grid(images[0, 0:self.num_images, 1:2, :, :])
                # trainer.logger.experiment.add_image('Image sa', grid_sa, pl_module.global_step)

                # grid_thickness = torchvision.utils.make_grid(images[0, 0:self.num_images, 1:2, :, :])
                # trainer.logger.experiment.add_image('Image thickness', grid_thickness, pl_module.global_step)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):        

        if batch_idx % self.log_steps == 0:

            V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max, data_lab, name_labels = batch
            # V2, F2, FF2, labels2, VFI2, FFI2, FFFI2, labelsFI2, VB2, FB2, FFB2, vfbounds2, sample_min_max2 = batch[1]
            # print("on val batch",batch.shape)
            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
            # VF = VF.to(pl_module.device,non_blocking=True)
            FF = FF.to(pl_module.device,non_blocking=True)
            VFI = VFI.to(pl_module.device,non_blocking=True)
            FFI = FFI.to(pl_module.device,non_blocking=True)
            FFFI = FFFI.to(pl_module.device,non_blocking=True)
            VB = VB.to(pl_module.device,non_blocking=True)
            FB = FB.to(pl_module.device,non_blocking=True)
            FFB = FFB.to(pl_module.device,non_blocking=True)
            # V2 = V2.to(pl_module.device,non_blocking=True)
            # F2 = F2.to(pl_module.device,non_blocking=True)
            # VF2 = VF2.to(pl_module.device,non_blocking=True)
            # FF2 = FF2.to(pl_module.device,non_blocking=True)
            # VFI2 = VFI2.to(pl_module.device,non_blocking=True)
            # FFI2 = FFI2.to(pl_module.device,non_blocking=True)
            # FFFI2 = FFFI2.to(pl_module.device,non_blocking=True)
            # VB2 = VB2.to(pl_module.device,non_blocking=True)
            # FB2 = FB2.to(pl_module.device,non_blocking=True)
            # FFB2 = FFB2.to(pl_module.device,non_blocking=True)
            # V = transformation_verts(V, sample_min_max)
            # V2 = transformation_verts(V2, sample_min_max2)
            # VFI = transformation_verts_by_fiber(VFI, vfbounds)
            # VFI2 = transformation_verts_by_fiber(VFI2, vfbounds2)
            # V_c,F_c,FF_c, VFI_c, FFI_c, FFFI_c = pad_double_batch(V,F,FF,VFI,FFI,FFFI, V2,F2,FF2,VFI2,FFI2,FFFI2)
            # a = torch.normal(mean=0, std=0.003, size=(10,5,2)).to(pl_module.device,non_blocking=True)
            # b = torch.normal(mean=0, std=0.003, size=(10,5,2)).to(pl_module.device,non_blocking=True)

            # print(a,b)
            # V_c1 = V_c +torch.normal(0, 0.03, size=V_c.shape).to(pl_module.device,non_blocking=True)
            # V_c2 = V_c +torch.normal(0, 0.03, size=V_c.shape).to(pl_module.device,non_blocking=True)
            # VFI_c1 = VFI_c +torch.normal(0, 0.03, size=VFI_c.shape).to(pl_module.device,non_blocking=True)
            # VFI_c2 = VFI_c +torch.normal(0, 0.03, size=VFI_c.shape).to(pl_module.device,non_blocking=True)
            # V_c1 = V_c1.to(pl_module.device,non_blocking=True)
            # V_c2 = V_c2.to(pl_module.device,non_blocking=True)
            # VFI_c1 = VFI_c1.to(pl_module.device,non_blocking=True)
            # VFI_c2 = VFI_c2.to(pl_module.device,non_blocking=True)

            # for i in range(V_c1.shape[0]):
            #     V_c1[i] = randomrotation(V_c1[i])
            #     V_c2[i] = randomrotation(V_c2[i])
            #     VFI_c1[i] = randomrotation(VFI_c1[i])
            #     VFI_c2[i] = randomrotation(VFI_c2[i])
                # 
            V1 = V + torch.normal(mean=0, std=0.003, size=V.shape).to(pl_module.device,non_blocking=True)
            V2 = V + torch.normal(mean=0, std=0.003, size=V.shape).to(pl_module.device,non_blocking=True)
            VFI1 = VFI + torch.normal(mean=0, std=0.003, size=VFI.shape).to(pl_module.device,non_blocking=True)
            VFI2 = VFI + torch.normal(mean=0, std=0.003, size=VFI.shape).to(pl_module.device,non_blocking=True)
            for i in range(0, V.shape[0]):
                V1[i] = randomrotation(V1[i])
                V2[i] = randomrotation(V2[i])
                VFI1[i] = randomrotation(VFI1[i])
                VFI2[i] = randomrotation(VFI2[i])
            with torch.no_grad():
                # print("F_c",F_c)
                # print("F",F.shape)
                # print("F2",F2.shape)
                images, PF = pl_module.render(V, F, FF)
                images1, PF1 = pl_module.render(V1, F, FF)
                images2, PF2 = pl_module.render(V2, F, FF)
                images_fiber, PF_fiber = pl_module.render(VFI, FFI, FFFI) 
                images_fiber_1, PF_fiber_1 = pl_module.render(VFI1, FFI, FFFI)
                images_fiber_2, PF_fiber_2 = pl_module.render(VFI2, FFI, FFFI)
                images_brain, PF_brain = pl_module.render(VB, FB, FFB)   
                # print("images", images.shape)
                # print("images_fiber", images_fiber.shape)
                # print("images_brain", images_brain.shape)
                # print("images1", images1.shape)
                # print("images2", images2.shape)
                # print("images_fiber_1", images_fiber_1.shape)
                # print("images_fiber_2", images_fiber_2.shape)
                images = torch.cat((images, images_fiber, images1, images2, images_fiber_1, images_fiber_2), dim=1)
                # print("images", images.shape)
                grid_images = torchvision.utils.make_grid(images[5, 0:self.num_images, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image val', grid_images, pl_module.global_step)
                grid_images_const = torchvision.utils.make_grid(images[5, self.num_images:48, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image val const', grid_images_const, pl_module.global_step)
                grid_images_const_fiber = torchvision.utils.make_grid(images[5, 48:, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image val const fiber', grid_images_const_fiber, pl_module.global_step)

                # images_noiseM = pl_module.noise(images)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):        

        if batch_idx % self.log_steps == 0:
            # print("lenght of batch", len(batch), len(batch[0]), len(batch[1]))
            V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max, data_lab, name_labels = batch
            # V2, F2, FF2, labels2, VFI2, FFI2, FFFI2, labelsFI2, VB2, FB2, FFB2, vfbounds2, sample_min_max2 = batch[1]
            
            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
            # VF = VF.to(pl_module.device,non_blocking=True)
            FF = FF.to(pl_module.device,non_blocking=True)
            VFI = VFI.to(pl_module.device,non_blocking=True)
            FFI = FFI.to(pl_module.device,non_blocking=True)
            FFFI = FFFI.to(pl_module.device,non_blocking=True)
            VB = VB.to(pl_module.device,non_blocking=True)
            FB = FB.to(pl_module.device,non_blocking=True)
            FFB = FFB.to(pl_module.device,non_blocking=True)

            # V2 = V2.to(pl_module.device,non_blocking=True)
            # F2 = F2.to(pl_module.device,non_blocking=True)
            # VF = VF.to(pl_module.device,non_blocking=True)
            # FF2 = FF2.to(pl_module.device,non_blocking=True)
            # VFI2 = VFI2.to(pl_module.device,non_blocking=True)
            # FFI2 = FFI2.to(pl_module.device,non_blocking=True)
            # FFFI2 = FFFI2.to(pl_module.device,non_blocking=True)
            # VB2 = VB2.to(pl_module.device,non_blocking=True)
            # FB2 = FB2.to(pl_module.device,non_blocking=True)
            # FFB2 = FFB2.to(pl_module.device,non_blocking=True)
            # V_c,F_c,FF_c, VFI_c, FFI_c, FFFI_c = pad_double_batch(V,F,FF,VFI,FFI,FFFI, V2,F2,FF2,VFI2,FFI2,FFFI2)

            # V_c1 = V_c +torch.normal(0, 0.03, size=V_c.shape).to(pl_module.device,non_blocking=True)
            # V_c2 = V_c +torch.normal(0, 0.03, size=V_c.shape).to(pl_module.device,non_blocking=True)
            # VFI_c1 = VFI_c +torch.normal(0, 0.03, size=VFI_c.shape).to(pl_module.device,non_blocking=True)
            # VFI_c2 = VFI_c +torch.normal(0, 0.03, size=VFI_c.shape).to(pl_module.device,non_blocking=True)
            # V_c1 = V_c1.to(pl_module.device,non_blocking=True)
            # V_c2 = V_c2.to(pl_module.device,non_blocking=True)
            # VFI_c1 = VFI_c1.to(pl_module.device,non_blocking=True)
            # VFI_c2 = VFI_c2.to(pl_module.device,non_blocking=True)
            # for i in range(V_c1.shape[0]):
            #     V_c1[i] = randomrotation(V_c1[i])
            #     V_c2[i] = randomrotation(V_c2[i])
            #     VFI_c1[i] = randomrotation(VFI_c1[i])
            #     VFI_c2[i] = randomrotation(VFI_c2[i])
            V1 = V + torch.normal(mean=0, std=0.003, size=V.shape).to(pl_module.device,non_blocking=True)
            V2 = V + torch.normal(mean=0, std=0.003, size=V.shape).to(pl_module.device,non_blocking=True)
            VFI1 = VFI + torch.normal(mean=0, std=0.003, size=VFI.shape).to(pl_module.device,non_blocking=True)
            VFI2 = VFI + torch.normal(mean=0, std=0.003, size=VFI.shape).to(pl_module.device,non_blocking=True)
            for i in range(0, V.shape[0]):
                V1[i] = randomrotation(V1[i])
                V2[i] = randomrotation(V2[i])
                VFI1[i] = randomrotation(VFI1[i])
                VFI2[i] = randomrotation(VFI2[i])

            with torch.no_grad():

                images, PF = pl_module.render(V, F, FF)       
                images1, PF1 = pl_module.render(V1, F, FF)
                images2, PF2 = pl_module.render(V2, F, FF)             
                images_fiber, PF_fiber = pl_module.render(VFI, FFI, FFFI)
                images_fiber_1, PF_fiber_1 = pl_module.render(VFI1, FFI, FFFI)
                images_fiber_2, PF_fiber_2 = pl_module.render(VFI2, FFI, FFFI)
                images_brain, PF_brain = pl_module.render(VB, FB, FFB)
                images = torch.cat((images, images_fiber, images1, images2, images_fiber_1, images_fiber_2), dim=1)
                grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image test', grid_images, pl_module.global_step)
                grid_images_const = torchvision.utils.make_grid(images[0, self.num_images:, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image test const', grid_images_const, pl_module.global_step)

                # images_noiseM = pl_module.noise(images)

class BrainNetImageLogger(Callback):
    def __init__(self,num_features = 3 , num_images=12, log_steps=10,mean=0,std=0.015):
        self.num_features = num_features
        self.log_steps = log_steps
        self.num_images = num_images
        self.mean = mean
        self.std = std

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

            V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = batch
            # print("on train batch",batch.shape)
            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
            # VF = VF.to(pl_module.device,non_blocking=True)
            FF = FF.to(pl_module.device,non_blocking=True)
            VFI = VFI.to(pl_module.device,non_blocking=True)
            FFI = FFI.to(pl_module.device,non_blocking=True)
            FFFI = FFFI.to(pl_module.device,non_blocking=True)
            VB = VB.to(pl_module.device,non_blocking=True)
            FB = FB.to(pl_module.device,non_blocking=True)
            FFB = FFB.to(pl_module.device,non_blocking=True)

            with torch.no_grad():

                images, PF = pl_module.render(V, F, FF)                    
                images_fiber, PF_fiber = pl_module.render(VFI, FFI, FFFI)
                images_brain, PF_brain = pl_module.render(VB, FB, FFB)
                images = torch.cat((images, images_fiber, images_brain), dim=1)
                grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image train', grid_images, pl_module.global_step)

                # images_noiseM = pl_module.noise(images)
# 
                # grid_images_noiseM = torchvision.utils.make_grid(images_noiseM[0, 0:self.num_images, 0:self.num_features, :, :])
                # trainer.logger.experiment.add_image('Image + noise M ', grid_images_noiseM, pl_module.global_step)

                # grid_eacsf = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:1, :, :])
                # trainer.logger.experiment.add_image('Image eacsf', grid_eacsf, pl_module.global_step)

                # grid_sa = torchvision.utils.make_grid(images[0, 0:self.num_images, 1:2, :, :])
                # trainer.logger.experiment.add_image('Image sa', grid_sa, pl_module.global_step)

                # grid_thickness = torchvision.utils.make_grid(images[0, 0:self.num_images, 1:2, :, :])
                # trainer.logger.experiment.add_image('Image thickness', grid_thickness, pl_module.global_step)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):        

        if batch_idx % self.log_steps == 0:

            V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = batch
            # print("on val batch",batch.shape)
            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
            # VF = VF.to(pl_module.device,non_blocking=True)
            FF = FF.to(pl_module.device,non_blocking=True)
            VFI = VFI.to(pl_module.device,non_blocking=True)
            FFI = FFI.to(pl_module.device,non_blocking=True)
            FFFI = FFFI.to(pl_module.device,non_blocking=True)
            VB = VB.to(pl_module.device,non_blocking=True)
            FB = FB.to(pl_module.device,non_blocking=True)
            FFB = FFB.to(pl_module.device,non_blocking=True)

            with torch.no_grad():

                images, PF = pl_module.render(V, F, FF)
                images_fiber, PF_fiber = pl_module.render(VFI, FFI, FFFI) 
                images_brain, PF_brain = pl_module.render(VB, FB, FFB)   
                images = torch.cat((images, images_fiber, images_brain), dim=1)
                grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image val', grid_images, pl_module.global_step)

                # images_noiseM = pl_module.noise(images)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):        

        if batch_idx % self.log_steps == 0:

            V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = batch

            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
            # VF = VF.to(pl_module.device,non_blocking=True)
            FF = FF.to(pl_module.device,non_blocking=True)
            VFI = VFI.to(pl_module.device,non_blocking=True)
            FFI = FFI.to(pl_module.device,non_blocking=True)
            FFFI = FFFI.to(pl_module.device,non_blocking=True)
            VB = VB.to(pl_module.device,non_blocking=True)
            FB = FB.to(pl_module.device,non_blocking=True)
            FFB = FFB.to(pl_module.device,non_blocking=True)

            with torch.no_grad():

                images, PF = pl_module.render(V, F, FF)                    
                images_fiber, PF_fiber = pl_module.render(VFI, FFI, FFFI)
                images_brain, PF_brain = pl_module.render(VB, FB, FFB)
                images = torch.cat((images, images_fiber, images_brain), dim=1)
                grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image test', grid_images, pl_module.global_step)

                # images_noiseM = pl_module.noise(images)