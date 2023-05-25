
from pytorch_lightning.callbacks import Callback
import torchvision
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import pytorch3d.transforms as T3d
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
from Transformations.transformations import *


class BrainNetImageLogger_contrastive_tractography_labeled(Callback):
    def __init__(self,num_features = 3 , num_images=12, log_steps=10,mean=0,std=0.015):
        self.num_features = num_features
        self.log_steps = log_steps
        self.num_images = num_images
        self.mean = mean
        self.std = std

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

            V, F, FF, labels, Fiber_infos = batch
            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
            FF = FF.to(pl_module.device,non_blocking=True)
            VFI = torch.clone(V)
            VFI = VFI.to(pl_module.device,non_blocking=True)
            FFI = torch.clone(F)
            FFI = FFI.to(pl_module.device,non_blocking=True)
            FFFI = torch.clone(FF)
            FFFI = FFFI.to(pl_module.device,non_blocking=True)

            # vfbounds = []
            sample_min_max = []
            data_lab = []
            name_labels = []

            for z in range(len(Fiber_infos)):
                # vfbounds += [Fiber_infos[z][0]]
                sample_min_max += [Fiber_infos[z][0]]
                data_lab += Fiber_infos[z][1]
                name_labels += Fiber_infos[z][2]

            sample_id = []
            for w in range(len(name_labels)):
                sample_id.append(name_labels[w][0])

            V1 = randomstretching(V).to(pl_module.device,non_blocking=True)
            V2 = randomstretching(V).to(pl_module.device,non_blocking=True)
            VFI1 = randomstretching(VFI).to(pl_module.device,non_blocking=True)
            VFI2 = randomstretching(VFI).to(pl_module.device,non_blocking=True)
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
                images = torch.cat((images, images_fiber, images1, images2, images_fiber_1, images_fiber_2), dim=1)
                grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image train', grid_images, pl_module.global_step)
                images_augmentations = torch.cat((images1, images2, images_fiber_1, images_fiber_2), dim=1)
                grid_images_const = torchvision.utils.make_grid(images_augmentations[0, self.num_images:, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image train const', grid_images_const, pl_module.global_step)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):        

        if batch_idx % self.log_steps == 0:

            V, F, FF, labels, Fiber_infos= batch
            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
            FF = FF.to(pl_module.device,non_blocking=True)
            VFI =torch.clone(V)
            VFI = VFI.to(pl_module.device,non_blocking=True)
            FFI =torch.clone(F)
            FFI = FFI.to(pl_module.device,non_blocking=True)
            FFFI =torch.clone(FF)
            FFFI = FFFI.to(pl_module.device,non_blocking=True)

            # vfbounds = []
            sample_min_max = []
            data_lab = []
            name_labels = []

            for z in range(len(Fiber_infos)):
                # vfbounds += [Fiber_infos[z][0]]
                sample_min_max += [Fiber_infos[z][0]]
                data_lab += Fiber_infos[z][1]
                name_labels += Fiber_infos[z][2]
            sample_id = []
            for w in range(len(name_labels)):
                sample_id.append(name_labels[w][0])

            V1 = randomstretching(V).to(pl_module.device,non_blocking=True)
            V2 = randomstretching(V).to(pl_module.device,non_blocking=True)
            VFI1 = randomstretching(VFI).to(pl_module.device,non_blocking=True)
            VFI2 = randomstretching(VFI).to(pl_module.device,non_blocking=True)
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
                # images_brain, PF_brain = pl_module.render(VB, FB, FFB)   
                images = torch.cat((images, images_fiber, images1, images2, images_fiber_1, images_fiber_2), dim=1)
                grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image val', grid_images, pl_module.global_step)
                grid_images_const = torchvision.utils.make_grid(images[0, self.num_images:48, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image val const', grid_images_const, pl_module.global_step)
                grid_images_const_fiber = torchvision.utils.make_grid(images[0, 48:, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image val const fiber', grid_images_const_fiber, pl_module.global_step)

    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):        

        if batch_idx % self.log_steps == 0:
            V, F, FF, labels, Fiber_infos = batch
            
            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
            FF = FF.to(pl_module.device,non_blocking=True)
            VFI = torch.clone(V)
            VFI = VFI.to(pl_module.device,non_blocking=True)
            FFI = torch.clone(F)
            FFI = FFI.to(pl_module.device,non_blocking=True)
            FFFI = torch.clone(FF)
            FFFI = FFFI.to(pl_module.device,non_blocking=True)
            # vfbounds  = []
            sample_min_max = []
            data_lab = []
            name_labels = []    
            for z in range(len(Fiber_infos)):
                # vfbounds += [Fiber_infos[z][0]]
                sample_min_max += [Fiber_infos[z][0]]
                data_lab += Fiber_infos[z][1]
                name_labels += Fiber_infos[z][2]
            sample_id = []
            for w in range(len(name_labels)):
                sample_id.append(name_labels[w][0])
                
            V1 = randomstretching(V).to(pl_module.device,non_blocking=True)
            V2 = randomstretching(V).to(pl_module.device,non_blocking=True)
            VFI1 = randomstretching(VFI).to(pl_module.device,non_blocking=True)
            VFI2 = randomstretching(VFI).to(pl_module.device,non_blocking=True)
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
                # images_brain, PF_brain = pl_module.render(VB, FB, FFB)
                images = torch.cat((images, images_fiber, images1, images2, images_fiber_1, images_fiber_2), dim=1)
                grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image test', grid_images, pl_module.global_step)
                grid_images_const = torchvision.utils.make_grid(images[0, self.num_images:, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image test const', grid_images_const, pl_module.global_step)


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
            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
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


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):        

        if batch_idx % self.log_steps == 0:

            V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = batch
            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
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

    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):        

        if batch_idx % self.log_steps == 0:

            V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = batch

            V = V.to(pl_module.device,non_blocking=True)
            F = F.to(pl_module.device,non_blocking=True)
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
