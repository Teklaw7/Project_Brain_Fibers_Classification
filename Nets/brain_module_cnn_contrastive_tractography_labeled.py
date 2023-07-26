import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl 
import torchvision.models as models
from tools import utils
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
import pandas as pd
from Transformations.transformations import *
# from tools.loss_function_ts_ss import TS_SS

def GetView(meshes,phong_renderer,R,T):
    R = R.to(torch.float32)
    T = T.to(torch.float32)
    images = phong_renderer(meshes.clone(), R=R, T=T)
    fragments = phong_renderer.rasterizer(meshes.clone(),R=R,T=T)
    pix_to_face = fragments.pix_to_face
    zbuf = fragments.zbuf #shape == (batchsize, image_size, image_size, faces_per_pixel) 
    images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
    images = images.permute(0,3,1,2)
    pix_to_face = pix_to_face.permute(0,3,1,2)
    return pix_to_face, images


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
        reshaped_input = reshaped_input.to(torch.float32)
        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output

class SelfAttention(nn.Module):
    def __init__(self,in_units,out_units):
        super().__init__()

        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, query, values):        
        score = self.Sigmoid(self.V(self.Tanh(self.W1(query))))
        attention_weights = score/torch.sum(score, dim=1,keepdim=True)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, score

class ProjectionHead(nn.Module): # class to create a projection of the fiber
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        # x = torch.abs(x)     # to add if we want to create projections on positive hypersphere
        return F.normalize(x, dim=1)


class Fly_by_Contrastive(nn.Module):
    def __init__(self):
        super().__init__()

        efficient_net = models.resnet18(pretrained = True)    ### maybe use weights instead of pretrained    
        efficient_net.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # depthmap # with new features from DTI
        efficient_net.fc = Identity()
        # self.drop = nn.Dropout(p=dropout_lvl) # if we want to add dropout
        self.TimeDistributed = TimeDistributed(efficient_net)
        self.WV = nn.Linear(512, 512)
        self.Attention = SelfAttention(512,512)

    def forward(self, x):
        x_f = self.TimeDistributed(x)
        x_v = self.WV(x_f)
        x, x_s = self.Attention(x_f, x_v)
        return x

class Fly_by_CNN_contrastive_tractography_labeled(pl.LightningModule):
    def __init__(self, radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.loss_contrastive = ContrastiveLoss(batch_size)
        self.radius = radius
        self.ico_lvl = ico_lvl
        self.dropout_lvl = dropout_lvl
        self.image_size = 224
        self.batch_size = batch_size ###be careful
        self.weights = weights
        self.num_classes = num_classes
        ico_sphere = utils.CreateIcosahedron(self.radius, ico_lvl)
        ico_sphere_verts, ico_sphere_faces, self.ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_faces = ico_sphere_faces
        self.ico_sphere_edges = np.array(self.ico_sphere_edges)
        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_pos = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_pos)
            T_current = -torch.bmm(R_current.transpose(1,2), camera_pos[:,:,None])[:,:,0]
            R.append(R_current)
            T.append(T_current)

        self.R = torch.cat(R)
        self.T = torch.cat(T)
        self.R = self.R.to(torch.float32)
        self.T = self.T.to(torch.float32)

        self.model_original = Fly_by_Contrastive()
        self.model_brain = Fly_by_Contrastive()
        self.Classification = nn.Linear(1024, 57)
        self.Projection = ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=128) # projections in 128D
        # self.Projection = ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=3) # projections in 3D

        self.cameras = FoVPerspectiveCameras()

        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0, 
            faces_per_pixel=1, 
            max_faces_per_bin=100000
        )

        lights = AmbientLights()
        rasterizer = MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            )

        self.phong_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=self.cameras, lights=lights)
        )

        # self.lights = torch.tensor(pd.read_pickle(r'/CMF/data/timtey/Lights/lights_good_on_sphere.pickle')).to(self.device) #normalized # if lights are needed
        self.loss_cossine = nn.CosineSimilarity()
        self.loss_cossine_dim2 = nn.CosineSimilarity(dim=2)
        self.loss_cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI = x
        
        x, PF = self.render(V,F,FF) # bs, 12,nb_features, 224, 224 #fiber normalized by brain bounds
        X1, PF1 = self.render(V1,F,FF)  #   fisrt augmentation of fiber normalized by brain bounds
        X2, PF2 = self.render(V2,F,FF)  #   second augmentation of fiber normalized by brain bounds

        x_fiber, PF_fiber = self.render(VFI,FFI,FFFI)   # bs, 12,nb_features, 224, 224  #fiber normalized by fiber bounds
        X1_fiber, PF1_fiber = self.render(VFI1,FFI,FFFI)    #   fisrt augmentation of fiber normalized by fiber bounds
        X2_fiber, PF2_fiber = self.render(VFI2,FFI,FFFI)    #   second augmentation of fiber normalized by fiber bounds

        proj_fiber = self.model_original(x_fiber) #bs,512   # result of the model for the fiber normalized by fiber bounds
        proj_brain = self.model_brain(x) #bs,512    # result of the model for the fiber normalized by brain bounds

        proj_fiber_1 = self.model_original(X1_fiber)    #   fisrt augmentation of fiber normalized by fiber bounds
        proj_brain_1 = self.model_brain(X1) #bs,512 #   fisrt augmentation of fiber normalized by brain bounds
        proj_fiber_2 = self.model_original(X2_fiber)    #   second augmentation of fiber normalized by fiber bounds
        proj_brain_2 = self.model_brain(X2) #bs,512 #   second augmentation of fiber normalized by brain bounds
        # The batch is composed of bs number of labeled fibers and bs number of tractography fibers
        
        middle = proj_fiber.shape[0]//2
        x = torch.cat((proj_fiber, proj_brain), dim=1) #bs,1024 # concatenation of the features from the fiber normalized by brain bounds and the fiber normalized by fiber bounds
        x_class = torch.cat((proj_fiber[:middle], proj_brain[:middle]), dim=1) # we take just the first half from labeled fibers for the classification
        xb1 = torch.cat((proj_fiber_1[:middle], proj_brain_1[:middle]), dim=1)  # concatenation of the features from the fiber normalized by brain bounds and the fiber normalized by fiber bounds
        xb2 = torch.cat((proj_fiber_2[:middle], proj_brain_2[:middle]), dim=1)  # concatenation of the features from the fiber normalized by brain bounds and the fiber normalized by fiber bounds
        xt1 = torch.cat((proj_fiber_1[middle:], proj_brain_1[middle:]), dim=1)  # concatenation of the features from the fiber normalized by brain bounds and the fiber normalized by fiber bounds
        xt2 = torch.cat((proj_fiber_2[middle:], proj_brain_2[middle:]), dim=1)  # concatenation of the features from the fiber normalized by brain bounds and the fiber normalized by fiber bounds
        # res_for_class = self.Classification(x_class) #bs,57   #if we use the classification in the loss function
        
        # return self.Projection(x), res_for_class, self.Projection(x1), self.Projection(x2)    # return the projections and the classification if needed
        return self.Projection(x), self.Projection(xb1), self.Projection(xb2), self.Projection(xt1), self.Projection(xt2) #return projections of the augmentations and the original fibers

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def render(self, V, F, FF):

        textures = TexturesVertex(verts_features=torch.ones(V.shape))
        V = V.to(torch.float32)
        F = F.to(torch.float32)
        V = V.to(self.device)
        F = F.to(self.device)
        textures = textures.to(self.device)
        meshes_fiber = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )

        phong_renderer = self.phong_renderer.to(self.device)
        meshes_fiber = meshes_fiber.to(self.device)
        PF = []
        X = []
        for i in range(len(self.R)):
            R = self.R[i][None].to(self.device)
            T = self.T[i][None].to(self.device)
            pixel_to_face, images = GetView(meshes_fiber,phong_renderer,R,T)
            images = images.to(self.device)
            PF.append(pixel_to_face.unsqueeze(dim=1))
            X.append(images.unsqueeze(dim=1))

        PF = torch.cat(PF,dim=1)
        X = torch.cat(X,dim=1)
        X = X[:,:,3,:,:] # the last one who has the picture in black and white of depth
        X = X.unsqueeze(dim=2)
        
        l_features = []

        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,index],PF)*(PF >= 0)) # take each feature

        x = torch.cat(l_features,dim=2)
        x = torch.cat((x,X),dim=2)
        return x, PF

    def training_step(self, train_batch, train_batch_idx):

        V, F, FF, labels, Fiber_infos, Mean, Scale = train_batch
        V = V.to(self.device)
        Vo = torch.detach(V).to(self.device)
        F = F.to(self.device)
        FF = FF.to(self.device)
        VFI = torch.detach(V).to(self.device)
        FFI = torch.detach(F).to(self.device)
        FFFI = torch.detach(FF).to(self.device)
        labels = labels.to(self.device)
        Fiber_infos = Fiber_infos.to(self.device)
        Mean = Mean.to(self.device)
        Scale = Scale.to(self.device)
        mean_v = Mean[:,:3].unsqueeze(dim=1).repeat(1,VFI.shape[1],1).to(self.device) # 1x3
        scale_v = Scale[:,0].to(self.device) # 1x1
        mean_s = Mean[:,3:].unsqueeze(dim=1).repeat(1,Vo.shape[1],1).to(self.device) # 1x3
        scale_s = Scale[:,1].to(self.device) # 1x1

        Vo = transformation_verts(Vo, mean_s, scale_s)         #normalization by the bounds of the brain
        VFI = transformation_verts_by_fiber(VFI, mean_v, scale_v)  #normalization by the bounds of the fiber
        V1 = randomstretching(Vo.double()).to(self.device)            #random stretching
        V2 = randomstretching(Vo.double()).to(self.device)
        VFI1 = randomstretching(VFI.double()).to(self.device)
        VFI2 = randomstretching(VFI.double()).to(self.device)
        V1 = randomrot(V1).to(self.device)                            # random rotation
        V2 = randomrot(V2).to(self.device)
        VFI1 = randomrot(VFI1).to(self.device)
        VFI2 = randomrot(VFI2).to(self.device)
        proj_test, proj_bundle_1, proj_bundle_2, proj_tract_1, proj_tract_2 = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI))
        proj_bundle_1 = proj_bundle_1.to(self.device)
        proj_bundle_2 = proj_bundle_2.to(self.device)
        proj_tract_1 = proj_tract_1.to(self.device)
        proj_tract_2 = proj_tract_2.to(self.device)

        # loss_cross_entropy = self.loss_cross_entropy(x_class, labels_b)   # if we use the classification in the loss function
        
        # loss align + uniformity
        lam = 1 # hyperparameter for loss align and uniformity
        loss_align_uniformity = align_loss(proj_bundle_1, proj_bundle_2) + 0.1*lam * (uniform_loss(proj_bundle_1) + uniform_loss(proj_bundle_2)) / 2 

        if proj_tract_1.shape[0] > 0:   # if tractography fibers are in the batch
            loss_align_uniformity_tract = align_loss(proj_tract_1,proj_tract_2) + 0.1*lam * (uniform_loss(proj_tract_1) + uniform_loss(proj_tract_2)) / 2

        Loss_combine = loss_align_uniformity + loss_align_uniformity_tract
        self.log('train_loss', Loss_combine.item(), batch_size=self.batch_size)
        return Loss_combine

        
    def validation_step(self, val_batch, val_batch_idx):
        
        V, F, FF, labels, Fiber_infos, Mean, Scale = val_batch
        V = V.to(self.device)
        Vo = torch.detach(V).to(self.device)
        F = F.to(self.device)
        FF = FF.to(self.device)
        VFI = torch.detach(V).to(self.device)
        FFI = torch.detach(F).to(self.device)
        FFFI = torch.detach(FF).to(self.device)
        labels = labels.to(self.device)
        Fiber_infos = Fiber_infos.to(self.device) # bs x 4
        Mean = Mean.to(self.device) # bs x 6
        Scale = Scale.to(self.device) # bs x 2
        mean_v = Mean[:,:3].unsqueeze(1).repeat(1,VFI.shape[1],1).to(self.device) # bs x 3
        scale_v = Scale[:,0].to(self.device) # bs x 1
        mean_s = Mean[:,3:].unsqueeze(1).repeat(1,Vo.shape[1],1).to(self.device) # bs x 3
        scale_s = Scale[:,1].to(self.device) # bs x 1
        Vo = transformation_verts(Vo, mean_s, scale_s)         #normalization by the bounds of the brain
        VFI = transformation_verts_by_fiber(VFI, mean_v, scale_v)  #normalization by the bounds of the fiber
        V1 = randomstretching(Vo.double()).to(self.device)            #random stretching of the fiber
        V2 = randomstretching(Vo.double()).to(self.device)
        VFI1 = randomstretching(VFI.double()).to(self.device)
        VFI2 = randomstretching(VFI.double()).to(self.device)
        V1 = randomrot(V1).to(self.device)
        V2 = randomrot(V2).to(self.device)
        VFI1 = randomrot(VFI1).to(self.device)
        VFI2 = randomrot(VFI2).to(self.device)
        proj_test, proj_bundle_1, proj_bundle_2, proj_tract_1, proj_tract_2 = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI))
        proj_bundle_1 = proj_bundle_1.to(self.device)
        proj_bundle_2 = proj_bundle_2.to(self.device)
        proj_tract_1 = proj_tract_1.to(self.device)
        proj_tract_2 = proj_tract_2.to(self.device)

        # loss_cross_entropy = self.loss_cross_entropy(x_class, labels_b) # return one value for classification loss if needed

        # loss align + uniformity
        lam = 1
        loss_align_uniformity = align_loss(proj_bundle_1, proj_bundle_2) + 0.1*lam * (uniform_loss(proj_bundle_1) + uniform_loss(proj_bundle_2)) / 2 

        if proj_tract_1.shape[0] > 0:   # if tractography fibers are in the batch
            loss_align_uniformity_tract = align_loss(proj_tract_1,proj_tract_2) + 0.1*lam * (uniform_loss(proj_tract_1) + uniform_loss(proj_tract_2)) / 2

        Loss_combine = loss_align_uniformity + loss_align_uniformity_tract
        self.log('val_loss', Loss_combine.item(), batch_size=self.batch_size)

    def test_step(self, test_batch, test_batch_idx):

        V, F, FF, labels, Fiber_infos, Mean, Scale = test_batch  
        V = V.to(self.device)
        Vo = torch.detach(V).to(self.device)
        F = F.to(self.device)
        FF = FF.to(self.device)
        VFI = torch.detach(V).to(self.device)
        FFI = torch.detach(F).to(self.device)
        FFFI = torch.detach(FF).to(self.device)
        labels = labels.to(self.device)
        Fiber_infos = Fiber_infos.to(self.device)
        Mean = Mean.to(self.device)
        Scale = Scale.to(self.device)
        mean_v = Mean[:,:3].unsqueeze(1).repeat(1,VFI.shape[1],1).to(self.device)
        scale_v = Scale[:,0].to(self.device)
        mean_s = Mean[:,3:].unsqueeze(1).repeat(1,Vo.shape[1],1).to(self.device)
        scale_s = Scale[:,1].to(self.device)
        data_lab = Fiber_infos[:,0].to(self.device)
        name_labels = Fiber_infos[:,1:]
        Vo = transformation_verts(Vo, mean_s,scale_s)         #normalisation by the bounds of the brain
        VFI = transformation_verts_by_fiber(VFI, mean_v, scale_v)  #normalisation by the bounds of the fiber
        V1 = randomstretching(Vo.double()).to(self.device)            #stretching
        V2 = randomstretching(Vo.double()).to(self.device)
        VFI1 = randomstretching(VFI.double()).to(self.device)
        VFI2 = randomstretching(VFI.double()).to(self.device)
        V1 = randomrot(V1).to(self.device)
        V2 = randomrot(V2).to(self.device)
        VFI1 = randomrot(VFI1).to(self.device)
        VFI2 = randomrot(VFI2).to(self.device)
        proj_test, proj_bundle1, proj_bundle2, proj_1, proj_2 = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI))
        tot = np.random.randint(0,1000000)  # random number to save the projected fibers and in order to get a different number for each file saved 
        labels2 = labels.unsqueeze(dim = 1)
        data_lab = data_lab.unsqueeze(dim = 1)
        proj_test_save = torch.cat((proj_test, labels2, name_labels, data_lab), dim=1)
        lab = np.array(torch.unique(labels).cpu())
        # torch.save(proj_test_save, f"/CMF/data/timtey/results_contrastive_learning_071823_best/proj_test_{lab[-1]}_{tot}.pt") # save the projected labeled and tractography fibers