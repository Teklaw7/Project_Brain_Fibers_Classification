import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl 
import torchvision.models as models
from torch.nn.functional import softmax
import torchmetrics
from tools import utils
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)
from pytorch3d.renderer.blending import sigmoid_alpha_blend, hard_rgb_blend
from pytorch3d.structures import Meshes, join_meshes_as_scene
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pytorch3d.vis.plotly_vis import plot_scene
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from sklearn.utils.class_weight import compute_class_weight
import random
import pytorch3d.transforms as T3d
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# import MLP
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
import pandas as pd
from Transformations.transformations import *
from tools.loss_function_ts_ss import TS_SS

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


# class SelfAttention_without_reduction(nn.Module):
#     def __init__(self,in_units,out_units):
#         super().__init__()

#         self.W1 = nn.Linear(in_units, out_units)
#         self.V = nn.Linear(out_units, 1)
#         self.Tanh = nn.Tanh()
#         self.Sigmoid = nn.Sigmoid()
    
#     def forward(self, query, values):        
#         score = self.Sigmoid(self.V(self.Tanh(self.W1(query))))
#         attention_weights = score/torch.sum(score, dim=1,keepdim=True)
#         context_vector = attention_weights * values
#         return context_vector, score


# class AvgPoolImages(nn.Module):
#     def __init__(self, nbr_images = 12):
#         super().__init__()
#         self.nbr_images = nbr_images
#         self.avg_pool = nn.AvgPool1d(self.nbr_images)
 
#     def forward(self,x):
#         x = x.permute(0,2,1)
#         output = self.avg_pool(x)
#         output = output.squeeze(dim=2)

#         return output


# class IcosahedronConv2d(nn.Module):
#     def __init__(self,module,verts,list_edges):
#         super().__init__()
#         self.module = module
#         self.verts = verts
#         self.list_edges = list_edges
#         self.nbr_vert = np.max(self.list_edges)+1

#         self.list_neighbors = self.get_neighbors()
#         self.list_neighbors = self.sort_neighbors()
#         self.list_neighbors = self.sort_rotation()
#         mat_neighbors = self.get_mat_neighbors()

#         self.register_buffer("mat_neighbors", mat_neighbors)

    
#     def get_neighbors(self):
#         neighbors = [[] for i in range(self.nbr_vert)]
#         for edge in self.list_edges:
#             v1 = edge[0]
#             v2 = edge[1]
#             neighbors[v1].append(v2)
#             neighbors[v2].append(v1)
#         return neighbors
    
#     def sort_neighbors(self):
#         new_neighbors = [[] for i in range(self.nbr_vert)]
#         for i in range(self.nbr_vert):
#             neighbors = self.list_neighbors[i].copy()
#             vert = neighbors[0]
#             new_neighbors[i].append(vert)
#             neighbors.remove(vert)
#             while len(neighbors) != 0:
#                 common_neighbors = list(set(neighbors).intersection(self.list_neighbors[vert]))
#                 vert = common_neighbors[0]
#                 new_neighbors[i].append(vert)
#                 neighbors.remove(vert)
#         return new_neighbors
    
#     def sort_rotation(self):
#         new_neighbors = [[] for i in range(self.nbr_vert)]
#         for i in range(self.nbr_vert):
#             p0 = self.verts[i]
#             p1 = self.verts[self.list_neighbors[i][0]]
#             p2 = self.verts[self.list_neighbors[i][1]]
#             v1 = p1 - p0
#             v2 = p2 - p1
#             vn = torch.cross(v1,v2)
#             n = vn/torch.norm(vn)

#             milieu = p1 + v2/2
#             v3 = milieu - p0
#             cg = p0 + 2*v3/3

#             if (torch.dot(n,cg) > 1 ):
#                 new_neighbors[i] = self.list_neighbors[i]
#             else:
#                 self.list_neighbors[i].reverse()
#                 new_neighbors[i] = self.list_neighbors[i]

#         return new_neighbors
    
#     def get_mat_neighbors(self):
#         mat = torch.zeros(self.nbr_vert,self.nbr_vert*9)
#         for index_cam in range(self.nbr_vert):
#             mat[index_cam][index_cam*9] = 1
#             for index_neighbor in range(len(self.list_neighbors[index_cam])):
#                 mat[self.list_neighbors[index_cam][index_neighbor]][index_cam*9+index_neighbor+1] = 1
#         return mat

 
#     def forward(self,x):
#         batch_size,nbr_cam,nbr_features = x.size()
#         x = x.permute(0,2,1)
#         size_reshape = [batch_size*nbr_features,nbr_cam]
#         x = x.contiguous().view(size_reshape)
#         x_fiber = x[:,0:12]
#         # x_brain = x[:,12:]    # with brain
#         x_fiber = torch.mm(x_fiber,self.mat_neighbors)
#         # x_brain = torch.mm(x_brain,self.mat_neighbors) # with brain
#         # x = torch.cat((x_fiber,x_brain),dim=1) # with brain
#         x = x_fiber
#         size_reshape2 = [batch_size,nbr_features,nbr_cam,3,3]
#         x = x.contiguous().view(size_reshape2)
#         x = x.permute(0,2,1,3,4)
#         #size_reshape2 = [batch_size,nbr_cam,nbr_features,3,3]
#         #x = x.contiguous().view(size_reshape2)
#         #x = x.permute(0,2,1,3,4)

#         size_reshape3 = [batch_size*nbr_cam,nbr_features,3,3]
#         x = x.contiguous().view(size_reshape3)

#         #size_reshape3 = [batch_size*nbr_cam,nbr_features,3,3]
#         #x = x.contiguous().view(size_reshape3)

#         output = self.module(x)
#         output_channels = self.module.out_channels
#         size_initial = [batch_size,nbr_cam,output_channels]
#         output = output.contiguous().view(size_initial)
#         return output

class ProjectionHead(nn.Module):
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
        # x = torch.abs(x)  
        return F.normalize(x, dim=1)


class Fly_by_Contrastive(nn.Module):
    def __init__(self):
        super().__init__()

        efficient_net = models.resnet18(pretrained = True)    ### maybe use weights instead of pretrained    
        # efficient_net.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # depthmap
        efficient_net.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # depthmap # with new features from DTI
        efficient_net.fc = Identity()
        # self.drop = nn.Dropout(p=dropout_lvl)
        self.TimeDistributed = TimeDistributed(efficient_net)
        self.WV = nn.Linear(512, 512)
        self.Attention = SelfAttention(512,512)
        # self.Projection = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)

    def forward(self, x):
        x_f = self.TimeDistributed(x)
        x_v = self.WV(x_f)
        x, x_s = self.Attention(x_f, x_v)
        # x = self.Projection(x)
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
        self.Projection = ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=128)
        # self.Projection = ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=3)

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

        self.lights = torch.tensor(pd.read_pickle(r'lights_good_on_sphere.pickle')).to(self.device) #normalized
        # self.lights = torch.tensor(pd.read_pickle(r'lights_good_on_sphere_without_norm.pickle')).to(self.device) #no normalized
        # self.lights = torch.tensor(pd.read_pickle(r'lights_57_3d_on_positive_sphere.pickle')).to(self.device) #no normalized
        # self.closest_lights = torch.load('closest_lights_57_3d_on_positive_sphere.pt').to(self.device) #no normalized
        # self.lights = torch.tensor(pd.read_pickle(r'lights_57_3d_on_sphere.pickle')).to(self.device) #no normalized
        self.loss_cossine = nn.CosineSimilarity()
        self.loss_cossine_dim2 = nn.CosineSimilarity(dim=2)
        self.loss_cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI = x
        x, PF = self.render(V,F,FF) # bs, 12,nb_features, 224, 224
        X1, PF1 = self.render(V1,F,FF)
        X2, PF2 = self.render(V2,F,FF)
        x_fiber, PF_fiber = self.render(VFI,FFI,FFFI)   # bs, 12,nb_features, 224, 224
        X1_fiber, PF1_fiber = self.render(VFI1,FFI,FFFI)
        X2_fiber, PF2_fiber = self.render(VFI2,FFI,FFFI)

        proj_fiber = self.model_original(x_fiber) #bs,512
        proj_brain = self.model_brain(x) #bs,512

        proj_fiber_1 = self.model_original(X1_fiber)
        proj_brain_1 = self.model_brain(X1)
        proj_fiber_2 = self.model_original(X2_fiber)
        proj_brain_2 = self.model_brain(X2)
        middle = proj_fiber.shape[0]//2
        x = torch.cat((proj_fiber, proj_brain), dim=1) #bs,1024
        x_class = torch.cat((proj_fiber[:middle], proj_brain[:middle]), dim=1) # we take just the first half from labeled fibers
        x1 = torch.cat((proj_fiber_1[middle:], proj_brain_1[middle:]), dim=1)
        x2 = torch.cat((proj_fiber_2[middle:], proj_brain_2[middle:]), dim=1)
        res_for_class = self.Classification(x_class) #bs,57
        return self.Projection(x), res_for_class, self.Projection(x1), self.Projection(x2)

    
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
        V1 = randomrot(V1).to(self.device)
        V2 = randomrot(V2).to(self.device)
        VFI1 = randomrot(VFI1).to(self.device)
        VFI2 = randomrot(VFI2).to(self.device)
        proj_test, x_class, proj_1, proj_2 = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI))
        middle = proj_test.shape[0]//2
        labels_b = labels[:middle].to(self.device)
        proj_test_b = proj_test[:middle].to(self.device)
        x1_t = proj_1.to(self.device)
        x2_t = proj_2.to(self.device)
        lights = self.lights.to(self.device)
        add_noise = False
        if add_noise:
            noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1]))
            lights += noise
            lights = lights / torch.norm(lights, dim=1, keepdim=True)

        loss_cross_entropy = self.loss_cross_entropy(x_class, labels_b)
        lam = 1
        loss_align_uniformity = align_loss(lights[labels_b], proj_test_b) + 0.1*lam * (uniform_loss(proj_test_b) + uniform_loss(lights[labels_b])) / 2 

        if x1_t.shape[0] > 0:   # if tractography fibers are in the batch
            loss_align_uniformity_tract = align_loss(x1_t,x2_t) + 0.1*lam * (uniform_loss(x1_t) + uniform_loss(x2_t)) / 2
            # loss_contrastive_tractography = 1 - self.loss_cossine(x1_t, x2_t) # 1 - cossine(Ft1,Ft2)
            # loss_contrastive_shuffle = self.loss_cossine(x1_t, x2_t_r)  # 1 - cossine(Ft1,Ft2_shuffled)
            # loss_contrastive_tractography = torch.sum(loss_contrastive_tractography).to(self.device)
            # loss_contrastive_shuffle = torch.sum(loss_contrastive_shuffle).to(self.device)

            # X1_T = x1_t.unsqueeze(1).repeat(1,self.lights.shape[0],1).to(self.device) #(7,57,128)
            # LT = self.lights.unsqueeze(0).repeat(x1_t.shape[0],1,1).to(self.device) #(7,57,128)
            # loss_for_best = self.loss_cossine_dim2(LT, X1_T) #(7,57)        #compute the cosinesimilarity between each tractography fiber and each cluster
            # topk_i = torch.topk(loss_for_best, 5,dim = 1).indices #(7,5)     #get the 5 closest clusters for each tractography fiber
            # Nl = [ i.item() for i in torch.randint(-5,0,(7,))]  #randomly choose one of the 5 closest clusters for each tractography fiber
            # topk_i_choosen=[topk_i[i,Nl[i]].item() for i in range(len(Nl))] #get the index of the choosen cluster for each tractography fiber
            # lights_topk = self.lights[topk_i_choosen].to(self.device) #(7,128)
            # loss_tract_cluster = self.loss_cossine(lights_topk, x1_t) #(7,57)
            # loss_tract_cluster = torch.sum(loss_tract_cluster).to(self.device)
        else:
            Loss_combine = 0#0loss_contrastive_bundle #+ loss_closest_point#+ 0.5*loss_contrastive_bundle_repulse #+ loss_contrastive_bundle_repulse
        Loss_combine = loss_align_uniformity + loss_cross_entropy + loss_align_uniformity_tract#+ loss_contrastive
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
        proj_test, x_class, proj_1, proj_2 = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI))
        middle = proj_test.shape[0]//2
        labels_b = labels[:middle].to(self.device)
        proj_test_b = proj_test[:middle].to(self.device)
        x1_t = proj_1.to(self.device)
        x2_t = proj_2.to(self.device)
        lights= self.lights.to(self.device)
        add_noise = False
        if add_noise:
            noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1]))
            lights += noise
            lights = lights / torch.norm(lights, dim=1, keepdim=True)
        # loss_contrastive = self.loss_contrastive(lights[labels_b],proj_test) # Simclr between the two augmentations of the original data
        # loss_contrastive_bundle = 1 - self.loss_cossine(lights[labels_b],proj_test_b)# + 1 - self.loss_cossine(self.lights[labels_b],x2_b) + 1 - self.loss_cossine(x1_b,x2_b)
        loss_cross_entropy = self.loss_cross_entropy(x_class, labels_b) # return one value

        lam = 1
        loss_align_uniformity = align_loss(lights[labels_b], proj_test_b) + 0.1*lam * (uniform_loss(proj_test_b) + uniform_loss(lights[labels_b])) / 2 

        if x1_t.shape[0] > 0:   # if tractography fibers are in the batch
            # r = torch.randperm(int(x2_t.shape[0]))
            # x2_t_r = x2_t[r].to(self.device)
            loss_align_uniformity_tract = align_loss(x1_t,x2_t) + 0.1*lam * (uniform_loss(x1_t) + uniform_loss(x2_t)) / 2

            # loss_contrastive_tractography = 1 - self.loss_cossine(x1_t, x2_t) # 1 - cossine(Ft1,Ft2)
            # loss_contrastive_shuffle = self.loss_cossine(x1_t, x2_t_r)  # 1 - cossine(Ft1,Ft2_shuffled)
            # loss_contrastive_tractography = torch.sum(loss_contrastive_tractography)
            # loss_contrastive_shuffle = torch.sum(loss_contrastive_shuffle)

            # X1_T = x1_t.unsqueeze(1).repeat(1,self.lights.shape[0],1).to(self.device) #(7,57,128)
            # LT = self.lights.unsqueeze(0).repeat(x1_t.shape[0],1,1).to(self.device)  #(7,57,128)
            # loss_for_best = self.loss_cossine_dim2(LT, X1_T) #(7,57)    #compute the cosinesimilarity between each tractography fiber and each cluster
            # topk_i = torch.topk(loss_for_best, 5,dim = 1).indices       #get the 5 best clusters for each tractography fiber
            # Nl = [ i.item() for i in torch.randint(-5,0,(7,))]          #get a random number between -5 and 0 for each tractography fiber
            # topk_i_choosen=[topk_i[i,Nl[i]].item() for i in range(len(Nl))] #get the cluster corresponding to the random number
            # lights_topk = self.lights[topk_i_choosen].to(self.device) #(7,128)
            # loss_tract_cluster = self.loss_cossine(lights_topk, x1_t) #(7,57)
            # loss_tract_cluster = torch.sum(loss_tract_cluster).to(self.device)

            # Loss_combine = loss_contrastive_bundle + loss_contrastive_tractography + loss_contrastive_shuffle #+ loss_tract_cluster
        else:
            Loss_combine = 0#loss_contrastive_bundle #+ loss_closest_point#+ 0.5*loss_contrastive_bundle_repulse #+ loss_contrastive_bundle_repulse
        Loss_combine = loss_align_uniformity + loss_cross_entropy + loss_align_uniformity_tract#+ loss_contrastive
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
        proj_test, x_class, proj_1, proj_2 = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI))
        tot = np.random.randint(0,1000000)
        labels2 = labels.unsqueeze(dim = 1)
        data_lab = data_lab.unsqueeze(dim = 1)
        proj_test_save = torch.cat((proj_test, labels2, name_labels, data_lab), dim=1)
        lab = np.array(torch.unique(labels).cpu())
        torch.save(proj_test_save, f"/CMF/data/timtey/results_contrastive_learning_063023/proj_test_{lab[-1]}_{tot}.pt")

    def test_epoch_end(self, outputs):
        self.y_pred = []
        self.y_true = []

        for output in outputs:
            self.y_pred.append(output[0].tolist())
            self.y_true.append(output[1].tolist())
        self.y_pred = [ele for sousliste in self.y_pred for ele in sousliste]
        self.y_true = [ele for sousliste in self.y_true for ele in sousliste]
        self.y_pred = [[int(ele)] for ele in self.y_pred]
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        # print("accuracy", self.accuracy)
        # print(classification_report(self.y_true, self.y_pred))
        
    def get_y_pred(self):
        return self.y_pred
    
    def get_y_true(self):
        return self.y_true
    
    def get_accuracy(self):
        return self.accuracy



def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)

class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.4):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)
        self.mask = (~torch.eye(proj_1.shape[0] * 2, proj_2.shape[0] * 2, dtype=bool)).float()
        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss