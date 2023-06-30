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


class SelfAttention_without_reduction(nn.Module):
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
        return context_vector, score


class AvgPoolImages(nn.Module):
    def __init__(self, nbr_images = 12):
        super().__init__()
        self.nbr_images = nbr_images
        self.avg_pool = nn.AvgPool1d(self.nbr_images)
 
    def forward(self,x):
        x = x.permute(0,2,1)
        output = self.avg_pool(x)
        output = output.squeeze(dim=2)

        return output


class IcosahedronConv2d(nn.Module):
    def __init__(self,module,verts,list_edges):
        super().__init__()
        self.module = module
        self.verts = verts
        self.list_edges = list_edges
        self.nbr_vert = np.max(self.list_edges)+1

        self.list_neighbors = self.get_neighbors()
        self.list_neighbors = self.sort_neighbors()
        self.list_neighbors = self.sort_rotation()
        mat_neighbors = self.get_mat_neighbors()

        self.register_buffer("mat_neighbors", mat_neighbors)

    
    def get_neighbors(self):
        neighbors = [[] for i in range(self.nbr_vert)]
        for edge in self.list_edges:
            v1 = edge[0]
            v2 = edge[1]
            neighbors[v1].append(v2)
            neighbors[v2].append(v1)
        return neighbors
    
    def sort_neighbors(self):
        new_neighbors = [[] for i in range(self.nbr_vert)]
        for i in range(self.nbr_vert):
            neighbors = self.list_neighbors[i].copy()
            vert = neighbors[0]
            new_neighbors[i].append(vert)
            neighbors.remove(vert)
            while len(neighbors) != 0:
                common_neighbors = list(set(neighbors).intersection(self.list_neighbors[vert]))
                vert = common_neighbors[0]
                new_neighbors[i].append(vert)
                neighbors.remove(vert)
        return new_neighbors
    
    def sort_rotation(self):
        new_neighbors = [[] for i in range(self.nbr_vert)]
        for i in range(self.nbr_vert):
            p0 = self.verts[i]
            p1 = self.verts[self.list_neighbors[i][0]]
            p2 = self.verts[self.list_neighbors[i][1]]
            v1 = p1 - p0
            v2 = p2 - p1
            vn = torch.cross(v1,v2)
            n = vn/torch.norm(vn)

            milieu = p1 + v2/2
            v3 = milieu - p0
            cg = p0 + 2*v3/3

            if (torch.dot(n,cg) > 1 ):
                new_neighbors[i] = self.list_neighbors[i]
            else:
                self.list_neighbors[i].reverse()
                new_neighbors[i] = self.list_neighbors[i]

        return new_neighbors
    
    def get_mat_neighbors(self):
        mat = torch.zeros(self.nbr_vert,self.nbr_vert*9)
        for index_cam in range(self.nbr_vert):
            mat[index_cam][index_cam*9] = 1
            for index_neighbor in range(len(self.list_neighbors[index_cam])):
                mat[self.list_neighbors[index_cam][index_neighbor]][index_cam*9+index_neighbor+1] = 1
        return mat

 
    def forward(self,x):
        batch_size,nbr_cam,nbr_features = x.size()
        x = x.permute(0,2,1)
        size_reshape = [batch_size*nbr_features,nbr_cam]
        x = x.contiguous().view(size_reshape)
        x_fiber = x[:,0:12]
        # x_brain = x[:,12:]    # with brain
        x_fiber = torch.mm(x_fiber,self.mat_neighbors)
        # x_brain = torch.mm(x_brain,self.mat_neighbors) # with brain
        # x = torch.cat((x_fiber,x_brain),dim=1) # with brain
        x = x_fiber
        size_reshape2 = [batch_size,nbr_features,nbr_cam,3,3]
        x = x.contiguous().view(size_reshape2)
        x = x.permute(0,2,1,3,4)
        #size_reshape2 = [batch_size,nbr_cam,nbr_features,3,3]
        #x = x.contiguous().view(size_reshape2)
        #x = x.permute(0,2,1,3,4)

        size_reshape3 = [batch_size*nbr_cam,nbr_features,3,3]
        x = x.contiguous().view(size_reshape3)

        #size_reshape3 = [batch_size*nbr_cam,nbr_features,3,3]
        #x = x.contiguous().view(size_reshape3)

        output = self.module(x)
        output_channels = self.module.out_channels
        size_initial = [batch_size,nbr_cam,output_channels]
        output = output.contiguous().view(size_initial)
        return output

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
        x = torch.abs(x)  
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
    
# class Fly_by_Classification(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.resnet18(pretrained=True)
#         self.model.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # depthmap # with new features from DTI
#         self.model.fc = Identity()
#         self.TimeDistributed = TimeDistributed(self.model)
#         self.WV = nn.Linear(512, 256)
#         # self.linear = nn.Linear(256, 256)
#         self.Attention = SelfAttention_without_reduction(512,128)
#         # self.conv2d = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
#         # self.IcosahedronConv2d = IcosahedronConv2d(self.conv2d,)

#     def forward(self, x):
#         x = self.TimeDistributed(x) #10,12,512
#         values = self.WV(x)#10,12,256
#         x, x_s = self.Attention(x, values)#10,12,256
#         return x
    
# class Fly_by_Classification_Icoconv2d(nn.Module):
#     def __init__(self, ico_sphere_verts, ico_sphere_edges):
#         super().__init__()
#         self.ico_sphere_verts = ico_sphere_verts
#         self.ico_sphere_edges = ico_sphere_edges
#         self.model = models.resnet18(pretrained=True)
#         self.model.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # depthmap # with new features from DTI
#         self.model.fc = Identity()
#         self.TimeDistributed = TimeDistributed(self.model)
#         # self.WV = nn.Linear(512, 256)
#         # self.linear = nn.Linear(256, 256)
#         # self.Attention = SelfAttention_without_reduction(512,128)
#         # self.conv2d = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
#         self.conv2d = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=2,padding=0)
#         self.IcosahedronConv2d = IcosahedronConv2d(self.conv2d,self.ico_sphere_verts, self.ico_sphere_edges)
#         self.linear = nn.Linear(256, 256)
#         # self.IcosahedronConv2d_model_original = self.IcosahedronConv2d

#     def forward(self, x):
#         x = self.TimeDistributed(x) #10,12,512
#         x_ico = self.IcosahedronConv2d(x)
#         x = self.linear(x_ico)
#         return x

# class Fly_by_Res_Classification(nn.Module):
#     def __init__(self, num_classes, ico_sphere_verts, ico_sphere_edges):
#         super().__init__()
#         self.num_classes = num_classes
#         self.ico_sphere_verts = ico_sphere_verts
#         self.ico_sphere_edges = ico_sphere_edges
#         # self.model_original_classification = Fly_by_Classification()
#         # self.model_brain_classification = Fly_by_Classification()
#         self.model_original_classification = Fly_by_Classification_Icoconv2d(self.ico_sphere_verts, self.ico_sphere_edges)
#         self.model_brain_classification = Fly_by_Classification_Icoconv2d(self.ico_sphere_verts, self.ico_sphere_edges)
#         self.WVConcat = nn.Linear(512, 512)
#         self.Attention_Concat = SelfAttention(512,128)
#         self.Classification = nn.Linear(512, self.num_classes)

#     def forward(self, x_brain, x_fiber):
#         x_brain = self.model_brain_classification(x_brain)
#         x_fiber = self.model_original_classification(x_fiber)
#         x = torch.cat((x_brain, x_fiber), dim=2)#10,12,512
#         values = self.WVConcat(x)
#         x, x_s = self.Attention_Concat(x, values)
#         x = self.Classification(x)
#         return x

class Fly_by_CNN_contrastive_tractography_labeled(pl.LightningModule):
    # def __init__(self, contrastive, radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, verts_left, faces_left, verts_right, faces_right, learning_rate=0.001):
    def __init__(self, radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters()
        # self.model = models.resnet18(pretrained=True)
        # self.model.fc = nn.Linear(512, num_classes)
        # self.loss = nn.CrossEntropyLoss()
        self.loss_contrastive = ContrastiveLoss(batch_size)
        # self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
        # self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
        # self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
        # self.contrastive = contrastive
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
        # if contrastive:
        self.R = self.R.to(torch.float32)
        self.T = self.T.to(torch.float32)

        self.model_original = Fly_by_Contrastive()
        self.model_brain = Fly_by_Contrastive()
        # self.model_original_classification = Fly_by_Classification()
        # self.model_brain_classification = Fly_by_Classification()
        self.Classification = nn.Linear(1024, 57)
        # self.Projection = ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=128)
        self.Projection = ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=3)

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
        # self.phong_renderer_brain = MeshRenderer(
        #     rasterizer=rasterizer,
        #     shader=HardPhongShader(cameras=self.cameras, lights=lights)
        # )

        # self.loss_train = nn.CrossEntropyLoss(weight = self.weights[0])
        # self.loss_val = nn.CrossEntropyLoss(weight = self.weights[1])
        # self.loss_test = nn.CrossEntropyLoss(weight = self.weights[2])

        # self.lights = torch.tensor(pd.read_pickle(r'lights_good_on_sphere.pickle')).to(self.device) #normalized
        # self.lights = torch.tensor(pd.read_pickle(r'lights_good_on_sphere_without_norm.pickle')).to(self.device) #no normalized
        self.lights = torch.tensor(pd.read_pickle(r'lights_57_3d_on_positive_sphere.pickle')).to(self.device) #no normalized
        # self.closest_lights = torch.load('closest_lights_57_3d_on_positive_sphere.pt').to(self.device) #no normalized
        # self.lights = torch.tensor(pd.read_pickle(r'lights_57_3d_on_sphere.pickle')).to(self.device) #no normalized
        self.loss_cossine = nn.CosineSimilarity()
        self.loss_cossine_dim2 = nn.CosineSimilarity(dim=2)
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        # self.loss_ts_ss = TS_SS().forward()

    def forward(self, x):
        V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI = x
        x, PF = self.render(V,F,FF) # bs, 12,nb_features, 224, 224
        # X1, PF1 = self.render(V1,F,FF)
        # X2, PF2 = self.render(V2,F,FF)
        x_fiber, PF_fiber = self.render(VFI,FFI,FFFI)   # bs, 12,nb_features, 224, 224
        # X1_fiber, PF1_fiber = self.render(VFI1,FFI,FFFI)
        # X2_fiber, PF2_fiber = self.render(VFI2,FFI,FFFI)
        # x_test = torch.cat((x,x_fiber),1)
        # X1 = torch.cat((X1,X1_fiber),1)
        # X2 = torch.cat((X2,X2_fiber),1)
        # proj_brain_1 = self.model_brain(X1)
        # proj_brain_2 = self.model_brain(X2)
        # proj_fiber_1 = self.model_original(x_fiber)
        # proj_fiber_2 = self.model_original(x_fiber)        
        proj_fiber = self.model_original(x_fiber) #bs,512
        proj_brain = self.model_brain(x) #bs,512

        x = torch.cat((proj_fiber, proj_brain), dim=1) #bs,1024

        res_for_class = self.Classification(x) #bs,57
        # x1 = torch.cat((proj_fiber_1, proj_brain_1), dim=1)
        # x2 = torch.cat((proj_fiber_2, proj_brain_2), dim=1)
        return self.Projection(x), res_for_class#, self.Projection(x), self.Projection(x)

    
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
        # phong_renderer_brain = self.phong_renderer_brain.to(self.device)
        meshes_fiber = meshes_fiber.to(self.device)
        PF = []
        X = []
        # PF_brain = []
        for i in range(len(self.R)):
            R = self.R[i][None].to(self.device)
            T = self.T[i][None].to(self.device)
            pixel_to_face, images = GetView(meshes_fiber,phong_renderer,R,T)
            # pixel_to_face_brain = GetView(meshes_brain,phong_renderer_brain,R,T) # with brain
            images = images.to(self.device)
            PF.append(pixel_to_face.unsqueeze(dim=1))
            X.append(images.unsqueeze(dim=1))
            # PF_brain.append(pixel_to_face_brain.unsqueeze(dim=1)) # with brain

        PF = torch.cat(PF,dim=1)
        X = torch.cat(X,dim=1)
        X = X[:,:,3,:,:] # the last one who has the picture in black and white of depth
        X = X.unsqueeze(dim=2)
        
        # PF_brain = torch.cat(PF_brain,dim=1) # with brain
        l_features = []
        # l_features_brain = [] # with brain

        # FF_brain = torch.ones(len_faces_brain,8) # with brain
        # FF_brain = FF_brain.to(self.device) # with brain

        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,index],PF)*(PF >= 0)) # take each feature
        # for index in range(FF_brain.shape[-1]): # with brain
            # l_features_brain.append(torch.take(FF_brain[:,index],PF_brain)*(PF_brain >= 0)) # take each feature # with brain

        x = torch.cat(l_features,dim=2)
        x = torch.cat((x,X),dim=2)
        # print("x.shape", x.shape) # (batch_size, nb_views, 8, 224, 224)  sans depthmap infos
        # print("x.shape", x.shape) # (batch_size, nb_views, 12, 224, 224)  avec depthmap infos
        # x_brain = torch.cat(l_features_brain,dim=2) # with brain
        
        return x, PF
        # return x, PF, x_brain, PF_brain # with brain

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
        V1 = torch.detach(Vo)
        V2 = torch.detach(Vo)
        VFI1 = torch.detach(VFI)
        VFI2 = torch.detach(VFI)
        # V1 = randomstretching(V).to(self.device)            #random stretching
        # V2 = randomstretching(V).to(self.device)
        # VFI1 = randomstretching(VFI).to(self.device)
        # VFI2 = randomstretching(VFI).to(self.device)
        #just for now
        # V1 = randomrot(V1).to(self.device)
        # V2 = randomrot(V2).to(self.device)
        # VFI1 = randomrot(VFI1).to(self.device)
        # VFI2 = randomrot(VFI2).to(self.device)

        # condition = True
        # x, proj_test, x1, x2 = self((V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, condition))
        # proj_test, x1, x2 = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, condition))
        # x1_b = x1[:self.batch_size].to(self.device)
        # x2_b = x2[:self.batch_size].to(self.device)
        # proj_test_b = proj_test[:self.batch_size].to(self.device)
        # labels_b = labels[:self.batch_size].to(self.device)
        # x1_t = x1[self.batch_size:].to(self.device)
        # x2_t = x2[self.batch_size:].to(self.device)
        # proj_test_t = proj_test[self.batch_size:].to(self.device)
        # proj_test = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, condition))
        proj_test, x_class = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI))
        labels_b = labels[:self.batch_size].to(self.device)
        x1_t = proj_test[self.batch_size:].to(self.device)
        x2_t = proj_test[self.batch_size:].to(self.device)
        # loss_contrastive = self.loss_contrastive(x1, x2) # Simclr between the two augmentations of the original data
        lights = self.lights.to(self.device)
        add_noise = False
        if add_noise:
            noise = gauss_law = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1]))
            lights += noise
            lights = lights / torch.norm(lights, dim=1, keepdim=True)

        # loss_contrastive = self.loss_contrastive(lights[labels_b],proj_test) # Simclr between the two augmentations of the original data
        loss_contrastive_bundle = 1 - self.loss_cossine(lights[labels_b],proj_test)# + 1 - self.loss_cossine(self.lights[labels_b],x2_b) + 1 - self.loss_cossine(x1_b,x2_b)
        loss_cross_entropy = self.loss_cross_entropy(x_class, labels_b)
        # loss_ts_ss, diag_loss_ts_ss = TS_SS(lights[labels_b].cpu(),proj_test.cpu()).forward()
        # r = torch.randperm(int(proj_test.shape[0]))
        # proj_test_r = proj_test[r].to(self.device)
        # labels_b_r = labels_b[r].to(self.device)
        # loss_contrastive_bundle_repulse = self.loss_cossine(proj_test,proj_test_r)#*torch.where(labels_b!=labels_b_r)
        # w = (labels_b!=labels_b_r).to(torch.float32)
        # loss_contrastive_bundle_repulse = loss_contrastive_bundle_repulse*w
        # closest_lights = self.closest_lights.to(self.device)
        # loss_closest_point = self.loss_cossine(lights[closest_lights[labels_b].long()],proj_test)
        lam = 1
        loss_align_uniformity = align_loss(lights[labels_b], proj_test) + 0.1*lam * (uniform_loss(proj_test) + uniform_loss(lights[labels_b])) / 2 
        # dist = torch.sqrt(torch.sum((lights[labels_b] - proj_test) ** 2, dim=1))
        # z = (dist < 0.5).to(torch.float32)
        # loss_dist = loss_contrastive_bundle * z

        if x1_t.shape[0] > 0:   # if tractography fibers are in the batch
            r = torch.randperm(int(x2_t.shape[0]))
            x2_t_r = x2_t[r].to(self.device)

            loss_contrastive_tractography = 1 - self.loss_cossine(x1_t, x2_t) # 1 - cossine(Ft1,Ft2)
            loss_contrastive_shuffle = self.loss_cossine(x1_t, x2_t_r)  # 1 - cossine(Ft1,Ft2_shuffled)
            loss_contrastive_tractography = torch.sum(loss_contrastive_tractography).to(self.device)
            loss_contrastive_shuffle = torch.sum(loss_contrastive_shuffle).to(self.device)

            X1_T = x1_t.unsqueeze(1).repeat(1,self.lights.shape[0],1).to(self.device) #(7,57,128)
            LT = self.lights.unsqueeze(0).repeat(x1_t.shape[0],1,1).to(self.device) #(7,57,128)
            loss_for_best = self.loss_cossine_dim2(LT, X1_T) #(7,57)        #compute the cosinesimilarity between each tractography fiber and each cluster
            topk_i = torch.topk(loss_for_best, 5,dim = 1).indices #(7,5)     #get the 5 closest clusters for each tractography fiber
            Nl = [ i.item() for i in torch.randint(-5,0,(7,))]  #randomly choose one of the 5 closest clusters for each tractography fiber
            topk_i_choosen=[topk_i[i,Nl[i]].item() for i in range(len(Nl))] #get the index of the choosen cluster for each tractography fiber
            lights_topk = self.lights[topk_i_choosen].to(self.device) #(7,128)
            loss_tract_cluster = self.loss_cossine(lights_topk, x1_t) #(7,57)
            loss_tract_cluster = torch.sum(loss_tract_cluster).to(self.device)

            Loss_combine = loss_contrastive_bundle + loss_contrastive_tractography + loss_contrastive_shuffle + loss_tract_cluster
        else:
            Loss_combine = 0#0loss_contrastive_bundle #+ loss_closest_point#+ 0.5*loss_contrastive_bundle_repulse #+ loss_contrastive_bundle_repulse
                            # sum                       # mean
        Loss_combine = loss_align_uniformity + loss_cross_entropy #+ loss_contrastive
        # Loss_combine = loss_align_uniformity #+ loss_contrastive
        # diag_loss_ts_ss = diag_loss_ts_ss.requires_grad_(True)
        # Loss_combine = torch.sum(diag_loss_ts_ss) #+ loss_cross_entropy
        self.log('train_loss', Loss_combine.item(), batch_size=self.batch_size)
        # print("accuracy", self.train_accuracy(x, labels))
        # self.log('train_accuracy', self.train_accuracy, batch_size=self.batch_size)

        # return loss_contrastive
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
        # V1 = randomstretching(V).to(self.device)            #random stretching of the fiber
        # V2 = randomstretching(V).to(self.device)
        # VFI1 = randomstretching(VFI).to(self.device)
        # VFI2 = randomstretching(VFI).to(self.device)
        #just for now
        # V1 = randomrot(V1).to(self.device)
        # V2 = randomrot(V2).to(self.device)
        # VFI1 = randomrot(VFI1).to(self.device)
        # VFI2 = randomrot(VFI2).to(self.device)
        V1 = torch.detach(Vo)
        V2 = torch.detach(Vo)
        VFI1 = torch.detach(VFI)
        VFI2 = torch.detach(VFI)
        # condition = True
        # x, proj_test, x1, x2 = self((V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, condition))    #.shape = (batch_size, num classes)
        # proj_test, x1, x2 = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, condition))    #.shape = (batch_size, num classes)
        # x1_b = x1[:self.batch_size].to(self.device)
        # x2_b = x2[:self.batch_size].to(self.device)
        # proj_test_b = proj_test[:self.batch_size].to(self.device)
        # labels_b = labels[:self.batch_size].to(self.device)
        # x1_t = x1[self.batch_size:].to(self.device)
        # x2_t = x2[self.batch_size:].to(self.device)
        # proj_test_t = proj_test[self.batch_size:].to(self.device)

        # proj_test = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, condition))    #.shape = (batch_size, num classes)
        proj_test, x_class = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI))
        labels_b = labels[:self.batch_size].to(self.device)
        x1_t = proj_test[self.batch_size:].to(self.device)
        x2_t = proj_test[self.batch_size:].to(self.device)
        # loss_contrastive = self.loss_contrastive(x1, x2) # Simclr between the two augmentations of the original data
        lights= self.lights.to(self.device)

        add_noise = False
        if add_noise:
            noise = gauss_law = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1]))
            lights += noise
            lights = lights / torch.norm(lights, dim=1, keepdim=True)
        # loss_contrastive = self.loss_contrastive(lights[labels_b],proj_test) # Simclr between the two augmentations of the original data
        loss_contrastive_bundle = 1 - self.loss_cossine(lights[labels_b],proj_test)# + 1 - self.loss_cossine(self.lights[labels_b],x2_b) + 1 - self.loss_cossine(x1_b,x2_b)
        loss_cross_entropy = self.loss_cross_entropy(x_class, labels_b) # return one value
        # loss_ts_ss, diag_loss_ts_ss = TS_SS(lights[labels_b].cpu(),proj_test.cpu()).forward()
        # r = torch.randperm(int(proj_test.shape[0]))
        # proj_test_r = proj_test[r].to(self.device)
        # labels_b_r = labels_b[r].to(self.device)
        # loss_contrastive_bundle_repulse = self.loss_cossine(proj_test,proj_test_r)#*torch.where(labels_b!=labels_b_r)
        # w = (labels_b!=labels_b_r).to(torch.float32)
        # loss_contrastive_bundle_repulse = loss_contrastive_bundle_repulse*w
        # closest_lights = self.closest_lights.to(self.device)
        # loss_closest_point = self.loss_cossine(lights[closest_lights[labels_b].long()],proj_test)
        lam = 1
        loss_align_uniformity = align_loss(lights[labels_b], proj_test) + 0.1*lam * (uniform_loss(proj_test) + uniform_loss(lights[labels_b])) / 2 
        # dist = torch.sqrt(torch.sum((lights[labels_b] - proj_test) ** 2, dim=1))
        # z = (dist < 0.5).to(torch.float32)
        # loss_dist = loss_contrastive_bundle * z

        if x1_t.shape[0] > 0:   # if tractography fibers are in the batch
            r = torch.randperm(int(x2_t.shape[0]))
            x2_t_r = x2_t[r].to(self.device)

            loss_contrastive_tractography = 1 - self.loss_cossine(x1_t, x2_t) # 1 - cossine(Ft1,Ft2)
            loss_contrastive_shuffle = self.loss_cossine(x1_t, x2_t_r)  # 1 - cossine(Ft1,Ft2_shuffled)
            loss_contrastive_tractography = torch.sum(loss_contrastive_tractography)
            loss_contrastive_shuffle = torch.sum(loss_contrastive_shuffle)

            X1_T = x1_t.unsqueeze(1).repeat(1,self.lights.shape[0],1).to(self.device) #(7,57,128)
            LT = self.lights.unsqueeze(0).repeat(x1_t.shape[0],1,1).to(self.device)  #(7,57,128)
            loss_for_best = self.loss_cossine_dim2(LT, X1_T) #(7,57)    #compute the cosinesimilarity between each tractography fiber and each cluster
            topk_i = torch.topk(loss_for_best, 5,dim = 1).indices       #get the 5 best clusters for each tractography fiber
            Nl = [ i.item() for i in torch.randint(-5,0,(7,))]          #get a random number between -5 and 0 for each tractography fiber
            topk_i_choosen=[topk_i[i,Nl[i]].item() for i in range(len(Nl))] #get the cluster corresponding to the random number
            lights_topk = self.lights[topk_i_choosen].to(self.device) #(7,128)
            loss_tract_cluster = self.loss_cossine(lights_topk, x1_t) #(7,57)
            loss_tract_cluster = torch.sum(loss_tract_cluster).to(self.device)

            Loss_combine = loss_contrastive_bundle + loss_contrastive_tractography + loss_contrastive_shuffle + loss_tract_cluster
        else:
            Loss_combine = 0#loss_contrastive_bundle #+ loss_closest_point#+ 0.5*loss_contrastive_bundle_repulse #+ loss_contrastive_bundle_repulse
        Loss_combine = loss_align_uniformity + loss_cross_entropy #+ loss_contrastive
        # Loss_combine = loss_align_uniformity #+ loss_contrastive
        self.log('val_loss', Loss_combine.item(), batch_size=self.batch_size)
        # predictions = torch.argmax(x, dim=1)
        # self.val_accuracy(predictions.reshape(-1,1), labels.reshape(-1,1))
        
        # self.log('val_accuracy', self.val_accuracy, batch_size=self.batch_size)

        # return loss

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
        # V1 = randomstretching(V).to(self.device)            #stretching
        # V2 = randomstretching(V).to(self.device)
        # VFI1 = randomstretching(VFI).to(self.device)
        # VFI2 = randomstretching(VFI).to(self.device)
        #just for now
        # V1 = randomrot(V1).to(self.device)
        # V2 = randomrot(V2).to(self.device)
        # VFI1 = randomrot(VFI1).to(self.device)
        # VFI2 = randomrot(VFI2).to(self.device)
        V1 = torch.detach(Vo)
        V2 = torch.detach(Vo)
        VFI1 = torch.detach(VFI)
        VFI2 = torch.detach(VFI)
        # condition = False
        # x,  proj_test, x1, x2 = self((V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, condition))
        # proj_test, x1, x2 = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, condition))
        # x1_b = x1[:self.batch_size].to(self.device)
        # x2_b = x2[:self.batch_size].to(self.device)
        # proj_test_b = proj_test[:self.batch_size].to(self.device)
        # labels_b = labels[:self.batch_size].to(self.device)
        # x1_t = x1[self.batch_size:].to(self.device)
        # x2_t = x2[self.batch_size:].to(self.device)
        # proj_test_t = proj_test[self.batch_size:].to(self.device)
        # proj_test = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, condition))
        proj_test, x_class = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI))
        tot = np.random.randint(0,1000000)
        labels2 = labels.unsqueeze(dim = 1)
        data_lab = data_lab.unsqueeze(dim = 1)
        proj_test_save = torch.cat((proj_test, labels2, name_labels, data_lab), dim=1)
        lab = np.array(torch.unique(labels).cpu())
        torch.save(proj_test_save, f"/CMF/data/timtey/results_contrastive_learning_062623/proj_test_{lab[-1]}_{tot}.pt")
        # loss_contrastive = self.loss_contrastive(x1, x2) # Simclr between the two augmentations of the original data
        # lights = self.lights.to(self.device)
        # loss_contrastive_bundle = 1 - self.loss_cossine(lights[labels_b],proj_test)# + 1 - self.loss_cossine(self.lights[labels_b],x2_b) + 1 - self.loss_cossine(x1_b,x2_b)

        # r = torch.randperm(int(proj_test.shape[0]))
        # proj_test_r = proj_test[r].to(self.device)
        # labels_b_r = labels_b[r].to(self.device)
        # loss_contrastive_bundle_repulse = self.loss_cossine(proj_test,proj_test_r)#*torch.where(labels_b!=labels_b_r)
        # w = (labels_b!=labels_b_r).to(torch.float32)
        # loss_contrastive_bundle_repulse = loss_contrastive_bundle_repulse*w
        
        # if x1_t.shape[0] > 0:   # if tractography fibers are in the batch
        #     r = torch.randperm(int(x2_t.shape[0]))
        #     x2_t_r = x2_t[r].to(self.device)

        #     loss_contrastive_tractography = 1 - self.loss_cossine(x1_t, x2_t) # 1 - cossine(Ft1,Ft2)
        #     loss_contrastive_shuffle = self.loss_cossine(x1_t, x2_t_r)  # 1 - cossine(Ft1,Ft2_shuffled)
        #     loss_contrastive_tractography = torch.sum(loss_contrastive_tractography)
        #     loss_contrastive_shuffle = torch.sum(loss_contrastive_shuffle)

        #     X1_T = x1_t.unsqueeze(1).repeat(1,self.lights.shape[0],1).to(self.device) #(7,57,128)
        #     LT = self.lights.unsqueeze(0).repeat(x1_t.shape[0],1,1).to(self.device) #(7,57,128)
        #     loss_for_best = self.loss_cossine_dim2(LT, X1_T) #(7,57)
        #     topk_i = torch.topk(loss_for_best, 5,dim = 1).indices
        #     Nl = [ i.item() for i in torch.randint(-5,0,(7,))]
        #     topk_i_choosen=[topk_i[i,Nl[i]].item() for i in range(len(Nl))]
        #     lights_topk = self.lights[topk_i_choosen].to(self.device) #(7,128)
        #     loss_tract_cluster = self.loss_cossine(lights_topk, x1_t) #(7,57)
        #     loss_tract_cluster = torch.sum(loss_tract_cluster).to(self.device)

        #     Loss_combine = loss_contrastive_bundle + loss_contrastive_tractography + loss_contrastive_shuffle + loss_tract_cluster
        # else:
        #     Loss_combine = loss_contrastive_bundle + loss_contrastive_bundle_repulse #+ loss_contrastive_bundle_repulse
        # Loss_combine = torch.sum(Loss_combine)
        # self.log('test_loss', Loss_combine.item(), batch_size=self.batch_size)
        '''
        # predictions = torch.argmax(x, dim=1)
        
        # self.log('test_accuracy', self.val_accuracy, batch_size=self.batch_size)
        # output = [predictions, labels]
        # return output
        '''

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
    









# class Fly_by_CNN_contrastive_tractography_labeled_2(pl.LightningModule):
#     def __init__(self, contrastive, radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, verts_left, faces_left, verts_right, faces_right, learning_rate=0.001):
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = models.resnet18(pretrained=True)
#         self.model.fc = nn.Linear(512, num_classes)
#         self.loss = nn.CrossEntropyLoss()
#         self.loss_contrastive = ContrastiveLoss(batch_size)
#         self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
#         self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
#         self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
#         self.contrastive = contrastive
#         self.radius = radius
#         self.ico_lvl = ico_lvl
#         self.dropout_lvl = dropout_lvl
#         self.image_size = 224
#         self.batch_size = 2*batch_size ###be careful
#         self.weights = weights
#         self.num_classes = num_classes
#         self.verts_left = verts_left
#         self.faces_left = faces_left
#         self.verts_right = verts_right
#         self.faces_right = faces_right
#         #ico_sphere, _a, _v = utils.RandomRotation(utils.CreateIcosahedron(self.radius, ico_lvl))
#         ico_sphere = utils.CreateIcosahedron(self.radius, ico_lvl)
#         ico_sphere_verts, ico_sphere_faces, self.ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
#         self.ico_sphere_verts = ico_sphere_verts
#         self.ico_sphere_faces = ico_sphere_faces
#         self.ico_sphere_edges = np.array(self.ico_sphere_edges)
#         R=[]
#         T=[]
#         for coords_cam in self.ico_sphere_verts.tolist():
#             camera_pos = torch.FloatTensor([coords_cam])
#             R_current = look_at_rotation(camera_pos)
#             T_current = -torch.bmm(R_current.transpose(1,2), camera_pos[:,:,None])[:,:,0]
#             R.append(R_current)
#             T.append(T_current)

#         self.R = torch.cat(R)
#         self.T = torch.cat(T)
#         if contrastive:
#             self.R = self.R.to(torch.float32)
#             self.T = self.T.to(torch.float32)
        
#         efficient_net = models.resnet18(pretrained = True)    ### maybe use weights instead of pretrained

#         if contrastive:
#             efficient_net.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#depthmap
#         else:
#             efficient_net.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#depthmap
#         efficient_net.fc = Identity()
 

#         self.drop = nn.Dropout(p=dropout_lvl)
#         self.TimeDistributed = TimeDistributed(efficient_net)

#         self.WV = nn.Linear(512, 256)

#         self.linear = nn.Linear(256, 256)

#         #######
#         output_size = self.TimeDistributed.module.inplanes
#         conv2d = nn.Conv2d(512, 256, kernel_size=(3,3),stride=2,padding=0) 
#         self.IcosahedronConv2d = IcosahedronConv2d(conv2d,self.ico_sphere_verts,self.ico_sphere_edges)
#         self.pooling = AvgPoolImages(nbr_images=12) #change if we want brains 24 with brains
#         #######

#         #conv2dForQuery = nn.Conv2d(1280, 1280, kernel_size=(3,3),stride=2,padding=0) #1280,512
#         #conv2dForValues = nn.Conv2d(512, 512, kernel_size=(3,3),stride=2,padding=0)  #512,512
#         #self.IcosahedronConv2dForQuery = IcosahedronConv2d(conv2dForQuery,self.ico_sphere_verts,self.ico_sphere_edges)
#         #self.IcosahedronConv2dForValues = IcosahedronConv2d(conv2dForValues,self.ico_sphere_verts,self.ico_sphere_edges)
        
#         #######
#         self.Attention = SelfAttention(512, 128)
#         self.Attention2 = SelfAttention(512, 128)
#         self.WV2 = nn.Linear(512, 512)
#         self.WV3 = nn.Linear(512, 512)
#         self.Attention3 = SelfAttention(512,256)
#         #######

#         self.Classification = nn.Linear(512, num_classes) #256, if just fiber normalized by brain, but 512 if fiber normalized by fiber and fiber normalized by brain
#         self.projection = nn.Sequential(
#            nn.Linear(in_features=512, out_features=512),
#            nn.BatchNorm1d(512),
#            nn.ReLU(),
#            nn.Linear(in_features=512, out_features=128),
#            nn.BatchNorm1d(128),
#        )
        
#         self.loss = nn.CrossEntropyLoss()
#         self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=57)
#         self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=57)

#         self.cameras = FoVPerspectiveCameras()

#         # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
#         raster_settings = RasterizationSettings(
#             image_size=self.image_size, 
#             blur_radius=0, 
#             faces_per_pixel=1, 
#             max_faces_per_bin=100000
#         )

#         lights = AmbientLights()
#         rasterizer = MeshRasterizer(
#                 cameras=self.cameras, 
#                 raster_settings=raster_settings
#             )

#         self.phong_renderer = MeshRenderer(
#             rasterizer=rasterizer,
#             shader=HardPhongShader(cameras=self.cameras, lights=lights)
#         )
#         self.phong_renderer_brain = MeshRenderer(
#             rasterizer=rasterizer,
#             shader=HardPhongShader(cameras=self.cameras, lights=lights)
#         )

#         self.loss_train = nn.CrossEntropyLoss(weight = self.weights[0])
#         self.loss_val = nn.CrossEntropyLoss(weight = self.weights[1])
#         self.loss_test = nn.CrossEntropyLoss(weight = self.weights[2])

#         self.lights = pd.read_pickle(r'Lights_good.pickle')

#         self.loss_cossine = nn.CosineSimilarity()

#     def forward(self, x):
#         V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB, condition = x
        
#         x, PF = self.render(V,F,FF)
#         X1, PF1 = self.render(V1,F,FF)
#         X2, PF2 = self.render(V2,F,FF)

#         x_fiber, PF_fiber = self.render(VFI,FFI,FFFI)
#         X1_fiber, PF1_fiber = self.render(VFI1,FFI,FFFI)
#         X2_fiber, PF2_fiber = self.render(VFI2,FFI,FFFI)

#         brain_help = False
#         if brain_help:
#             x_brain, PF_brain = self.render(VB, FB, FFB) # with brain
#             X1_brain, PF1_brain = self.render(VB, FB, FFB) # with brain
#             X2_brain, PF2_brain = self.render(VB, FB, FFB) # with brain

#         x_test = torch.cat((x,x_fiber),1)
#         X1 = torch.cat((X1,X1_fiber),1)
#         X2 = torch.cat((X2,X2_fiber),1)
#         query = self.TimeDistributed(x)
#         query_test = self.TimeDistributed(x_test)
#         query_1 = self.TimeDistributed(X1)
#         query_2 = self.TimeDistributed(X2)
#         values_1 = self.WV3(query_1)
#         values_2 = self.WV3(query_2)
#         values_test = self.WV3(query_test)
#         x_a_1, w_a_1 = self.Attention3(query_1, values_1)
#         x_a_2, w_a_2 = self.Attention3(query_2, values_2)
#         x_a_test, w_a_test = self.Attention3(query_test, values_test)
#         # taille = [10,24*512]
#         # query_1 = query_1.view(taille)
#         # query_2 = query_2.view(taille)
#         proj1 = self.projection(x_a_1)
#         # print("proj1", proj1.shape) # (bs, 128)
#         proj2 = self.projection(x_a_2)
#         # print("proj2", proj2.shape)
#         proj_test = self.projection(x_a_test)
#         # print("proj_test", proj_test.shape)
#         # print(kdjhfg)
#         query_fiber = self.TimeDistributed(x_fiber)
#         # query_fiber_1 = self.TimeDistributed(X1_fiber)
#         # query_fiber_2 = self.TimeDistributed(X2_fiber)
#         if brain_help:
#             query_brain = self.TimeDistributed(x_brain)
#         # query_brain_1 = self.TimeDistributed(X1_brain)
#         # query_brain_2 = self.TimeDistributed(X2_brain)
#         icoconv2d = True
#         pool = False
#         # print("query",query.shape) #(batch_size, 12, 512)
#         if icoconv2d:
#             # print("query ",query.shape)
#             x= self.IcosahedronConv2d(query)
#             # x1= self.IcosahedronConv2d(query_1)
#             # x2= self.IcosahedronConv2d(query_2)
#             x_fiber = self.IcosahedronConv2d(query_fiber)
#             # x_fiber_1 = self.IcosahedronConv2d(query_fiber_1)
#             # x_fiber_2 = self.IcosahedronConv2d(query_fiber_2)
#             if brain_help:
#                 x_brain = self.IcosahedronConv2d(query_brain)
#             # x_brain_1 = self.IcosahedronConv2d(query_brain_1)
#             # x_brain_2 = self.IcosahedronConv2d(query_brain_2)
#             # print("x ",x.shape) #(batch_size, 12, 256)
#             if pool:
#                 x = self.pooling(x)
#                 x1 = self.pooling(x1)
#                 x2 = self.pooling(x2)
#                 x_fiber = self.pooling(x_fiber)
#                 x_fiber_1 = self.pooling(x_fiber_1)
#                 x_fiber_2 = self.pooling(x_fiber_2)
#                 if brain_help:
#                     x_brain = self.pooling(x_brain)
#                 # x_brain_1 = self.pooling(x_brain_1)
#                 # x_brain_2 = self.pooling(x_brain_2)
#             else:
#                 # print("x ",x.shape) #(batch_size, 12, 256)
#                 x_a =self.linear(x)
#                 # x_a_1 =self.linear(x1)
#                 # x_a_2 =self.linear(x2)
#                 x_a_fiber =self.linear(x_fiber)
#                 # x_a_fiber_1 =self.linear(x_fiber_1)
#                 # x_a_fiber_2 =self.linear(x_fiber_2)
#                 if brain_help:
#                     x_a_brain =self.linear(x_brain)
#                 # x_a_brain_1 =self.linear(x_brain_1)
#                 # x_a_brain_2 =self.linear(x_brain_2)
#                 # print("x ",x_a.shape) #(batch_size, 12, 256)
#                 # print("x ",x.shape) #(batch_size, 12, 256)
#                 # print("x_fiber ",x_a_fiber.shape) #(batch_size, 12, 256)
#                 # print("x_brain ",x_a_brain.shape) #(batch_size, 12, 256)
#             # print("x_a ",x_a.shape)
#         else:
#             values = self.WV(query)
#             x_a, w_a = self.Attention(query, values)
#         # print("x_a ",x_a.shape) #(batch_size,256)
#         # print("x_a_fiber ",x_a_fiber.shape) #(batch_size,256)
#         # print("x_a_brain ",x_a_brain.shape) #(batch_size,256)
#         # x_a_brain  = torch.tile(x_a_brain,(self.batch_size,1))
#         # print("x_a_brain ",x_a_brain.shape) #(batch_size,256)

#         if brain_help:
#             x_a = torch.cat((x_a,x_a_fiber,x_a_brain),2)
#             # x_a_1 = torch.cat((x_a_1,x_a_fiber_1,x_a_brain),2)
#             # x_a_2 = torch.cat((x_a_2,x_a_fiber_2,x_a_brain),2)
#         else:
#             x_a = torch.cat((x_a,x_a_fiber),2)
#             # x_a_1 = torch.cat((x_a_1,x_a_fiber_1),2)
#             # x_a_2 = torch.cat((x_a_2,x_a_fiber_2),2)
#         # print("x_a ",x_a.shape) #(batch_size,12,768)
#         values = self.WV2(x_a)
#         # print("values ",values.shape) #(batch_size,12,768)
#         x_a, w_a = self.Attention2(x_a, values)
#         # print("x_a ",x_a.shape) #(batch_size,256) 768 si attention2
#         x_a = self.drop(x_a)
#         # print("x_a ",x_a.shape) #(batch_size,256) 768 si attention2
#         x = self.Classification(x_a)
#         # print("x classsif ",x.shape)  #(batch_size,nb class)
        
#         return x, proj_test, proj1, proj2
#     """
#     def forward(self, x):
#         V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB, condition = x
#         brain_help = False
#         if condition:
#             X1, PF1 = self.render(V1,F,FF)
#             X2, PF2 = self.render(V2,F,FF)
#             X1_fiber, PF1_fiber = self.render(VFI1,FFI,FFFI)
#             X2_fiber, PF2_fiber = self.render(VFI2,FFI,FFFI)
#             if brain_help:
#                 X1_brain, PF1_brain = self.render(VB, FB, FFB)
#                 X2_brain, PF2_brain = self.render(VB, FB, FFB)
#             X1 = torch.cat((X1,X1_fiber),1)
#             X2 = torch.cat((X2,X2_fiber),1)
#             query_1 = self.TimeDistributed(X1)
#             query_2 = self.TimeDistributed(X2)
#             values_1 = self.WV3(query_1)
#             values_2 = self.WV3(query_2)
#             x_a_1, w_a_1 = self.Attention3(query_1, values_1)
#             x_a_2, w_a_2 = self.Attention3(query_2, values_2)
#             proj1 = self.projection(x_a_1)
#             proj2 = self.projection(x_a_2)
#             classique = False
#             if classique:
#                 query_fiber = self.TimeDistributed(x_fiber)
#                 query = self.TimeDistributed(x)
#                 icoconv2d = True
#                 pool = False
#                 if icoconv2d:
#                     x = self.IcosahedronConv2d(query)
#                     x_fiber = self.IcosahedronConv2d(query_fiber)
#                     if brain_help:
#                         x_brain = self.IcosahedronConv2d(query_brain)
#                     if pool:
#                         x = self.pool(x)
#                         x_fiber = self.pool(x_fiber)
#                         if brain_help:
#                             x_brain = self.pool(x_brain)
#                     else:
#                         x_a = self.linear(x)
#                         x_a_fiber = self.linear(x_fiber)
#                         if brain_help:
#                             x_a_brain = self.linear(x_brain)
#                 else:
#                     values = self.WV(query)
#                     x_a, w_a = self.Attention(query, values)
#                 if brain_help:
#                     x_a = torch.cat((x_a,x_a_fiber,x_a_brain),2)
#                 else:
#                     x_a = torch.cat((x_a,x_a_fiber),2)
#                 values = self.WV2(x_a)
#                 x_a, w_a = self.Attention2(x_a, values)
#                 x_a = self.drop(x_a)
#                 x = self.Classification(x_a)
#                 return x, proj1, proj2
#             else:
#                 return proj1, proj2
#         else:
#             x, PF = self.render(V,F,FF)
#             x_fiber, PF_fiber = self.render(VFI,FFI,FFFI)
#             if brain_help:
#                 x_brain, PF_brain = self.render(VB, FB, FFB)
#                 query_brain = self.TimeDistributed(x_brain)
#             x_test = torch.cat((x,x_fiber),1)
#             query_test = self.TimeDistributed(x_test)
#             values_test = self.WV3(query_test)
#             x_a_test, w_a_test = self.Attention3(query_test, values_test)
#             proj_test = self.projection(x_a_test)
#             classique = False
#             if classique:
#                 query_fiber = self.TimeDistributed(x_fiber)
#                 query = self.TimeDistributed(x)
#                 icoconv2d = True
#                 pool = False
#                 if icoconv2d:
#                     x = self.IcosahedronConv2d(query)
#                     x_fiber = self.IcosahedronConv2d(query_fiber)
#                     if brain_help:
#                         x_brain = self.IcosahedronConv2d(query_brain)
#                     if pool:
#                         x = self.pool(x)
#                         x_fiber = self.pool(x_fiber)
#                         if brain_help:
#                             x_brain = self.pool(x_brain)
#                     else:
#                         x_a = self.linear(x)
#                         x_a_fiber = self.linear(x_fiber)
#                         if brain_help:
#                             x_a_brain = self.linear(x_brain)
#                 else:
#                     values = self.WV(query)
#                     x_a, w_a = self.Attention(query, values)
#                 if brain_help:
#                     x_a = torch.cat((x_a,x_a_fiber,x_a_brain),2)
#                 else:
#                     x_a = torch.cat((x_a,x_a_fiber),2)
#                 values = self.WV2(x_a)
#                 x_a, w_a = self.Attention2(x_a, values)
#                 x_a = self.drop(x_a)
#                 x = self.Classification(x_a)
#                 return x, proj_test
#             else:
#                 return proj_test



#         '''
#         x, PF = self.render(V,F,FF)
#         X1, PF1 = self.render(V1,F,FF)
#         X2, PF2 = self.render(V2,F,FF)

#         x_fiber, PF_fiber = self.render(VFI,FFI,FFFI)
#         # print("x_fiber", x_fiber.shape)
#         X1_fiber, PF1_fiber = self.render(VFI1,FFI,FFFI)
#         X2_fiber, PF2_fiber = self.render(VFI2,FFI,FFFI)

#         brain_help = False
#         if brain_help:
#             x_brain, PF_brain = self.render(VB, FB, FFB) # with brain
#             X1_brain, PF1_brain = self.render(VB, FB, FFB) # with brain
#             X2_brain, PF2_brain = self.render(VB, FB, FFB) # with brain

#         x_test = torch.cat((x,x_fiber),1)
#         X1 = torch.cat((X1,X1_fiber),1)
#         X2 = torch.cat((X2,X2_fiber),1)
#         # print("X1 ",X1.shape)
#         # print("X2 ",X2.shape)
#         query = self.TimeDistributed(x)
#         query_test = self.TimeDistributed(x_test)
#         # print("query", query.shape)
#         query_1 = self.TimeDistributed(X1)
#         query_2 = self.TimeDistributed(X2)
#         # print("query_1", query_1.shape)
#         # print("query_2", query_2.shape)
#         # print("query_test", query_test.shape)
#         # tada = nn.Linear(in_features=512, out_features=512).to(self.device)
#         # query_1 = tada(query_1)
#         # print("query_1", query_1.shape)
#         # bno1 = nn.BatchNorm1d(512).to(self.device)
#         values_1 = self.WV3(query_1)
#         values_2 = self.WV3(query_2)
#         # print("values_1", values_1.shape)
#         # print("values_2", values_2.shape)
#         values_test = self.WV3(query_test)
#         # print("values_test", values_test.shape)
        
        
#         x_a_1, w_a_1 = self.Attention3(query_1, values_1)
#         x_a_2, w_a_2 = self.Attention3(query_2, values_2)
#         x_a_test, w_a_test = self.Attention3(query_test, values_test)
#         # print("x_a_1", x_a_1.shape)
#         # print("x_a_2", x_a_2.shape)
#         # print("x_a_test", x_a_test.shape)
#         # taille = [10,24*512]
#         # query_1 = query_1.view(taille)
#         # query_2 = query_2.view(taille)
#         # print("query_1", query_1.shape)
#         # print("query_2", query_2.shape)
#         # print("query_1", query_1.shape)
#         # query_1 = bno1(query_1)
#         # print("query_1", query_1.shape)
#         # print("query_2", query_2.shape)
#         proj1 = self.projection(x_a_1)
#         # print("proj1", proj1.shape) # (bs, 128)
#         proj2 = self.projection(x_a_2)
#         # print("proj2", proj2.shape)
#         proj_test = self.projection(x_a_test)
#         # print("proj_test", proj_test.shape)
#         # print(kdjhfg)
#         query_fiber = self.TimeDistributed(x_fiber)
#         # query_fiber_1 = self.TimeDistributed(X1_fiber)
#         # query_fiber_2 = self.TimeDistributed(X2_fiber)
#         if brain_help:
#             query_brain = self.TimeDistributed(x_brain)
#         # query_brain_1 = self.TimeDistributed(X1_brain)
#         # query_brain_2 = self.TimeDistributed(X2_brain)
#         # print(query)
#         icoconv2d = True
#         pool = False
#         # print("querry",query.shape) #(batch_size, 12, 512)
#         if icoconv2d:
#             # print("query ",query.shape)
#             x= self.IcosahedronConv2d(query)
#             # x1= self.IcosahedronConv2d(query_1)
#             # x2= self.IcosahedronConv2d(query_2)
#             x_fiber = self.IcosahedronConv2d(query_fiber)
#             # x_fiber_1 = self.IcosahedronConv2d(query_fiber_1)
#             # x_fiber_2 = self.IcosahedronConv2d(query_fiber_2)
#             if brain_help:
#                 x_brain = self.IcosahedronConv2d(query_brain)
#             # x_brain_1 = self.IcosahedronConv2d(query_brain_1)
#             # x_brain_2 = self.IcosahedronConv2d(query_brain_2)
#             # print("x ",x.shape) #(batch_size, 12, 256)
#             if pool:
#                 x = self.pooling(x)
#                 x1 = self.pooling(x1)
#                 x2 = self.pooling(x2)
#                 x_fiber = self.pooling(x_fiber)
#                 x_fiber_1 = self.pooling(x_fiber_1)
#                 x_fiber_2 = self.pooling(x_fiber_2)
#                 if brain_help:
#                     x_brain = self.pooling(x_brain)
#                 # x_brain_1 = self.pooling(x_brain_1)
#                 # x_brain_2 = self.pooling(x_brain_2)
#             else:
#                 # print("x ",x.shape) #(batch_size, 12, 256)
#                 x_a =self.linear(x)
#                 # x_a_1 =self.linear(x1)
#                 # x_a_2 =self.linear(x2)
#                 x_a_fiber =self.linear(x_fiber)
#                 # x_a_fiber_1 =self.linear(x_fiber_1)
#                 # x_a_fiber_2 =self.linear(x_fiber_2)
#                 if brain_help:
#                     x_a_brain =self.linear(x_brain)
#                 # x_a_brain_1 =self.linear(x_brain_1)
#                 # x_a_brain_2 =self.linear(x_brain_2)
#                 # print("x ",x_a.shape) #(batch_size, 12, 256)
#                 # print("x ",x.shape) #(batch_size, 12, 256)
#                 # print("x_fiber ",x_a_fiber.shape) #(batch_size, 12, 256)
#                 # print("x_brain ",x_a_brain.shape) #(batch_size, 12, 256)
#             # print("x_a ",x_a.shape)
#         else:
#             values = self.WV(query)
#             x_a, w_a = self.Attention(query, values)
#         # print("x_a ",x_a.shape) #(batch_size,256)
#         # print("x_a_fiber ",x_a_fiber.shape) #(batch_size,256)
#         # print("x_a_brain ",x_a_brain.shape) #(batch_size,256)
#         # x_a_brain  = torch.tile(x_a_brain,(self.batch_size,1))
#         # print("x_a_brain ",x_a_brain.shape) #(batch_size,256)

#         if brain_help:
#             x_a = torch.cat((x_a,x_a_fiber,x_a_brain),2)
#             # x_a_1 = torch.cat((x_a_1,x_a_fiber_1,x_a_brain),2)
#             # x_a_2 = torch.cat((x_a_2,x_a_fiber_2,x_a_brain),2)
#         else:
#             x_a = torch.cat((x_a,x_a_fiber),2)
#             # x_a_1 = torch.cat((x_a_1,x_a_fiber_1),2)
#             # x_a_2 = torch.cat((x_a_2,x_a_fiber_2),2)
#         # print("x_a ",x_a.shape) #(batch_size,12,768)
#         # print(szdjgf)
#         values = self.WV2(x_a)
#         # values_1 = self.WV2(x_a_1)
#         # values_2 = self.WV2(x_a_2)
#         # print("values ",values.shape) #(batch_size,12,768)
#         x_a, w_a = self.Attention2(x_a, values)
#         # x_a_1, w_a_1 = self.Attention2(x_a_1, values_1)
#         # x_a_2, w_a_2 = self.Attention2(x_a_2, values_2)
#         # print("x_a ",x_a.shape) #(batch_size,256) 768 si attention2
#         x_a = self.drop(x_a)
#         # x_a_1 = self.drop(x_a_1)
#         # x_a_2 = self.drop(x_a_2)
#         # print("x_a ",x_a.shape) #(batch_size,256) 768 si attention2
#         x = self.Classification(x_a)
#         # x1 = self.Classification(x_a_1)
#         # x2 = self.Classification(x_a_2)
#         # print("x classsif ",x.shape)  #(batch_size,nb class)
        
#         return x, proj_test, proj1, proj2
#         '''
#         """
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
#         return optimizer

#     def render(self, V, F, FF):

#         textures = TexturesVertex(verts_features=torch.ones(V.shape))
#         V = V.to(torch.float32)
#         F = F.to(torch.float32)
#         V = V.to(self.device)
#         F = F.to(self.device)
#         textures = textures.to(self.device)
#         meshes_fiber = Meshes(
#             verts=V,   
#             faces=F, 
#             textures=textures
#         )

#         # t_left = [0]*self.verts_left.shape[0] # with brain
#         # self.verts_left = self.verts_left.to(self.device) # with brain
#         # self.verts_right = self.verts_right.to(self.device) # with brain
#         # t_left[:] = np.sqrt(self.verts_left[:,0]**2 + self.verts_left[:,1]**2 + self.verts_left[:,2]**2) # with brain
#         # for i in range(self.verts_left.shape[0]): # with brain
#             # t_left.append(np.sqrt(self.verts_left[i,0]**2 + self.verts_left[i,1]**2 + self.verts_left[i,2]**2)) # with brain
#         # t_right = [0]*self.verts_right.shape[0] # with brain
#         # t_right[:] = np.sqrt(self.verts_right[:,0]**2 + self.verts_right[:,1]**2 + self.verts_right[:,2]**2) # with brain
#         # for i in range(self.verts_right.shape[0]): # with brain
#             # t_right.append(np.sqrt(self.verts_right[i,0]**2 + self.verts_right[i,1]**2 + self.verts_right[i,2]**2)) # with brain
#         # t_left = torch.tensor(t_left).unsqueeze(1) # with brain
#         # t_right = torch.tensor(t_right).unsqueeze(1) # with brain
#         # t_left = t_left.unsqueeze(0) # with brain
#         # t_right = t_right.unsqueeze(0) # with brain
#         # t_left = t_left.to(self.device) # with brain
#         # t_right = t_right.to(self.device) # with brain
#         # texture_left = TexturesVertex(verts_features= t_left) # with brain
#         # texture_right = TexturesVertex(verts_features= t_right) # with brain

#         phong_renderer = self.phong_renderer.to(self.device)
#         # phong_renderer_brain = self.phong_renderer_brain.to(self.device)
#         meshes_fiber = meshes_fiber.to(self.device)
#         PF = []
#         X = []
#         # PF_brain = []
#         for i in range(len(self.R)):
#             R = self.R[i][None].to(self.device)
#             T = self.T[i][None].to(self.device)
#             pixel_to_face, images = GetView(meshes_fiber,phong_renderer,R,T)
#             # pixel_to_face_brain = GetView(meshes_brain,phong_renderer_brain,R,T) # with brain
#             images = images.to(self.device)
#             PF.append(pixel_to_face.unsqueeze(dim=1))
#             X.append(images.unsqueeze(dim=1))
#             # PF_brain.append(pixel_to_face_brain.unsqueeze(dim=1)) # with brain

#         PF = torch.cat(PF,dim=1)
#         X = torch.cat(X,dim=1)
#         X = X[:,:,3,:,:] # the last one who has the picture in black and white of depth
#         X = X.unsqueeze(dim=2)
        
#         # print("PF", PF.shape) # (batch_size, nb_views, 1, 224, 224)
#         # print("X", X.shape) # (batch_size, nb_views, 4, 224, 224)
#         # PF_brain = torch.cat(PF_brain,dim=1) # with brain

#         l_features = []
#         # l_features_brain = [] # with brain

#         # FF_brain = torch.ones(len_faces_brain,8) # with brain
#         # FF_brain = FF_brain.to(self.device) # with brain

#         for index in range(FF.shape[-1]):
#             l_features.append(torch.take(FF[:,index],PF)*(PF >= 0)) # take each feature
#         # for index in range(FF_brain.shape[-1]): # with brain
#             # l_features_brain.append(torch.take(FF_brain[:,index],PF_brain)*(PF_brain >= 0)) # take each feature # with brain

#         x = torch.cat(l_features,dim=2)
#         x = torch.cat((x,X),dim=2)
#         # print("x.shape", x.shape) # (batch_size, nb_views, 8, 224, 224)  sans depthmap infos
#         # print("x.shape", x.shape) # (batch_size, nb_views, 12, 224, 224)  avec depthmap infos
#         # x_brain = torch.cat(l_features_brain,dim=2) # with brain
#         # x === mes 12 images de 224*224*8
#         #x.shape(batch_size,nb_cameras,8,224,224)
#         # x_brain_f = torch.tile(x_brain,(x.shape[0],1,1,1,1)) # with brain
        
#         # x = torch.cat((x,x_brain_f),dim=1) # with brain
#         return x, PF
#         # return x, PF, x_brain, PF_brain # with brain

#     def training_step(self, train_batch, train_batch_idx):

#         V, F, FF, labels, Fiber_infos = train_batch
#         V = V.to(self.device)
#         F = F.to(self.device)
#         FF = FF.to(self.device)
#         VFI = torch.clone(V)
#         VFI = VFI.to(self.device)
#         FFI = torch.clone(F)
#         FFI = FFI.to(self.device)
#         FFFI = torch.clone(FF)
#         FFFI = FFFI.to(self.device)
#         labels = labels.to(self.device)
#         labelsFI = torch.clone(labels)
#         labelsFI = labelsFI.to(self.device)
#         vfbounds = []
#         sample_min_max = []
#         data_lab = []
#         name_labels = []
        
#         for z in range(len(Fiber_infos)):
#             vfbounds += [Fiber_infos[z][0]]
#             sample_min_max += [Fiber_infos[z][1]]
#             data_lab += Fiber_infos[z][2]
#             name_labels += Fiber_infos[z][3]

#         sample_id = []
#         for w in range(len(name_labels)):
#             sample_id.append(name_labels[w][0])
#         VB = []
#         FB = []
#         FFB = []
#         for i in range(len(sample_id)):
#             VBi = torch.load(f"brain_structures/verts_brain_{sample_id[i]}.pt")
#             FBi = torch.load(f"brain_structures/faces_brain_{sample_id[i]}.pt")
#             FFBi = torch.load(f"brain_structures/face_features_brain_{sample_id[i]}.pt")
#             VBi = VBi.to(self.device)
#             FBi = FBi.to(self.device)
#             FFBi = FFBi.to(self.device)
#             VB.append(VBi)
#             FB.append(FBi)
#             FFB.append(FFBi)
        
#         VB = pad_sequence(VB, batch_first=True, padding_value=0)
#         FB = pad_sequence(FB, batch_first=True, padding_value=-1)
#         FFB = torch.cat(FFB)
#         VB = VB.to(self.device)
#         FB = FB.to(self.device)
#         FFB = FFB.to(self.device)
#         ###
#         # change fibers points with stretching, rotation, translation twices for the V and twices for the VFI
#         V = transformation_verts(V, sample_min_max)
#         VFI = transformation_verts_by_fiber(VFI, vfbounds)


#         V1 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
#         V2 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
#         VFI1 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)
#         VFI2 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)

#         V1 = V1.to(self.device)
#         V2 = V2.to(self.device)
#         VFI1 = VFI1.to(self.device)
#         VFI2 = VFI2.to(self.device)
#         for i in range(V1.shape[0]):
#             V1[i] = randomrotation(V1[i])
#             V2[i] = randomrotation(V2[i])
#             VFI1[i] = randomrotation(VFI1[i])
#             VFI2[i] = randomrotation(VFI2[i])
#         condition = True
#         x, proj_test, x1, x2 = self((V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB, condition))
#         x1_b = torch.tensor([]).to(self.device)
#         x1_t = torch.tensor([]).to(self.device)
#         x2_b = torch.tensor([]).to(self.device)
#         x2_t = torch.tensor([]).to(self.device)
#         proj_test_b = torch.tensor([]).to(self.device)
#         proj_test_t = torch.tensor([]).to(self.device)

#         labels_b = []
#         for i in range(len(data_lab)):
#             if data_lab[i] == 0 :
#                 x1_b = torch.cat((x1_b, x1[i].unsqueeze(0)))
#                 x2_b = torch.cat((x2_b, x2[i].unsqueeze(0)))
#                 proj_test_b = torch.cat((proj_test_b, proj_test[i].unsqueeze(0)))
#                 labels_b.append(labels[i])
#             else:
#                 x1_t = torch.cat((x1_t, x1[i].unsqueeze(0)))
#                 x2_t = torch.cat((x2_t, x2[i].unsqueeze(0)))
#                 proj_test_t = torch.cat((proj_test_t, proj_test[i].unsqueeze(0)))
#         # loss = self.loss_train(x, labels)
    
#         loss_contrastive = self.loss_contrastive(x1, x2)
#         loss_contrastive_bundle = 0
#         lights = torch.tensor(self.lights).to(self.device)
        
#         for i in range(x1_b.shape[0]):
#             loss_contrastive_bundle += 1-self.loss_cossine(lights[labels_b[i]], x1_b[i].unsqueeze(0)) + 1 - self.loss_cossine(lights[labels_b[i]], x2_b[i].unsqueeze(0)) + 1 - self.loss_cossine(x1_b[i].unsqueeze(0), x2_b[i].unsqueeze(0))
        
#         r = torch.randperm(int(x2_t.shape[0]))
#         x2_t_r = x2_t[r].to(self.device)
        
#         loss_contrastive_tractography = 1 - self.loss_cossine(x1_t, x2_t)
#         loss_contrastive_shuffle = self.loss_cossine(x1_t, x2_t_r)
#         loss_contrastive_tractography = torch.sum(loss_contrastive_tractography)
#         loss_contrastive_shuffle = torch.sum(loss_contrastive_shuffle)

#         loss_tract_cluster = 0
#         for j in range(x1_t.shape[0]):
#             loss_tract_cluster_i = torch.tensor([]).to(self.device)
#             for i in range(lights.shape[0]):
#                 loss_tract_cluster_i = torch.cat((loss_tract_cluster_i, self.loss_cossine(lights[i], x1_t[j].unsqueeze(0))))
#             topk = torch.topk(loss_tract_cluster_i, 5)
#             # loss_tract_cluster = torch.argsort(loss_tract_cluster)
#             topk_indices = topk.indices
#             n = random.randint(-5,-1)
#             cluster_k_indices = topk_indices[n].item()
#             loss_tract_cluster += 1-self.loss_cossine(lights[cluster_k_indices], x1_t[j].unsqueeze(0))

#         Loss_combine = loss_contrastive_bundle + loss_contrastive_tractography + loss_contrastive_shuffle + loss_tract_cluster

        
#         self.log('train_loss', Loss_combine.item(), batch_size=self.batch_size)
#         self.log('train_accuracy', self.train_accuracy, batch_size=self.batch_size)

#         # return loss_contrastive
#         return Loss_combine

        
#     def validation_step(self, val_batch, val_batch_idx):
        
#         V, F, FF, labels, Fiber_infos= val_batch
#         V = V.to(self.device)
#         F = F.to(self.device)
#         FF = FF.to(self.device)
#         VFI = torch.clone(V)
#         VFI = VFI.to(self.device)
#         FFI = torch.clone(F)
#         FFI = FFI.to(self.device)
#         FFFI = torch.clone(FF)
#         FFFI = FFFI.to(self.device)
        
#         labels = labels.to(self.device)
#         labelsFI = torch.clone(labels)
#         labelsFI = labelsFI.to(self.device)
        
#         vfbounds = []
#         sample_min_max = []
#         data_lab = []
#         name_labels = []
        
#         for z in range(len(Fiber_infos)):
#             vfbounds += [Fiber_infos[z][0]]
#             sample_min_max += [Fiber_infos[z][1]]
#             data_lab += Fiber_infos[z][2]
#             name_labels += Fiber_infos[z][3]
        
#         sample_id = []
#         for w in range(len(name_labels)):
#             sample_id.append(name_labels[w][0])
#         VB = []
#         FB = []
#         FFB = []
#         for i in range(len(sample_id)):
#             VBi = torch.load(f"brain_structures/verts_brain_{sample_id[i]}.pt")
#             FBi = torch.load(f"brain_structures/faces_brain_{sample_id[i]}.pt")
#             FFBi = torch.load(f"brain_structures/face_features_brain_{sample_id[i]}.pt")
#             VBi = VBi.to(self.device)
#             FBi = FBi.to(self.device)
#             FFBi = FFBi.to(self.device)
#             VB.append(VBi)
#             FB.append(FBi)
#             FFB.append(FFBi)
        
#         VB = pad_sequence(VB, batch_first=True, padding_value=0)
#         FB = pad_sequence(FB, batch_first=True, padding_value=-1)
#         FFB = torch.cat(FFB)
#         VB = VB.to(self.device)
#         FB = FB.to(self.device)
#         FFB = FFB.to(self.device)

#         V = transformation_verts(V, sample_min_max)
#         VFI = transformation_verts_by_fiber(VFI, vfbounds)

#         V1 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
#         V2 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
#         VFI1 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)
#         VFI2 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)
#         V1 = V1.to(self.device)
#         V2 = V2.to(self.device)
#         VFI1 = VFI1.to(self.device)
#         VFI2 = VFI2.to(self.device)
#         for i in range(V1.shape[0]):
#             V1[i] = randomrotation(V1[i])
#             V2[i] = randomrotation(V2[i])
#             VFI1[i] = randomrotation(VFI1[i])
#             VFI2[i] = randomrotation(VFI2[i])
#         condition = True
#         x, proj_test, x1, x2 = self((V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB, condition))    #.shape = (batch_size, num classes)
#         x1_b = torch.tensor([]).to(self.device)
#         x1_t = torch.tensor([]).to(self.device)
#         x2_b = torch.tensor([]).to(self.device)
#         x2_t = torch.tensor([]).to(self.device)
#         proj_test_b = torch.tensor([]).to(self.device)
#         proj_test_t = torch.tensor([]).to(self.device)
#         labels_b = []

#         for i in range(len(data_lab)):
#             if data_lab[i] == 0 :
#                 x1_b = torch.cat((x1_b, x1[i].unsqueeze(0)))
#                 x2_b = torch.cat((x2_b, x2[i].unsqueeze(0)))
#                 proj_test_b = torch.cat((proj_test_b, proj_test[i].unsqueeze(0)))
#                 labels_b.append(labels[i])
#             else:
#                 x1_t = torch.cat((x1_t, x1[i].unsqueeze(0)))
#                 x2_t = torch.cat((x2_t, x2[i].unsqueeze(0)))
#                 proj_test_t = torch.cat((proj_test_t, proj_test[i].unsqueeze(0)))

#         loss_contrastive = self.loss_contrastive(x1, x2)
        
#         loss_contrastive_bundle = 0
#         lights = torch.tensor(self.lights).to(self.device)
        
#         for i in range(x1_b.shape[0]):
#             loss_contrastive_bundle += 1-self.loss_cossine(lights[labels_b[i]], x1_b[i].unsqueeze(0)) + 1 - self.loss_cossine(lights[labels_b[i]], x2_b[i].unsqueeze(0)) + 1 - self.loss_cossine(x1_b[i].unsqueeze(0), x2_b[i].unsqueeze(0))
        
#         r = torch.randperm(int(x2_t.shape[0]))
#         x2_t_r = x2_t[r].to(self.device)
        
#         loss_contrastive_tractography = 1 - self.loss_cossine(x1_t, x2_t)
#         loss_contrastive_shuffle = self.loss_cossine(x1_t, x2_t_r)
#         loss_contrastive_tractography = torch.sum(loss_contrastive_tractography)
#         loss_contrastive_shuffle = torch.sum(loss_contrastive_shuffle)
        
#         loss_tract_cluster = 0
#         for j in range(x1_t.shape[0]):
#             loss_tract_cluster_i = torch.tensor([]).to(self.device)
#             for i in range(lights.shape[0]):
#                 loss_tract_cluster_i = torch.cat((loss_tract_cluster_i, self.loss_cossine(lights[i], x1_t[j].unsqueeze(0))))
#             topk = torch.topk(loss_tract_cluster_i, 5)
#             n = random.randint(-5,-1)
#             topk_indices = topk.indices
#             cluster_k_indices = topk_indices[n].item()
#             loss_tract_cluster += 1-self.loss_cossine(lights[cluster_k_indices], x1_t[j].unsqueeze(0))            
        
#         Loss_combine = loss_contrastive_bundle + loss_contrastive_tractography + loss_contrastive_shuffle + loss_tract_cluster       
        
        
       
#         self.log('val_loss', Loss_combine.item(), batch_size=self.batch_size)
#         predictions = torch.argmax(x, dim=1)
#         self.val_accuracy(predictions.reshape(-1,1), labels.reshape(-1,1))
        
#         self.log('val_accuracy', self.val_accuracy, batch_size=self.batch_size)

#     def test_step(self, test_batch, test_batch_idx):

#         V, F, FF, labels, Fiber_infos = test_batch
#         V = V.to(self.device)
#         F = F.to(self.device)
#         FF = FF.to(self.device)
#         VFI = torch.clone(V)
#         VFI = VFI.to(self.device)
#         FFI = torch.clone(F)
#         FFI = FFI.to(self.device)
#         FFFI = torch.clone(FF)
#         FFFI = FFFI.to(self.device)
#         labels = labels.to(self.device)
#         labelsFI = torch.clone(labels)
#         labelsFI = labelsFI.to(self.device)
#         vfbounds = []
#         sample_min_max = []
#         data_lab = []
#         name_labels = []
        
#         for z in range(len(Fiber_infos)):
#             vfbounds += [Fiber_infos[z][0]]
#             sample_min_max += [Fiber_infos[z][1]]
#             data_lab += Fiber_infos[z][2]
#             name_labels += Fiber_infos[z][3]

#         sample_id = []
#         for w in range(len(name_labels)):
#             sample_id.append(name_labels[w][0])
#         VB = []
#         FB = []
#         FFB = []
#         for i in range(len(sample_id)):
#             VBi = torch.load(f"brain_structures/verts_brain_{sample_id[i]}.pt")
#             FBi = torch.load(f"brain_structures/faces_brain_{sample_id[i]}.pt")
#             FFBi = torch.load(f"brain_structures/face_features_brain_{sample_id[i]}.pt")
#             VBi = VBi.to(self.device)
#             FBi = FBi.to(self.device)
#             FFBi = FFBi.to(self.device)
#             VB.append(VBi)
#             FB.append(FBi)
#             FFB.append(FFBi)
        
#         VB = pad_sequence(VB, batch_first=True, padding_value=0)
#         FB = pad_sequence(FB, batch_first=True, padding_value=-1)
#         FFB = torch.cat(FFB)
#         VB = VB.to(self.device)
#         FB = FB.to(self.device)
#         FFB = FFB.to(self.device)
        
#         V = transformation_verts(V, sample_min_max)
#         VFI = transformation_verts_by_fiber(VFI, vfbounds)
        


#         V1 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
#         V2 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
#         VFI1 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)
#         VFI2 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)
#         V1 = V1.to(self.device)
#         V2 = V2.to(self.device)
#         VFI1 = VFI1.to(self.device)
#         VFI2 = VFI2.to(self.device)
#         for i in range(V1.shape[0]):
#             V1[i] = randomrotation(V1[i])
#             V2[i] = randomrotation(V2[i])
#             VFI1[i] = randomrotation(VFI1[i])
#             VFI2[i] = randomrotation(VFI2[i])
#         condition = False
#         # x, proj_test, x1, x2 = self((V_c, V_c1, V_c2, F_c, FF_c, VFI_c, VFI_c1, VFI_c2, FFI_c, FFFI_c, VB, FB, FFB, condition))    #.shape = (batch_size, num classes)
#         x,  proj_test, x1, x2 = self((V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB, condition))

#         x1_b = torch.tensor([]).to(self.device)
#         x1_t = torch.tensor([]).to(self.device)
#         x2_b = torch.tensor([]).to(self.device)
#         x2_t = torch.tensor([]).to(self.device)
#         proj_test_b = torch.tensor([]).to(self.device)
#         proj_test_t = torch.tensor([]).to(self.device)
#         labels_b = []
#         for i in range(len(data_lab)):
#             if data_lab[i] == 0 :

#                 x1_b = torch.cat((x1_b, x1[i].unsqueeze(0)))
#                 x2_b = torch.cat((x2_b, x2[i].unsqueeze(0)))
#                 proj_test_b = torch.cat((proj_test_b, proj_test[i].unsqueeze(0)))
#                 labels_b.append(labels[i])
#             else:
#                 x1_t = torch.cat((x1_t, x1[i].unsqueeze(0)))
#                 x2_t = torch.cat((x2_t, x2[i].unsqueeze(0)))
#                 proj_test_t = torch.cat((proj_test_t, proj_test[i].unsqueeze(0)))



#         a = random.randint(0,10)
#         b = random.randint(0,10)
#         c = random.randint(0,10)
#         d = random.randint(0,10)
#         e = random.randint(0,10)
#         f = random.randint(0,10)
#         tot = f*1+e*10+d*100+c*1000+b*10000+a*100000
#         name_labels = torch.tensor(name_labels)
#         name_labels = name_labels.to(self.device)
#         data_lab = torch.tensor(data_lab)
#         data_lab = data_lab.to(self.device)
#         labels2 = labels.unsqueeze(dim = 1)
#         data_lab = data_lab.unsqueeze(dim = 1)

#         proj_test = torch.cat((proj_test, labels2, name_labels, data_lab), dim=1)
        
#         lab = torch.unique(labels)
#         lab = lab.cpu()
#         lab = np.array(lab)
    
#         torch.save(proj_test, f"/CMF/data/timtey/results_contrastive_loss_combine_loss_tract_cluster_with_simclr/proj_test_{lab[-1]}_{tot}.pt")
        
#         loss_contrastive = self.loss_contrastive(x1, x2)

#         loss_contrastive_bundle = 0
#         lights = torch.tensor(self.lights).to(self.device)
        
#         for i in range(x1_b.shape[0]):
#             loss_contrastive_bundle += 1-self.loss_cossine(lights[labels_b[i]], x1_b[i].unsqueeze(0)) + 1 - self.loss_cossine(lights[labels_b[i]], x2_b[i].unsqueeze(0)) + 1 - self.loss_cossine(x1_b[i].unsqueeze(0), x2_b[i].unsqueeze(0))
        
#         r = torch.randperm(int(x2_t.shape[0]))
#         x2_t_r = x2_t[r].to(self.device)
        
#         loss_contrastive_tractography = 1 - self.loss_cossine(x1_t, x2_t)
#         loss_contrastive_shuffle = self.loss_cossine(x1_t, x2_t_r)
#         loss_contrastive_tractography = torch.sum(loss_contrastive_tractography)
#         loss_contrastive_shuffle = torch.sum(loss_contrastive_shuffle)
        
        
#         loss_tract_cluster = 0
#         for j in range(x1_t.shape[0]):
#             loss_tract_cluster_i = torch.tensor([]).to(self.device)
#             for i in range(lights.shape[0]):
#                 loss_tract_cluster_i = torch.cat((loss_tract_cluster_i, self.loss_cossine(lights[i], x1_t[j].unsqueeze(0))))
#             topk = torch.topk(loss_tract_cluster_i, 5)
#             topk_indices = topk.indices
#             # loss_tract_cluster = torch.argsort(loss_tract_cluster)
#             n = random.randint(-5,-1)
#             cluster_k_indices = topk_indices[n].item()
#             loss_tract_cluster += 1-self.loss_cossine(lights[cluster_k_indices], x1_t[j].unsqueeze(0))
#         # loss_tract_cluster = loss_tract_cluster[n]
#         Loss_combine = loss_contrastive_bundle + loss_contrastive_tractography + loss_contrastive_shuffle + loss_tract_cluster
        
#         # Loss_combine = loss_contrastive_bundle + loss_contrastive
#         self.log('test_loss', Loss_combine, batch_size=self.batch_size)
#         predictions = torch.argmax(x, dim=1)
        
#         self.log('test_accuracy', self.val_accuracy, batch_size=self.batch_size)
#         output = [predictions, labels]
#         return output

#     def test_epoch_end(self, outputs):
#         self.y_pred = []
#         self.y_true = []

#         for output in outputs:
#             self.y_pred.append(output[0].tolist())
#             self.y_true.append(output[1].tolist())
#         self.y_pred = [ele for sousliste in self.y_pred for ele in sousliste]
#         self.y_true = [ele for sousliste in self.y_true for ele in sousliste]
#         self.y_pred = [[int(ele)] for ele in self.y_pred]
#         self.accuracy = accuracy_score(self.y_true, self.y_pred)
#         # print("accuracy", self.accuracy)
#         # print(classification_report(self.y_true, self.y_pred))
        
#     def get_y_pred(self):
#         return self.y_pred
    
#     def get_y_true(self):
#         return self.y_true
    
#     def get_accuracy(self):
#         return self.accuracy