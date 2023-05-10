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



class RotationTransform:
    def __call__(self, verts, rotation_matrix):
        b = torch.transpose(verts,0,1)
        # print("b", b.shape)
        a= torch.mm(rotation_matrix,torch.transpose(verts,0,1))
        verts = torch.transpose(torch.mm(rotation_matrix,torch.transpose(verts,0,1)),0,1)
        return verts

# class RandomRotationTransform:
#     def __call__(self, verts):
#         rotation_matrix = T3d.random_rotation()
#         rotation_transform = RotationTransform()
#         verts = rotation_transform(verts,rotation_matrix)
#         return verts
    
def randomrotation(verts):
    verts_device = verts.get_device()
    rotation_matrix = T3d.random_rotation().to(verts_device)
    rotation_transform = RotationTransform()
    # print("verts", verts.shape)
    # print("rotation_matrix", rotation_matrix.shape)
    # print("rotation_matrix", type(rotation_matrix))
    verts = rotation_transform(verts,rotation_matrix)
    return verts

def pad_double_batch(V,F,FF,VFI,FFI,FFFI,VB,FB,FFB, V2,F2,FF2,VFI2,FFI2,FFFI2,VB2,FB2,FFB2):
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
    delta_VB = VB.shape[1] - VB2.shape[1]
    if delta_VB > 0:
        VB2 = torch.cat((VB2, torch.zeros(VB2.shape[0],delta_VB,3).to(VB2.device)), dim=1)
        VB_c = torch.cat((VB,VB2), dim=0)
    elif delta_VB < 0:
        VB = torch.cat((VB, torch.zeros(VB.shape[0],-delta_VB,3).to(VB.device)), dim=1)
        VB_c = torch.cat((VB2,VB), dim=0)
    elif delta_VB == 0:
        VB_c = torch.cat((VB,VB2), dim=0)
    # VB_c = torch.cat((VB,VB2), dim=0)
    delta_FB = FB.shape[1] - FB2.shape[1]
    if delta_FB > 0:
        FB2 = torch.cat((FB2, -torch.ones(FB2.shape[0],delta_FB,3).to(FB2.device)), dim=1)
        FB_c = torch.cat((FB,FB2), dim=0)
    elif delta_FB < 0:
        FB = torch.cat((FB, -torch.ones(FB.shape[0],-delta_FB,3).to(FB.device)), dim=1)
        FB_c = torch.cat((FB2,FB), dim=0)
    elif delta_FB == 0:
        FB_c = torch.cat((FB,FB2), dim=0)
    # FB_c = torch.cat((FB,FB2), dim=0)
    delta_FFB = FFB.shape[1] - FFB2.shape[1]
    if delta_FFB > 0:
        FFB2 = torch.cat((FFB2, torch.zeros(FFB2.shape[0],delta_FFB,3).to(FFB2.device)), dim=1)
        FFB_c = torch.cat((FFB,FFB2), dim=0)
    elif delta_FFB < 0:
        FFB = torch.cat((FFB, torch.zeros(FFB.shape[0],-delta_FFB,3).to(FFB.device)), dim=1)
        FFB_c = torch.cat((FFB2,FFB), dim=0)
    elif delta_FFB == 0:
        FFB_c = torch.cat((FFB,FFB2), dim=0)
    # FFB_c = torch.cat((FFB,FFB2), dim=0)
    return V_c,F_c,FF_c,VFI_c,FFI_c,FFFI_c,VB_c,FB_c,FFB_c



def GetView(meshes,phong_renderer,R,T):
    R = R.to(torch.float32)
    T = T.to(torch.float32)
    # print("R", R.get_device())
    # print("T", T.get_device())
    # print("meshes", meshes)
    # print("phong_renderer", phong_renderer)
    images = phong_renderer(meshes.clone(), R=R, T=T)  
    fragments = phong_renderer.rasterizer(meshes.clone(),R=R,T=T)
    # print(type(fragments))
    # print("fragemts", torch.sum(fragments!=-1))
    # print("images", torch.sum(images!=-1))
    # print("images", images[0,0,0,0:3])
    # print("images", images)
    # print("fragments", fragments)
    pix_to_face = fragments.pix_to_face
    zbuf = fragments.zbuf #shape == (batchsize, image_size, image_size, faces_per_pixel) 
    # print("zbuf", type(zbuf))
    # print("images", images[:,:,:,0:3].shape)
    images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
    # print("fragments", torch.sum(fragments!=-1))
    images = images.permute(0,3,1,2)
    # print("pix_to_face", torch.sum(pix_to_face!=-1))
    pix_to_face = pix_to_face.permute(0,3,1,2)
    return pix_to_face, images

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

def stretch_verts(verts):
    val = 0.03
    for i in range (verts.shape[0]):
        for j in range (verts.shape[1]):
            rng = random.randrange(-10,11)
            verts[i,j,0] = verts[i,j,0]+(rng*val)
            rng = random.randrange(-10,11)
            verts[i,j,1] = verts[i,j,1]+(rng*val)
            rng = random.randrange(-10,11)
            verts[i,j,2] = verts[i,j,2]+(rng*val)
    return verts

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
        # print("pute", reshaped_input.size())
        # print("grosse", type(reshaped_input))
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
        # print("score", score.shape)
        attention_weights = score/torch.sum(score, dim=1,keepdim=True)
        # print("attention_weights", attention_weights.shape)

        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

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
        # print("x", x.shape)
        x_fiber = x[:,0:12]
        # x_brain = x[:,12:]    # with brain
        x_fiber = torch.mm(x_fiber,self.mat_neighbors)
        # x_brain = torch.mm(x_brain,self.mat_neighbors) # with brain
        # print("x_fiber", x_fiber.shape)
        # print("x_brain", x_brain.shape)
        # x = torch.cat((x_fiber,x_brain),dim=1) # with brain
        x = x_fiber
        # print("x", x.shape)
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
        # print("output", output.shape)
        return output


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
        # print("similarity_matrix", similarity_matrix.shape)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss
    
# class LinearLayer(nn.Module):
#     def __init__(self,
#                  in_features,
#                  out_features,
#                  use_bias = True,
#                  use_bn = False,
#                  **kwargs):
#         super(LinearLayer, self).__init__(**kwargs)

#         self.in_features = in_features
#         self.out_features = out_features
#         self.use_bias = use_bias
#         self.use_bn = use_bn
        
#         self.linear = nn.Linear(self.in_features, 
#                                 self.out_features, 
#                                 bias = self.use_bias and not self.use_bn)
#         if self.use_bn:
#              self.bn = nn.BatchNorm1d(self.out_features)

#     def forward(self,x):
#         x = self.linear(x)
#         if self.use_bn:
#             x = self.bn(x)
#         return x

# class ProjectionHead(nn.Module):
#     def __init__(self,
#                  in_features,
#                  hidden_features,
#                  out_features,
#                  head_type = 'nonlinear',
#                  **kwargs):
#         super(ProjectionHead,self).__init__(**kwargs)
#         self.in_features = in_features
#         self.out_features = out_features
#         self.hidden_features = hidden_features
#         self.head_type = head_type

#         if self.head_type == 'linear':
#             self.layers = LinearLayer(self.in_features,self.out_features,False, True)
#         elif self.head_type == 'nonlinear':
#             self.layers = nn.Sequential(
#                 LinearLayer(self.in_features,self.hidden_features,True, True),
#                 nn.ReLU(),
#                 LinearLayer(self.hidden_features,self.out_features,False,True))
        
#     def forward(self,x):
#         x = self.layers(x)
#         return x
    