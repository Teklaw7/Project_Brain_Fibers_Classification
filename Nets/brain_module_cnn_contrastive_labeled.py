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
# rendering components
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



class Fly_by_CNN_contrastive_labeled(pl.LightningModule):
    def __init__(self, contrastive, radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, verts_left, faces_left, verts_right, faces_right, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.loss_contrastive = ContrastiveLoss(batch_size)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
        self.contrastive = contrastive
        self.radius = radius
        self.ico_lvl = ico_lvl
        self.dropout_lvl = dropout_lvl
        self.image_size = 224
        self.augment = Augment(self.image_size)
        self.batch_size = batch_size
        self.weights = weights
        self.num_classes = num_classes
        self.verts_left = verts_left
        self.faces_left = faces_left
        self.verts_right = verts_right
        self.faces_right = faces_right
        #ico_sphere, _a, _v = utils.RandomRotation(utils.CreateIcosahedron(self.radius, ico_lvl))
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
        if contrastive:
            self.R = self.R.to(torch.float32)
            self.T = self.T.to(torch.float32)
        
        efficient_net = models.resnet18(pretrained = True)    ### maybe use weights instead of pretrained
        # efficient_net_fibers = models.resnet18() 
        # efficient_net_brain = models.resnet18()
        # efficient_net = models.resnet18(pretrained=True)
        if contrastive:
            efficient_net.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#depthmap
            # efficient_net_fibers.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#depthmap
            # efficient_net_brain.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #depthmap
        else:
            # efficient_net.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # efficient_net_fibers.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # efficient_net_brain.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            efficient_net.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#depthmap
            # efficient_net_fibers.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#depthmap
            # efficient_net_brain.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #depthmap
        efficient_net.fc = Identity()
        # efficient_net_fibers.fc = Identity()
        # efficient_net_brain.fc = Identity()

        self.drop = nn.Dropout(p=dropout_lvl)
        self.TimeDistributed = TimeDistributed(efficient_net)
        # print("TimeDistributed", self.TimeDistributed)
        # self.TimeDistributed_fiber = TimeDistributed(efficient_net_fibers)
        # self.TimeDistributed_brain = TimeDistributed(efficient_net_brain)

        self.WV = nn.Linear(512, 256)
        # self.WV_fiber = nn.Linear(512, 256)
        # self.WV_brain = nn.Linear(512, 256)

        self.linear = nn.Linear(256, 256)
        # self.linear_fiber = nn.Linear(256, 256)
        # self.linear_brain = nn.Linear(256, 256)

        #######
        output_size = self.TimeDistributed.module.inplanes
        # output_size_fiber = self.TimeDistributed_fiber.module.inplanes
        # output_size_brain = self.TimeDistributed_brain.module.inplanes
        conv2d = nn.Conv2d(512, 256, kernel_size=(3,3),stride=2,padding=0) 
        # conv2d_fiber = nn.Conv2d(512, 256, kernel_size=(3,3),stride=2,padding=0)
        # conv2d_brain = nn.Conv2d(512, 256, kernel_size=(3,3),stride=2,padding=0)
        self.IcosahedronConv2d = IcosahedronConv2d(conv2d,self.ico_sphere_verts,self.ico_sphere_edges)
        # self.IcosahedronConv2d_fiber = IcosahedronConv2d(conv2d_fiber,self.ico_sphere_verts,self.ico_sphere_edges)
        # self.IcosahedronConv2d_brain = IcosahedronConv2d(conv2d_brain,self.ico_sphere_verts,self.ico_sphere_edges)
        self.pooling = AvgPoolImages(nbr_images=12) #change if we want brains 24 with brains
        # self.pooling_fiber = AvgPoolImages(nbr_images=12) #change if we want brains 24 with brains
        # self.pooling_brain = AvgPoolImages(nbr_images=12) #change if we want brains 24 with brains
        #######

        #conv2dForQuery = nn.Conv2d(1280, 1280, kernel_size=(3,3),stride=2,padding=0) #1280,512
        #conv2dForValues = nn.Conv2d(512, 512, kernel_size=(3,3),stride=2,padding=0)  #512,512

        #self.IcosahedronConv2dForQuery = IcosahedronConv2d(conv2dForQuery,self.ico_sphere_verts,self.ico_sphere_edges)
        #self.IcosahedronConv2dForValues = IcosahedronConv2d(conv2dForValues,self.ico_sphere_verts,self.ico_sphere_edges)
        
        #######
        self.Attention = SelfAttention(512, 128)
        # self.Attention_fiber = SelfAttention(512, 128)
        # self.Attention_brain = SelfAttention(512, 128)
        # self.Attention2 = SelfAttention(768, 128)
        self.Attention2 = SelfAttention(512, 128)
        self.WV2 = nn.Linear(512, 512)
        self.WV3 = nn.Linear(512, 512)
        self.Attention3 = SelfAttention(512,256)
        #######

        self.Classification = nn.Linear(512, num_classes) #256, if just fiber normalized by brain, but 512 if fiber normalized by fiber and fiber normalized by brain
        # self.Classification_fiber = nn.Linear(256, num_classes)
        self.projection = nn.Sequential(
           nn.Linear(in_features=512, out_features=512),
           nn.BatchNorm1d(512),
           nn.ReLU(),
           nn.Linear(in_features=512, out_features=128),
           nn.BatchNorm1d(128),
       )
        # self.projection = MLP
        # self.projection = ProjectionHead(512,24,24)
        # self.Sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=57)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=57)

        # compute_class_weight('balanced', np.unique(self.train_dataset.labels), self.train_dataset.labels)

        self.cameras = FoVPerspectiveCameras()

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0, 
            faces_per_pixel=1, 
            max_faces_per_bin=100000
        )
        # We can add a point light in front of the object.

        lights = AmbientLights()
        rasterizer = MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            )

        # sigmoid_blend_params = BlendParams(sigma=1e-8, gamma=1e-8)

        self.phong_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=self.cameras, lights=lights)
        )
        self.phong_renderer_brain = MeshRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=self.cameras, lights=lights)
        )

        self.loss_train = nn.CrossEntropyLoss(weight = self.weights[0])
        self.loss_val = nn.CrossEntropyLoss(weight = self.weights[1])
        self.loss_test = nn.CrossEntropyLoss(weight = self.weights[2])

        # mesh_left = Meshes(verts=[self.verts_left], faces=[self.faces_left]) # with brain
        # mesh_right = Meshes(verts=[self.verts_right], faces=[self.faces_right]) # with brain
        # mesh_left = mesh_left.to(self.device) # with brain
        # mesh_right = mesh_right.to(self.device) # with brain
        # self.mesh_left.textures = textures # with brain
        # self.mesh_right.textures = textures # with brain
        # self.meshes_brain = join_meshes_as_scene([mesh_left, mesh_right]) # with brain
        # self.meshes_brain = self.meshes_brain.to(self.device) # with brain
# 
        # len_faces_brain = meshes_brain.GetNumberOfFaces() 
        # self.len_faces_brain = 327680*2 # with brain
        # self.len_faces_brain = len(self.faces_right) + len(self.faces_left) # with brain

    def forward(self, x):
        V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB, condition = x

        x, PF = self.render(V,F,FF)
        X1, PF1 = self.render(V1,F,FF)
        X2, PF2 = self.render(V2,F,FF)

        x_fiber, PF_fiber = self.render(VFI,FFI,FFFI)
        # print("x_fiber", x_fiber.shape)
        X1_fiber, PF1_fiber = self.render(VFI1,FFI,FFFI)
        X2_fiber, PF2_fiber = self.render(VFI2,FFI,FFFI)

        brain_help = False
        if brain_help:
            x_brain, PF_brain = self.render(VB, FB, FFB) # with brain
            X1_brain, PF1_brain = self.render(VB, FB, FFB) # with brain
            X2_brain, PF2_brain = self.render(VB, FB, FFB) # with brain

        x_test = torch.cat((x,x_fiber),1)
        X1 = torch.cat((X1,X1_fiber),1)
        X2 = torch.cat((X2,X2_fiber),1)
        # print("X1 ",X1.shape)
        # print("X2 ",X2.shape)
        query = self.TimeDistributed(x)
        query_test = self.TimeDistributed(x_test)
        # print("query", query.shape)
        query_1 = self.TimeDistributed(X1)
        query_2 = self.TimeDistributed(X2)
        # print("query_1", query_1.shape)
        # print("query_2", query_2.shape)
        # print("query_test", query_test.shape)
        # tada = nn.Linear(in_features=512, out_features=512).to(self.device)
        # query_1 = tada(query_1)
        # print("query_1", query_1.shape)
        # bno1 = nn.BatchNorm1d(512).to(self.device)
        values_1 = self.WV3(query_1)
        values_2 = self.WV3(query_2)
        # print("values_1", values_1.shape)
        # print("values_2", values_2.shape)
        values_test = self.WV3(query_test)
        # print("values_test", values_test.shape)
        
        
        x_a_1, w_a_1 = self.Attention3(query_1, values_1)
        x_a_2, w_a_2 = self.Attention3(query_2, values_2)
        x_a_test, w_a_test = self.Attention3(query_test, values_test)
        # print("x_a_1", x_a_1.shape)
        # print("x_a_2", x_a_2.shape)
        # print("x_a_test", x_a_test.shape)
        # taille = [10,24*512]
        # query_1 = query_1.view(taille)
        # query_2 = query_2.view(taille)
        # print("query_1", query_1.shape)
        # print("query_2", query_2.shape)
        # print("query_1", query_1.shape)
        # query_1 = bno1(query_1)
        # print("query_1", query_1.shape)
        # print("query_2", query_2.shape)
        proj1 = self.projection(x_a_1)
        # print("proj1", proj1.shape) # (bs, 128)
        proj2 = self.projection(x_a_2)
        # print("proj2", proj2.shape)
        proj_test = self.projection(x_a_test)
        # print("proj_test", proj_test.shape)
        # print(kdjhfg)
        query_fiber = self.TimeDistributed(x_fiber)
        # query_fiber_1 = self.TimeDistributed(X1_fiber)
        # query_fiber_2 = self.TimeDistributed(X2_fiber)
        if brain_help:
            query_brain = self.TimeDistributed(x_brain)
        # query_brain_1 = self.TimeDistributed(X1_brain)
        # query_brain_2 = self.TimeDistributed(X2_brain)
        # print(query)
        icoconv2d = True
        pool = False
        # print("querry",query.shape) #(batch_size, 12, 512)
        if icoconv2d:
            # print("query ",query.shape)
            x= self.IcosahedronConv2d(query)
            # x1= self.IcosahedronConv2d(query_1)
            # x2= self.IcosahedronConv2d(query_2)
            x_fiber = self.IcosahedronConv2d(query_fiber)
            # x_fiber_1 = self.IcosahedronConv2d(query_fiber_1)
            # x_fiber_2 = self.IcosahedronConv2d(query_fiber_2)
            if brain_help:
                x_brain = self.IcosahedronConv2d(query_brain)
            # x_brain_1 = self.IcosahedronConv2d(query_brain_1)
            # x_brain_2 = self.IcosahedronConv2d(query_brain_2)
            # print("x ",x.shape) #(batch_size, 12, 256)
            if pool:
                x = self.pooling(x)
                x1 = self.pooling(x1)
                x2 = self.pooling(x2)
                x_fiber = self.pooling(x_fiber)
                x_fiber_1 = self.pooling(x_fiber_1)
                x_fiber_2 = self.pooling(x_fiber_2)
                if brain_help:
                    x_brain = self.pooling(x_brain)
                # x_brain_1 = self.pooling(x_brain_1)
                # x_brain_2 = self.pooling(x_brain_2)
            else:
                # print("x ",x.shape) #(batch_size, 12, 256)
                x_a =self.linear(x)
                # x_a_1 =self.linear(x1)
                # x_a_2 =self.linear(x2)
                x_a_fiber =self.linear(x_fiber)
                # x_a_fiber_1 =self.linear(x_fiber_1)
                # x_a_fiber_2 =self.linear(x_fiber_2)
                if brain_help:
                    x_a_brain =self.linear(x_brain)
                # x_a_brain_1 =self.linear(x_brain_1)
                # x_a_brain_2 =self.linear(x_brain_2)
                # print("x ",x_a.shape) #(batch_size, 12, 256)
                # print("x ",x.shape) #(batch_size, 12, 256)
                # print("x_fiber ",x_a_fiber.shape) #(batch_size, 12, 256)
                # print("x_brain ",x_a_brain.shape) #(batch_size, 12, 256)
            # print("x_a ",x_a.shape)
        else:
            values = self.WV(query)
            x_a, w_a = self.Attention(query, values)
        # print("x_a ",x_a.shape) #(batch_size,256)
        # print("x_a_fiber ",x_a_fiber.shape) #(batch_size,256)
        # print("x_a_brain ",x_a_brain.shape) #(batch_size,256)
        # x_a_brain  = torch.tile(x_a_brain,(self.batch_size,1))
        # print("x_a_brain ",x_a_brain.shape) #(batch_size,256)

        if brain_help:
            x_a = torch.cat((x_a,x_a_fiber,x_a_brain),2)
            # x_a_1 = torch.cat((x_a_1,x_a_fiber_1,x_a_brain),2)
            # x_a_2 = torch.cat((x_a_2,x_a_fiber_2,x_a_brain),2)
        else:
            x_a = torch.cat((x_a,x_a_fiber),2)
            # x_a_1 = torch.cat((x_a_1,x_a_fiber_1),2)
            # x_a_2 = torch.cat((x_a_2,x_a_fiber_2),2)
        # print("x_a ",x_a.shape) #(batch_size,12,768)
        # print(szdjgf)
        values = self.WV2(x_a)
        # values_1 = self.WV2(x_a_1)
        # values_2 = self.WV2(x_a_2)
        # print("values ",values.shape) #(batch_size,12,768)
        x_a, w_a = self.Attention2(x_a, values)
        # x_a_1, w_a_1 = self.Attention2(x_a_1, values_1)
        # x_a_2, w_a_2 = self.Attention2(x_a_2, values_2)
        # print("x_a ",x_a.shape) #(batch_size,256) 768 si attention2
        x_a = self.drop(x_a)
        # x_a_1 = self.drop(x_a_1)
        # x_a_2 = self.drop(x_a_2)
        # print("x_a ",x_a.shape) #(batch_size,256) 768 si attention2
        x = self.Classification(x_a)
        # x1 = self.Classification(x_a_1)
        # x2 = self.Classification(x_a_2)
        # print("x classsif ",x.shape)  #(batch_size,nb class)
        
        return x, proj_test, proj1, proj2
    """
    def forward(self, x):
        V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB, condition = x
        brain_help = False
        if condition:
            X1, PF1 = self.render(V1,F,FF)
            X2, PF2 = self.render(V2,F,FF)
            X1_fiber, PF1_fiber = self.render(VFI1,FFI,FFFI)
            X2_fiber, PF2_fiber = self.render(VFI2,FFI,FFFI)
            if brain_help:
                X1_brain, PF1_brain = self.render(VB, FB, FFB)
                X2_brain, PF2_brain = self.render(VB, FB, FFB)
            X1 = torch.cat((X1,X1_fiber),1)
            X2 = torch.cat((X2,X2_fiber),1)
            query_1 = self.TimeDistributed(X1)
            query_2 = self.TimeDistributed(X2)
            values_1 = self.WV3(query_1)
            values_2 = self.WV3(query_2)
            x_a_1, w_a_1 = self.Attention3(query_1, values_1)
            x_a_2, w_a_2 = self.Attention3(query_2, values_2)
            proj1 = self.projection(x_a_1)
            proj2 = self.projection(x_a_2)
            classique = False
            if classique:
                query_fiber = self.TimeDistributed(x_fiber)
                query = self.TimeDistributed(x)
                icoconv2d = True
                pool = False
                if icoconv2d:
                    x = self.IcosahedronConv2d(query)
                    x_fiber = self.IcosahedronConv2d(query_fiber)
                    if brain_help:
                        x_brain = self.IcosahedronConv2d(query_brain)
                    if pool:
                        x = self.pool(x)
                        x_fiber = self.pool(x_fiber)
                        if brain_help:
                            x_brain = self.pool(x_brain)
                    else:
                        x_a = self.linear(x)
                        x_a_fiber = self.linear(x_fiber)
                        if brain_help:
                            x_a_brain = self.linear(x_brain)
                else:
                    values = self.WV(query)
                    x_a, w_a = self.Attention(query, values)
                if brain_help:
                    x_a = torch.cat((x_a,x_a_fiber,x_a_brain),2)
                else:
                    x_a = torch.cat((x_a,x_a_fiber),2)
                values = self.WV2(x_a)
                x_a, w_a = self.Attention2(x_a, values)
                x_a = self.drop(x_a)
                x = self.Classification(x_a)
                return x, proj1, proj2
            else:
                return proj1, proj2
        else:
            x, PF = self.render(V,F,FF)
            x_fiber, PF_fiber = self.render(VFI,FFI,FFFI)
            if brain_help:
                x_brain, PF_brain = self.render(VB, FB, FFB)
                query_brain = self.TimeDistributed(x_brain)
            x_test = torch.cat((x,x_fiber),1)
            query_test = self.TimeDistributed(x_test)
            values_test = self.WV3(query_test)
            x_a_test, w_a_test = self.Attention3(query_test, values_test)
            proj_test = self.projection(x_a_test)
            classique = False
            if classique:
                query_fiber = self.TimeDistributed(x_fiber)
                query = self.TimeDistributed(x)
                icoconv2d = True
                pool = False
                if icoconv2d:
                    x = self.IcosahedronConv2d(query)
                    x_fiber = self.IcosahedronConv2d(query_fiber)
                    if brain_help:
                        x_brain = self.IcosahedronConv2d(query_brain)
                    if pool:
                        x = self.pool(x)
                        x_fiber = self.pool(x_fiber)
                        if brain_help:
                            x_brain = self.pool(x_brain)
                    else:
                        x_a = self.linear(x)
                        x_a_fiber = self.linear(x_fiber)
                        if brain_help:
                            x_a_brain = self.linear(x_brain)
                else:
                    values = self.WV(query)
                    x_a, w_a = self.Attention(query, values)
                if brain_help:
                    x_a = torch.cat((x_a,x_a_fiber,x_a_brain),2)
                else:
                    x_a = torch.cat((x_a,x_a_fiber),2)
                values = self.WV2(x_a)
                x_a, w_a = self.Attention2(x_a, values)
                x_a = self.drop(x_a)
                x = self.Classification(x_a)
                return x, proj_test
            else:
                return proj_test



        '''
        x, PF = self.render(V,F,FF)
        X1, PF1 = self.render(V1,F,FF)
        X2, PF2 = self.render(V2,F,FF)

        x_fiber, PF_fiber = self.render(VFI,FFI,FFFI)
        # print("x_fiber", x_fiber.shape)
        X1_fiber, PF1_fiber = self.render(VFI1,FFI,FFFI)
        X2_fiber, PF2_fiber = self.render(VFI2,FFI,FFFI)

        brain_help = False
        if brain_help:
            x_brain, PF_brain = self.render(VB, FB, FFB) # with brain
            X1_brain, PF1_brain = self.render(VB, FB, FFB) # with brain
            X2_brain, PF2_brain = self.render(VB, FB, FFB) # with brain

        x_test = torch.cat((x,x_fiber),1)
        X1 = torch.cat((X1,X1_fiber),1)
        X2 = torch.cat((X2,X2_fiber),1)
        # print("X1 ",X1.shape)
        # print("X2 ",X2.shape)
        query = self.TimeDistributed(x)
        query_test = self.TimeDistributed(x_test)
        # print("query", query.shape)
        query_1 = self.TimeDistributed(X1)
        query_2 = self.TimeDistributed(X2)
        # print("query_1", query_1.shape)
        # print("query_2", query_2.shape)
        # print("query_test", query_test.shape)
        # tada = nn.Linear(in_features=512, out_features=512).to(self.device)
        # query_1 = tada(query_1)
        # print("query_1", query_1.shape)
        # bno1 = nn.BatchNorm1d(512).to(self.device)
        values_1 = self.WV3(query_1)
        values_2 = self.WV3(query_2)
        # print("values_1", values_1.shape)
        # print("values_2", values_2.shape)
        values_test = self.WV3(query_test)
        # print("values_test", values_test.shape)
        
        
        x_a_1, w_a_1 = self.Attention3(query_1, values_1)
        x_a_2, w_a_2 = self.Attention3(query_2, values_2)
        x_a_test, w_a_test = self.Attention3(query_test, values_test)
        # print("x_a_1", x_a_1.shape)
        # print("x_a_2", x_a_2.shape)
        # print("x_a_test", x_a_test.shape)
        # taille = [10,24*512]
        # query_1 = query_1.view(taille)
        # query_2 = query_2.view(taille)
        # print("query_1", query_1.shape)
        # print("query_2", query_2.shape)
        # print("query_1", query_1.shape)
        # query_1 = bno1(query_1)
        # print("query_1", query_1.shape)
        # print("query_2", query_2.shape)
        proj1 = self.projection(x_a_1)
        # print("proj1", proj1.shape) # (bs, 128)
        proj2 = self.projection(x_a_2)
        # print("proj2", proj2.shape)
        proj_test = self.projection(x_a_test)
        # print("proj_test", proj_test.shape)
        # print(kdjhfg)
        query_fiber = self.TimeDistributed(x_fiber)
        # query_fiber_1 = self.TimeDistributed(X1_fiber)
        # query_fiber_2 = self.TimeDistributed(X2_fiber)
        if brain_help:
            query_brain = self.TimeDistributed(x_brain)
        # query_brain_1 = self.TimeDistributed(X1_brain)
        # query_brain_2 = self.TimeDistributed(X2_brain)
        # print(query)
        icoconv2d = True
        pool = False
        # print("querry",query.shape) #(batch_size, 12, 512)
        if icoconv2d:
            # print("query ",query.shape)
            x= self.IcosahedronConv2d(query)
            # x1= self.IcosahedronConv2d(query_1)
            # x2= self.IcosahedronConv2d(query_2)
            x_fiber = self.IcosahedronConv2d(query_fiber)
            # x_fiber_1 = self.IcosahedronConv2d(query_fiber_1)
            # x_fiber_2 = self.IcosahedronConv2d(query_fiber_2)
            if brain_help:
                x_brain = self.IcosahedronConv2d(query_brain)
            # x_brain_1 = self.IcosahedronConv2d(query_brain_1)
            # x_brain_2 = self.IcosahedronConv2d(query_brain_2)
            # print("x ",x.shape) #(batch_size, 12, 256)
            if pool:
                x = self.pooling(x)
                x1 = self.pooling(x1)
                x2 = self.pooling(x2)
                x_fiber = self.pooling(x_fiber)
                x_fiber_1 = self.pooling(x_fiber_1)
                x_fiber_2 = self.pooling(x_fiber_2)
                if brain_help:
                    x_brain = self.pooling(x_brain)
                # x_brain_1 = self.pooling(x_brain_1)
                # x_brain_2 = self.pooling(x_brain_2)
            else:
                # print("x ",x.shape) #(batch_size, 12, 256)
                x_a =self.linear(x)
                # x_a_1 =self.linear(x1)
                # x_a_2 =self.linear(x2)
                x_a_fiber =self.linear(x_fiber)
                # x_a_fiber_1 =self.linear(x_fiber_1)
                # x_a_fiber_2 =self.linear(x_fiber_2)
                if brain_help:
                    x_a_brain =self.linear(x_brain)
                # x_a_brain_1 =self.linear(x_brain_1)
                # x_a_brain_2 =self.linear(x_brain_2)
                # print("x ",x_a.shape) #(batch_size, 12, 256)
                # print("x ",x.shape) #(batch_size, 12, 256)
                # print("x_fiber ",x_a_fiber.shape) #(batch_size, 12, 256)
                # print("x_brain ",x_a_brain.shape) #(batch_size, 12, 256)
            # print("x_a ",x_a.shape)
        else:
            values = self.WV(query)
            x_a, w_a = self.Attention(query, values)
        # print("x_a ",x_a.shape) #(batch_size,256)
        # print("x_a_fiber ",x_a_fiber.shape) #(batch_size,256)
        # print("x_a_brain ",x_a_brain.shape) #(batch_size,256)
        # x_a_brain  = torch.tile(x_a_brain,(self.batch_size,1))
        # print("x_a_brain ",x_a_brain.shape) #(batch_size,256)

        if brain_help:
            x_a = torch.cat((x_a,x_a_fiber,x_a_brain),2)
            # x_a_1 = torch.cat((x_a_1,x_a_fiber_1,x_a_brain),2)
            # x_a_2 = torch.cat((x_a_2,x_a_fiber_2,x_a_brain),2)
        else:
            x_a = torch.cat((x_a,x_a_fiber),2)
            # x_a_1 = torch.cat((x_a_1,x_a_fiber_1),2)
            # x_a_2 = torch.cat((x_a_2,x_a_fiber_2),2)
        # print("x_a ",x_a.shape) #(batch_size,12,768)
        # print(szdjgf)
        values = self.WV2(x_a)
        # values_1 = self.WV2(x_a_1)
        # values_2 = self.WV2(x_a_2)
        # print("values ",values.shape) #(batch_size,12,768)
        x_a, w_a = self.Attention2(x_a, values)
        # x_a_1, w_a_1 = self.Attention2(x_a_1, values_1)
        # x_a_2, w_a_2 = self.Attention2(x_a_2, values_2)
        # print("x_a ",x_a.shape) #(batch_size,256) 768 si attention2
        x_a = self.drop(x_a)
        # x_a_1 = self.drop(x_a_1)
        # x_a_2 = self.drop(x_a_2)
        # print("x_a ",x_a.shape) #(batch_size,256) 768 si attention2
        x = self.Classification(x_a)
        # x1 = self.Classification(x_a_1)
        # x2 = self.Classification(x_a_2)
        # print("x classsif ",x.shape)  #(batch_size,nb class)
        
        return x, proj_test, proj1, proj2
        '''
        """
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def render(self, V, F, FF):
        # print("V ",V.shape)
        textures = TexturesVertex(verts_features=torch.ones(V.shape))
        V = V.to(torch.float32)
        F = F.to(torch.float32)
        # print("V ",V.shape)
        # print("F ",F.shape)
        # 
        V = V.to(self.device)
        F = F.to(self.device)
        textures = textures.to(self.device)
        meshes_fiber = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )

        # t_left = [0]*self.verts_left.shape[0] # with brain
        # print(len(t_left))
        # self.verts_left = self.verts_left.to(self.device) # with brain
        # self.verts_right = self.verts_right.to(self.device) # with brain
        # print(self.verts_left.shape[0])
        # t_left[:] = np.sqrt(self.verts_left[:,0]**2 + self.verts_left[:,1]**2 + self.verts_left[:,2]**2) # with brain
        # print("done")
        # for i in range(self.verts_left.shape[0]): # with brain
            # print(i)
            # t_left.append(np.sqrt(self.verts_left[i,0]**2 + self.verts_left[i,1]**2 + self.verts_left[i,2]**2)) # with brain
        # t_right = [0]*self.verts_right.shape[0] # with brain
        # t_right[:] = np.sqrt(self.verts_right[:,0]**2 + self.verts_right[:,1]**2 + self.verts_right[:,2]**2) # with brain
        # print("done")
        # for i in range(self.verts_right.shape[0]): # with brain
            # print(i)
            # t_right.append(np.sqrt(self.verts_right[i,0]**2 + self.verts_right[i,1]**2 + self.verts_right[i,2]**2)) # with brain
        # t_left = torch.tensor(t_left).unsqueeze(1) # with brain
        # print("coucou")
        # t_right = torch.tensor(t_right).unsqueeze(1) # with brain
        # t_left = t_left.unsqueeze(0) # with brain
        # print("coucou2")
        # t_right = t_right.unsqueeze(0) # with brain
        # t_left = t_left.to(self.device) # with brain
        # t_right = t_right.to(self.device) # with brain
        
        # texture_left = TexturesVertex(verts_features= t_left) # with brain
        # texture_right = TexturesVertex(verts_features= t_right) # with brain
        
        # print(self.mesh_left)
        # print(self.mesh_right)
        
        # print("len_faces_brain ",len_faces_brain)
        # len_faces_fiber = meshes_fiber.GetNumberOfFaces()
        # print("len_faces_fiber ",len_faces_fiber)
        # print("len_faces_brain ",len_faces_brain)
        # sigmoid_alpha_blend()
        # fig2 = plot_scene({
        #            "display of all fibers": {
        #             #    "mesh": meshes_fiber,
        #                "mesh2": meshes_fiber,
        #            }
        #        })
        # fig2.show()
        
        phong_renderer = self.phong_renderer.to(self.device)
        # phong_renderer_brain = self.phong_renderer_brain.to(self.device)
        meshes_fiber = meshes_fiber.to(self.device)
        PF = []
        X = []
        # PF_brain = []
        for i in range(len(self.R)):
            R = self.R[i][None].to(self.device)
            T = self.T[i][None].to(self.device)
            # print("R.shape", R)
            # print("T.shape", T)
            pixel_to_face, images = GetView(meshes_fiber,phong_renderer,R,T)
            # print("pixel_to_face.shape", pixel_to_face)
            # pixel_to_face_brain = GetView(meshes_brain,phong_renderer_brain,R,T) # with brain
            # print("pixel_to_face.shape", torch.sum(pixel_to_face!=-1))
            images = images.to(self.device)
            PF.append(pixel_to_face.unsqueeze(dim=1))
            X.append(images.unsqueeze(dim=1))
            # PF_brain.append(pixel_to_face_brain.unsqueeze(dim=1)) # with brain

        PF = torch.cat(PF,dim=1)
        X = torch.cat(X,dim=1)
        X = X[:,:,3,:,:] # the last one who has the picture in black and white of depth
        # torch.save(X, "X.pt")
        X = X.unsqueeze(dim=2)
        # print("X", X.shape)
        # print(klshdksj)
        # print("max X", torch.max(X))
        # print("min X", torch.min(X))
        # print("X", torch.mean(X>0))
        # print("PF", PF.shape) # (batch_size, nb_views, 1, 224, 224)
        # print("X", X.shape) # (batch_size, nb_views, 4, 224, 224)
        # PF_brain = torch.cat(PF_brain,dim=1) # with brain

        l_features = []
        # l_features_brain = [] # with brain
        # print("FF.shape", FF.shape)

        # FF_brain = torch.ones(len_faces_brain,8) # with brain
        # FF_brain = FF_brain.to(self.device) # with brain
        
        # print("FF_brain.shape", FF_brain.shape)
        # print("FF_brain", FF_brain[0].shape)
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,index],PF)*(PF >= 0)) # take each feature
            # a = torch.take(FF[:,index],PF)*(PF >= 0)
            # print("ca",)
        # for index in range(FF_brain.shape[-1]): # with brain
            # l_features_brain.append(torch.take(FF_brain[:,index],PF_brain)*(PF_brain >= 0)) # take each feature # with brain

        x = torch.cat(l_features,dim=2)
        x = torch.cat((x,X),dim=2)
        # print("x.shape", x.shape) # (batch_size, nb_views, 8, 224, 224)  sans depthmap infos
        # print("x.shape", x.shape) # (batch_size, nb_views, 12, 224, 224)  avec depthmap infos
        # x_brain = torch.cat(l_features_brain,dim=2) # with brain

        # x === mes 12 images de 224*224*8
        #x.shape(batch_size,nb_cameras,8,224,224)
        # print("self.batch_size", self.batch_size)
        # x_brain_f = torch.tile(x_brain,(x.shape[0],1,1,1,1)) # with brain
        # print("x_brain_f.shape", x_brain_f.shape)
        # print("x.shape", x.shape)
        # print(x.shape)
        
        # x = torch.cat((x,x_brain_f),dim=1) # with brain
        # print("x.shape final", x.shape)
        # x_photo = x[:,0,:3]
        # if self.contrastive:
            # x_photo = x[:,0,:1]
        # print(x_photo.shape)
        # print(ksdhg)
        # x_photo = x[0,:,:3]
        # torch.save(x_photo, "x_photo_new.pt")
        # print(dkjfhs)
        # print(adkjfhsguh)
        # print(jsfgrkjfh)adkjfhsguh
        # print(skjhsfd)
        # torch.save(x, "x.pt")
        # torch.save(PF, "PF.pt")
        # torch.save(x_brain, "x_brain.pt")
        # torch.save(PF_brain, "PF_brain.pt")
        # print(kjhd)
        # print("x.shape", x.shape) # (batch_size, nb_views, nb_features, 224, 224)
        # print("x", x)
        return x, PF
        # return x, PF, x_brain, PF_brain # with brain

    def training_step(self, train_batch, train_batch_idx):

        V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = train_batch
        V = V.to(self.device)
        F = F.to(self.device)
        FF = FF.to(self.device)
        VFI = VFI.to(self.device)
        FFI = FFI.to(self.device)
        FFFI = FFFI.to(self.device)
        VB = VB.to(self.device)
        FB = FB.to(self.device)
        FFB = FFB.to(self.device)
        labels = labels.to(self.device)
        labelsFI = labelsFI.to(self.device)
        # print("VFI", torch.sum(VFI!=0))
        # print("V", torch.sum(V!=0))
        # print("coucou")
        # labels = labels.squeeze(dim=1)
        # if self.contrastive:
            # train_batch_1, train_batch_2 = self.augment(train_batch)

        ###
        # change fibers points with stretching, rotation, translation twices for the V and twices for the VFI
        V = transformation_verts(V, sample_min_max)
        VFI = transformation_verts_by_fiber(VFI, vfbounds)
        
        # V1 = stretch_verts(V)
        V1 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
        V2 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
        VFI1 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)
        VFI2 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)
        # VFI1 = stretch_verts(VFI)
        # V2 = stretch_verts(V)
        # VFI2 = stretch_verts(VFI)  
        V1 = V1.to(self.device)
        V2 = V2.to(self.device)
        VFI1 = VFI1.to(self.device)
        VFI2 = VFI2.to(self.device)
        for i in range(V1.shape[0]):
            V1[i] = randomrotation(V1[i])
            V2[i] = randomrotation(V2[i])
            VFI1[i] = randomrotation(VFI1[i])
            VFI2[i] = randomrotation(VFI2[i])
        condition = True
        x, proj_test, x1, x2 = self((V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB, condition))

        loss = self.loss_train(x, labels)
    
        loss_contrastive = self.loss_contrastive(x1, x2)

        self.log('train_loss', loss_contrastive, batch_size=self.batch_size)
        # print("accuracy", self.train_accuracy(x, labels))
        self.log('train_accuracy', self.train_accuracy, batch_size=self.batch_size)

        return loss_contrastive

        
    def validation_step(self, val_batch, val_batch_idx):

        V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max= val_batch
        V = V.to(self.device)
        F = F.to(self.device)
        FF = FF.to(self.device)
        VFI = VFI.to(self.device)
        FFI = FFI.to(self.device)
        FFFI = FFFI.to(self.device)
        VB = VB.to(self.device)
        FB = FB.to(self.device)
        FFB = FFB.to(self.device)
        labels = labels.to(self.device)
        labelsFI = labelsFI.to(self.device)
        # print("VFI", torch.sum(VFI!=0))
        # print("V", torch.sum(V!=0))
        # print("coucou")
        # labels = labels.squeeze(dim=1)
        # if self.contrastive:
            # train_batch_1, train_batch_2 = self.augment(train_batch)
        print("F", F.shape)
        print("FFI",FFI.shape)
        print("FF", FF.shape)
        print("FFFI", FFFI.shape)
        print("FFB", FFB.shape)
        # print(akjhf)
        V = transformation_verts(V, sample_min_max)
        VFI = transformation_verts_by_fiber(VFI, vfbounds)
        V1 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
        V2 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
        VFI1 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)
        VFI2 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)
        V1 = V1.to(self.device)
        V2 = V2.to(self.device)
        VFI1 = VFI1.to(self.device)
        VFI2 = VFI2.to(self.device)
        # print("V1[:]", V1[:].shape)
        for i in range(V1.shape[0]):
            V1[i] = randomrotation(V1[i])
            V2[i] = randomrotation(V2[i])
            VFI1[i] = randomrotation(VFI1[i])
            VFI2[i] = randomrotation(VFI2[i])
        condition = True
        x, proj_test, x1, x2 = self((V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB, condition))    #.shape = (batch_size, num classes)
        # print("x", x.shape)
        # print("val_batch", val_batch.shape)
        # if self.contrastive:
            # val_batch_i = val_batch[0:3]
            # print("val_batch_1", val_batch_1)
            # print("val_batch", val_batch[0:4])
            # val_batch_1, val_batch_2 = self.augment(val_batch_i)
            # print("val_batch_1", val_batch_1.shape)
            # print("val_batch_2", val_batch_2.shape)
        # x_fiber = self((VFI, F, FF)) #.shape = (batch_size, num classes)
        # print("x", x.shape)
        # print("x_fiber", x_fiber.shape)
        # print(kjsfh)
        # print("x2", x2.shape)
        # if self.contrastive:
            # x1 = torch.cat((x1,x1,x1),1)
            # x2 = torch.cat((x2,x2,x2),1)
        # print("x2 after", x2.shape)
        # z1 = self.model(x1)
        # z2 = self.model(x2)
        # print("loss val")
        # loss_contrastive = self.loss(z1, z2)
        # x = self.Sigmoid(x).squeeze(dim=1)
        # print("x", x.shape)
        # print("labels", labels.shape)
        # print("x", x)
        # print("labels", labels)
        loss = self.loss_val(x, labels)
        # taille = [240,24]
        # x1 = x1.contiguous().view(taille)
        # x2 = x2.contiguous().view(taille)
        # print("x1", x1.shape) 
        # print("x2", x2.shape)
        # print("x_test", proj_test.shape)
        loss_contrastive = self.loss_contrastive(x1, x2)
        # print(kdjhg)
        
        self.log('val_loss', loss_contrastive.item(), batch_size=self.batch_size)
        predictions = torch.argmax(x, dim=1)
        self.val_accuracy(predictions.reshape(-1,1), labels.reshape(-1,1))
        
        self.log('val_accuracy', self.val_accuracy, batch_size=self.batch_size)

        # return loss

    def test_step(self, test_batch, test_batch_idx):
        V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = test_batch
        V = V.to(self.device)
        F = F.to(self.device)
        FF = FF.to(self.device)
        VFI = VFI.to(self.device)
        FFI = FFI.to(self.device)
        FFFI = FFFI.to(self.device)
        VB = VB.to(self.device)
        FB = FB.to(self.device)
        FFB = FFB.to(self.device)
        labels = labels.to(self.device)
        labelsFI = labelsFI.to(self.device)
        # print("VFI", torch.sum(VFI!=0))
        # print("V", torch.sum(V!=0))
        # print("coucou")
        # labels = labels.squeeze(dim=1)
        # if self.contrastive:
            # train_batch_1, train_batch_2 = self.augment(train_batch)
        V = transformation_verts(V, sample_min_max)
        VFI = transformation_verts_by_fiber(VFI, vfbounds)
        V1 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
        V2 = V +torch.normal(0, 0.03, size=V.shape).to(self.device)
        VFI1 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)
        VFI2 = VFI +torch.normal(0, 0.03, size=VFI.shape).to(self.device)
        V1 = V1.to(self.device)
        V2 = V2.to(self.device)
        VFI1 = VFI1.to(self.device)
        VFI2 = VFI2.to(self.device)
        for i in range(V1.shape[0]):
            V1[i] = randomrotation(V1[i])
            V2[i] = randomrotation(V2[i])
            VFI1[i] = randomrotation(VFI1[i])
            VFI2[i] = randomrotation(VFI2[i])
        condition = False
        x,  proj_test, x1, x2 = self((V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB, condition))
        a = random.randint(0,10)
        b = random.randint(0,10)
        c = random.randint(0,10)
        d = random.randint(0,10)
        e = random.randint(0,10)
        f = random.randint(0,10)
        tot = f*1+e*10+d*100+c*1000+b*10000+a*100000
        # proj_test = proj_test.unsqueeze(dim = 0)
        # print("proj_test", proj_test.shape)
        labels2 = labels.unsqueeze(dim = 1)
        # print("labels", labels.shape)
        # print(labels)
        proj_test = torch.cat((proj_test, labels2), dim=1)
        # print("proj_test", proj_test.shape)
        # print(torch.unique(labels))
        lab = torch.unique(labels)
        lab = lab.cpu()
        lab = np.array(lab)
        # print(lab[0])
        # print(kdhjksg)
        # for i in range(proj_test.shape[0]):
            # print("proj_test", proj_test[i])
        torch.save(proj_test, f"/CMF/data/timtey/results_contrastive_learning/results2_pretrained_true_t_04/proj_test_{lab[0]}_{tot}.pt")
        # print(ksjhf)

        # if self.contrastive:
            # test_batch_1, test_batch_2 = self.augment(test_batch)

        # x_fiber = self((VFI, F, FF))
        # x = self.Sigmoid(x).squeeze(dim=1)
        # print("x1", x1.shape)
        # if self.contrastive:
            # x1 = torch.cat((x1,x1,x1),1)
            # x2 = torch.cat((x2,x2,x2),1)
        # x1 = torch.cat((x1,x1,x1),1)
        # x2 = torch.cat((x2,x2,x2),1)
        # print("x1 after", x1.shape)
        # z1 = self.model(x1)
        # z2 = self.model(x2)

        # loss_contrastive = self.loss(z1, z2, labels)
        # loss = self.loss_test(x, labels)
        # tsne = TSNE(n_components=2)
        # tsne = tsne.to(self.device)
        # x1 = x1.to(self.device)
        # print("x1", x1.shape)
        # print("x1", x1.get_device())
        # print("x2", x2.shape)
        # print("x2", x2.get_device())
        # print("tsne", tsne.get_device())
        # x1 = x1.cpu()
        # x2 = x2.cpu()
        # proj_test = proj_test.cpu()
        # print("x1", x1.get_device())
        # print("x2", x2.get_device())
        # tsne_results1 = tsne.fit_transform(x1)
        # tsne_results2 = tsne.fit_transform(x2)
        # tsne_results_test = tsne.fit_transform(proj_test)
        # print("labels", labels)
        # se = set(labels.tolist())
        # nb_clusters = len(se)
        # print("nb_clusters", nb_clusters)
        # kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(tsne_results_test)
        # print("kmeans", kmeans)
        # y_kmeans = kmeans.predict(tsne_results_test)
        # centers = kmeans.cluster_centers_
        # print("tsne_results1", tsne_results1.shape) #(bs,2)
        # print("tsne_results2", tsne_results2.shape) #(bs,2)
        # print("tsne_results_test", tsne_results_test.shape) #(bs,2)
        # plt.scatter(tsne_results_test[:, 0], tsne_results_test[:, 1], c=y_kmeans, s=50, cmap='viridis')
        # plt.show()

        # x1 = x1.to(self.device)
        # x2 = x2.to(self.device)
        # proj_test = proj_test.to(self.device)
        loss_contrastive = self.loss_contrastive(x1, x2)
        self.log('test_loss', loss_contrastive, batch_size=self.batch_size)
        # torch.save(tsne_results_test, f"/CMF/data/timtey/tsne/tsne_results_test_{loss_contrastive}.pt")
        # print("x", x.shape)
        predictions = torch.argmax(x, dim=1)
        # print("predictions", predictions)
        #for i in range(len(predictions)):
            #print("x[i][predictions[i]]", x[i][predictions[i]])
            #if x[i][predictions[i]]<0.5:
            #    predictions[i] = 0
        #print("predictions", predictions)
        self.log('test_accuracy', self.val_accuracy, batch_size=self.batch_size)
        output = [predictions, labels]
        return output

    def test_epoch_end(self, outputs):
        self.y_pred = []
        self.y_true = []
        # print("outputs", outputs)

        for output in outputs:
            self.y_pred.append(output[0].tolist())
            self.y_true.append(output[1].tolist())

        self.y_pred = [ele for sousliste in self.y_pred for ele in sousliste]
        self.y_true = [ele for sousliste in self.y_true for ele in sousliste]
        
        
        self.y_pred = [[int(ele)] for ele in self.y_pred]
        
        # print("y_pred", self.y_pred)
        # print("y_true", self.y_true)
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        # print("accuracy", self.accuracy)
        # print(classification_report(self.y_true, self.y_pred))
        
    def get_y_pred(self):
        return self.y_pred
    
    def get_y_true(self):
        return self.y_true
    
    def get_accuracy(self):
        return self.accuracy







class Augment:
   """
   A stochastic data augmentation module
   Transforms any given data example randomly
   resulting in two correlated views of the same example,
   denoted x ̃i and x ̃j, which we consider as a positive pair.
   """

   def __init__(self, img_size, s=1):
       color_jitter = T.ColorJitter(
           0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
       )
       # 10% of the image
    #    print("img_size", img_size)
       blur = T.GaussianBlur((3, 3), (0.1, 2.0))
       
    #    print("blur") 
       self.train_transform = torch.nn.Sequential(
        
           T.RandomResizedCrop(size=img_size),
        #    T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
        #    T.RandomApply([color_jitter], p=0.8),
        #    T.RandomApply([blur], p=0.5),
        #    T.RandomGrayscale(p=0.2),
        #    imagenet stats
        #    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       )
    #    print("train_transform")

   def __call__(self, x):
    #    print("x", x.shape)
    #    print("type", type(x))
       a = self.train_transform(x)
       return a, self.train_transform(x)

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
        # print("denom_1", device_as(self.mask, similarity_matrix).shape)
        # print("denom_2", torch.exp(similarity_matrix / self.temperature).shape)
        # print(self.mask.shape)
        # print(similarity_matrix.shape)
        # print(self.temperature)
        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        # print("loss", loss)
        return loss
    
class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, 
                                self.out_features, 
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,False,True))
        
    def forward(self,x):
        x = self.layers(x)
        return x
    
