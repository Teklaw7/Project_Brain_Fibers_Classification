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
from monai.networks.nets import TorchVisionFCModel, FullyConnectedNet, SEResNet50


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
        x_fiber = torch.mm(x_fiber,self.mat_neighbors)
        x = x_fiber
        size_reshape2 = [batch_size,nbr_features,nbr_cam,3,3]
        x = x.contiguous().view(size_reshape2)
        x = x.permute(0,2,1,3,4)

        size_reshape3 = [batch_size*nbr_cam,nbr_features,3,3]
        x = x.contiguous().view(size_reshape3)

        output = self.module(x)
        output_channels = self.module.out_channels
        size_initial = [batch_size,nbr_cam,output_channels]
        output = output.contiguous().view(size_initial)

        return output



class Fly_by_CNN(pl.LightningModule):
    def __init__(self, contrastive, radius, ico_lvl, dropout_lvl, batch_size, weights, num_classes, verts_left, faces_left, verts_right, faces_right, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=57)
        self.contrastive = contrastive
        self.radius = radius
        self.ico_lvl = ico_lvl
        self.dropout_lvl = dropout_lvl
        self.image_size = 224
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
        efficient_net = models.resnet50(pretrained=True)
        efficient_net_fibers = models.resnet50(pretrained=True)
        efficient_net_brain = models.resnet50(pretrained=True)
        if contrastive:
            efficient_net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            # efficient_net.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # efficient_net_fibers.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # efficient_net_brain.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            efficient_net.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#depthmap
            efficient_net_fibers.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#depthmap
            efficient_net_brain.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #depthmap
        efficient_net.fc = Identity()
        efficient_net_fibers.fc = Identity()
        efficient_net_brain.fc = Identity()

        self.drop = nn.Dropout(p=dropout_lvl)
        self.TimeDistributed = TimeDistributed(efficient_net)
        # print("TimeDistributed", self.TimeDistributed)
        self.TimeDistributed_fiber = TimeDistributed(efficient_net_fibers)
        self.TimeDistributed_brain = TimeDistributed(efficient_net_brain)

        self.WV = nn.Linear(512, 256)
        self.WV_fiber = nn.Linear(512, 256)
        self.WV_brain = nn.Linear(512, 256)

        self.linear = nn.Linear(256, 256)
        self.linear_fiber = nn.Linear(256, 256)
        self.linear_brain = nn.Linear(256, 256)

        #######
        output_size = self.TimeDistributed.module.inplanes
        output_size_fiber = self.TimeDistributed_fiber.module.inplanes
        output_size_brain = self.TimeDistributed_brain.module.inplanes
        # conv2d = nn.Conv2d(512, 256, kernel_size=(3,3),stride=2,padding=0) 
        conv2d = nn.Conv2d(2048, 256, kernel_size=(3,3),stride=2,padding=0)
        conv2d_fiber = nn.Conv2d(2048, 256, kernel_size=(3,3),stride=2,padding=0)
        conv2d_brain = nn.Conv2d(2048, 256, kernel_size=(3,3),stride=2,padding=0)
        # conv2d_fiber = nn.Conv2d(512, 256, kernel_size=(3,3),stride=2,padding=0)
        # conv2d_brain = nn.Conv2d(512, 256, kernel_size=(3,3),stride=2,padding=0)
        self.IcosahedronConv2d = IcosahedronConv2d(conv2d,self.ico_sphere_verts,self.ico_sphere_edges)
        self.IcosahedronConv2d_fiber = IcosahedronConv2d(conv2d_fiber,self.ico_sphere_verts,self.ico_sphere_edges)
        self.IcosahedronConv2d_brain = IcosahedronConv2d(conv2d_brain,self.ico_sphere_verts,self.ico_sphere_edges)
        self.pooling = AvgPoolImages(nbr_images=12) #change if we want brains 24 with brains
        self.pooling_fiber = AvgPoolImages(nbr_images=12) #change if we want brains 24 with brains
        self.pooling_brain = AvgPoolImages(nbr_images=12) #change if we want brains 24 with brains
        #######

        #conv2dForQuery = nn.Conv2d(1280, 1280, kernel_size=(3,3),stride=2,padding=0) #1280,512
        #conv2dForValues = nn.Conv2d(512, 512, kernel_size=(3,3),stride=2,padding=0)  #512,512

        #self.IcosahedronConv2dForQuery = IcosahedronConv2d(conv2dForQuery,self.ico_sphere_verts,self.ico_sphere_edges)
        #self.IcosahedronConv2dForValues = IcosahedronConv2d(conv2dForValues,self.ico_sphere_verts,self.ico_sphere_edges)
        
        #######
        self.Attention = SelfAttention(512, 128)
        self.Attention_fiber = SelfAttention(512, 128)
        self.Attention_brain = SelfAttention(512, 128)
        self.Attention2 = SelfAttention(768, 128)
        self.WV2 = nn.Linear(768, 768)
        #######

        self.Classification = nn.Linear(768, num_classes) #256, if just fiber normalized by brain, but 512 if fiber normalized by fiber and fiber normalized by brain
        # self.Classification_fiber = nn.Linear(256, num_classes)


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

        # len_faces_brain = meshes_brain.GetNumberOfFaces() 
        # self.len_faces_brain = 327680*2 # with brain
        # self.len_faces_brain = len(self.faces_right) + len(self.faces_left) # with brain

    def forward(self, x):
        V, F, FF, VFI, FFI, FFFI, VB, FB, FFB = x
        x, PF = self.render(V,F,FF)
        x_fiber, PF_fiber = self.render(VFI,FFI,FFFI)
        # FF_brain = torch.ones(self.len_faces_brain,8) # with brain
        # FF_brain = FF_brain.to(self.device) # with brain
        x_brain, PF_brain = self.render(VB, FB, FFB) # with brain
        # x_fiber, PF_fiber, x_brain_fiber, PF_brain_fiber = self.render(V,F,FF) # with brain
        # x_brain = torch.cat((x_brain,x_brain),1)
        # PF_brain = torch.cat((PF_brain,PF_brain),1)
        # x = torch.cat((x,x_brain),1) # with brain
        # x_fiber = torch.cat((x_fiber,x_brain_fiber),1) # with brain
        query = self.TimeDistributed(x)
        query_fiber = self.TimeDistributed_fiber(x_fiber)
        query_brain = self.TimeDistributed_brain(x_brain)
        icoconv2d = True
        pool = False
        if icoconv2d:
            x= self.IcosahedronConv2d(query)
            x_fiber = self.IcosahedronConv2d_fiber(query_fiber)
            x_brain = self.IcosahedronConv2d_brain(query_brain)
            # print("x ",x.shape) #(batch_size, 12, 256)
            if pool:
                x = self.pooling(x)
                x_fiber = self.pooling(x_fiber)
                x_brain = self.pooling(x_brain)
            else:
                # print("x ",x.shape) #(batch_size, 12, 256)
                x_a =self.linear(x)
                x_a_fiber =self.linear_fiber(x_fiber)
                x_a_brain =self.linear_brain(x_brain)
                # print("x ",x_a.shape) #(batch_size, 12, 256)
                # print("x ",x.shape) #(batch_size, 12, 256)
                # print("x_fiber ",x_a_fiber.shape) #(batch_size, 12, 256)
                # print("x_brain ",x_a_brain.shape) #(batch_size, 12, 256)
        else:
            values = self.WV(query)
            x_a, w_a = self.Attention(query, values)
        # print("x_a ",x_a.shape) #(batch_size,256)
        # print("x_a_fiber ",x_a_fiber.shape) #(batch_size,256)
        # print("x_a_brain ",x_a_brain.shape) #(batch_size,256)
        # x_a_brain  = torch.tile(x_a_brain,(self.batch_size,1))
        # print("x_a_brain ",x_a_brain.shape) #(batch_size,256)
        x_a = torch.cat((x_a,x_a_fiber,x_a_brain),2)
        # print("x_a ",x_a.shape) #(batch_size,12,768)
        values = self.WV2(x_a)
        # print("values ",values.shape) #(batch_size,12,768)
        x_a, w_a = self.Attention2(x_a, values)
        # print("x_a ",x_a.shape) #(batch_size,256) 768 si attention2
        x_a = self.drop(x_a)
        # print("x_a ",x_a.shape) #(batch_size,256) 768 si attention2
        x = self.Classification(x_a)
        # print("x classsif ",x.shape)  #(batch_size,nb class)

        
        return x

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
            pixel_to_face, images = GetView(meshes_fiber,phong_renderer,R,T)
            # pixel_to_face_brain = GetView(meshes_brain,phong_renderer_brain,R,T) # with brain
            images = images.to(self.device)
            PF.append(pixel_to_face.unsqueeze(dim=1))
            X.append(images.unsqueeze(dim=1))
            # PF_brain.append(pixel_to_face_brain.unsqueeze(dim=1)) # with brain

        PF = torch.cat(PF,dim=1)
        X = torch.cat(X,dim=1)
        X = X[:,:,3,:,:] # the last one who has the picture in black and white of depth
        # torch.save(X, "X.pt")
        X = X.unsqueeze(dim=2)
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
        # x_brain_f = torch.tile(x_brain,(x.shape[0],1,1,1,1)) # with brain
        
        # x = torch.cat((x,x_brain_f),dim=1) # with brain
        
        return x, PF
        # return x, PF, x_brain, PF_brain # with brain

    def training_step(self, train_batch, train_batch_idx):

        V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = train_batch

        # labels = labels.squeeze(dim=1)
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
        
        V = transformation_verts(V, sample_min_max)
        VFI = transformation_verts_by_fiber(VFI, vfbounds)
        x = self((V, F, FF, VFI, FFI, FFFI, VB, FB, FFB))
        # x_fiber = self((VFI, F, FF))
        # if self.contrastive:
        #     x1 = torch.cat((x1, x1, x1), dim=1)
        #     x2 = torch.cat((x2, x2, x2), dim=1)
        # x1, x2 = self.augment(x)
        # z1 = self.model(x1)
        # z2 = self.model(x2)
        # loss_contrastive = self.loss_contrastive(z1, z2)
        # x = self.Sigmoid(x).squeeze(dim=1)
        print("labels", labels)
        predictions = torch.argmax(x, dim=1)
        print("predictions", predictions)
        loss = self.loss_train(x, labels)
        print("loss", loss)
        self.log('train_loss', loss, batch_size=self.batch_size)
        # print("accuracy", self.train_accuracy(x, labels))
        self.train_accuracy(predictions.reshape(-1,1), labels.reshape(-1,1))
        self.log('train_accuracy', self.train_accuracy, batch_size=self.batch_size)

        return loss

        
    def validation_step(self, val_batch, val_batch_idx):

        V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = val_batch
        # labels = labels.squeeze(dim=1)
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
        V = transformation_verts(V, sample_min_max)
        VFI = transformation_verts_by_fiber(VFI, vfbounds)
        x = self((V, F, FF, VFI, FFI, FFFI, VB, FB, FFB))    #.shape = (batch_size, num classes)
        # x_fiber = self((VFI, F, FF)) #.shape = (batch_size, num classes)
        # z1 = self.model(x1)
        # z2 = self.model(x2)
        # loss_contrastive = self.loss(z1, z2)
        # x = self.Sigmoid(x).squeeze(dim=1)
        print("labels", labels)
        loss = self.loss_val(x, labels)
        print("loss", loss)
        self.log('val_loss', loss.item(), batch_size=self.batch_size)
        predictions = torch.argmax(x, dim=1)
        print("predictions", predictions)
        self.val_accuracy(predictions.reshape(-1,1), labels.reshape(-1,1))
        
        self.log('val_accuracy', self.val_accuracy, batch_size=self.batch_size)

        # return loss

    def test_step(self, test_batch, test_batch_idx):
        V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = test_batch
        # labels = labels.squeeze(dim=1)
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

        V = transformation_verts(V, sample_min_max)
        VFI = transformation_verts_by_fiber(VFI, vfbounds)
        x = self((V, F, FF, VFI, FFI, FFFI, VB, FB, FFB))
        # x_fiber = self((VFI, F, FF))
        loss = self.loss_test(x, labels)
        self.log('test_loss', loss, batch_size=self.batch_size)
        
        predictions = torch.argmax(x, dim=1)
        
        self.log('test_accuracy', self.val_accuracy, batch_size=self.batch_size)
        output = [predictions, labels]
        return output

    def test_epoch_end(self, outputs):
        self.y_pred = []
        self.y_true = []

        for output in outputs:
            self.y_pred.append(output[0].tolist())
            self.y_true.append(output[1].tolist())

        self.y_pred = [ele for sousliste in self.y_pred for ele in sousliste]
        self.y_true = [ele for sousliste in self.y_true for ele in sousliste]
        
        
        self.y_pred = [[int(ele)] for ele in self.y_pred]
        
        # print("y_pred", self.y_pred)
        # print("y_true", self.y_true)
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        print("accuracy", self.accuracy)
        print(classification_report(self.y_true, self.y_pred))
        
    def get_y_pred(self):
        return self.y_pred
    
    def get_y_true(self):
        return self.y_true
    
    def get_accuracy(self):
        return self.accuracy
    










