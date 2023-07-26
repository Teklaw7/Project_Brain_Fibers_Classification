import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl 
import torchvision.models as models
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
from pytorch3d.structures import Meshes, join_meshes_as_scene

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from  Transformations.transformations import *


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
        self.batch_size = batch_size
        self.weights = weights
        self.num_classes = num_classes
        self.verts_left = verts_left
        self.faces_left = faces_left
        self.verts_right = verts_right
        self.faces_right = faces_right
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
        
        efficient_net = models.resnet18(pretrained = True)    ### maybe use weights instead of pretrained
        efficient_net.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#depthmap
        efficient_net.fc = Identity()

        self.drop = nn.Dropout(p=dropout_lvl)
        self.TimeDistributed = TimeDistributed(efficient_net)

        self.WV = nn.Linear(512, 256)

        self.linear = nn.Linear(256, 256)

        #######
        conv2d = nn.Conv2d(512, 256, kernel_size=(3,3),stride=2,padding=0) 
        self.IcosahedronConv2d = IcosahedronConv2d(conv2d,self.ico_sphere_verts,self.ico_sphere_edges)
        self.pooling = AvgPoolImages(nbr_images=12) #change if we want brains 24 with brains
        self.Attention = SelfAttention(512, 128)
        self.Attention2 = SelfAttention(512, 128)
        self.WV2 = nn.Linear(512, 512)
        self.WV3 = nn.Linear(512, 512)
        self.Attention3 = SelfAttention(512,256)
        #######

        self.Classification = nn.Linear(512, num_classes) #256, if just fiber normalized by brain, but 512 if fiber normalized by fiber and fiber normalized by brain
        self.projection = nn.Sequential(
           nn.Linear(in_features=512, out_features=512),
           nn.BatchNorm1d(512),
           nn.ReLU(),
           nn.Linear(in_features=512, out_features=128),
           nn.BatchNorm1d(128),
       )
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=57)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=57)

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

    def forward(self, x):
        V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB = x

        x, PF = self.render(V,F,FF)
        X1, PF1 = self.render(V1,F,FF)
        X2, PF2 = self.render(V2,F,FF)

        x_fiber, PF_fiber = self.render(VFI,FFI,FFFI)
        X1_fiber, PF1_fiber = self.render(VFI1,FFI,FFFI)
        X2_fiber, PF2_fiber = self.render(VFI2,FFI,FFFI)

        x_test = torch.cat((x,x_fiber),1)
        X1 = torch.cat((X1,X1_fiber),1)
        X2 = torch.cat((X2,X2_fiber),1)
        query = self.TimeDistributed(x)
        query_test = self.TimeDistributed(x_test)
        query_1 = self.TimeDistributed(X1)
        query_2 = self.TimeDistributed(X2)
        values_1 = self.WV3(query_1)
        values_2 = self.WV3(query_2)
        values_test = self.WV3(query_test)
        
        x_a_1, w_a_1 = self.Attention3(query_1, values_1)
        x_a_2, w_a_2 = self.Attention3(query_2, values_2)
        x_a_test, w_a_test = self.Attention3(query_test, values_test)
        proj1 = self.projection(x_a_1)
        proj2 = self.projection(x_a_2)
        proj_test = self.projection(x_a_test)
        query_fiber = self.TimeDistributed(x_fiber)
        icoconv2d = True
        pool = False
        if icoconv2d:
            x= self.IcosahedronConv2d(query)
            x_fiber = self.IcosahedronConv2d(query_fiber)
            if pool:
                x = self.pooling(x)
                x1 = self.pooling(x1)
                x2 = self.pooling(x2)
                x_fiber = self.pooling(x_fiber)
                x_fiber_1 = self.pooling(x_fiber_1)
                x_fiber_2 = self.pooling(x_fiber_2)
            else:
                x_a =self.linear(x)
                x_a_fiber =self.linear(x_fiber)
        else:
            values = self.WV(query)
            x_a, w_a = self.Attention(query, values)

        x_a = torch.cat((x_a,x_a_fiber),2)
        values = self.WV2(x_a)
        x_a, w_a = self.Attention2(x_a, values)
        x_a = self.drop(x_a)
        x = self.Classification(x_a)
        
        return x, proj_test, proj1, proj2
    
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

        V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = train_batch
        V = V.to(self.device)
        Vo = torch.detach(V).to(self.device)
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

        V = transformation_verts(V, sample_min_max)     # normalisation
        VFI = transformation_verts_by_fiber(VFI, vfbounds)
        
        V1 = randomstretching(Vo.double()).to(self.device)            #random stretching
        V2 = randomstretching(Vo.double()).to(self.device)
        VFI1 = randomstretching(VFI.double()).to(self.device)
        VFI2 = randomstretching(VFI.double()).to(self.device)
        V1 = randomrot(V1).to(self.device)          #random rotation
        V2 = randomrot(V2).to(self.device)
        VFI1 = randomrot(VFI1).to(self.device)
        VFI2 = randomrot(VFI2).to(self.device)
        x, proj_test, x1, x2 = self((Vo, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB))

        loss_contrastive = self.loss_contrastive(x1, x2)

        self.log('train_loss', loss_contrastive, batch_size=self.batch_size)
        self.log('train_accuracy', self.train_accuracy, batch_size=self.batch_size)

        return loss_contrastive

        
    def validation_step(self, val_batch, val_batch_idx):

        V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max= val_batch
        V = V.to(self.device)
        Vo = torch.detach(V).to(self.device)
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
        V = transformation_verts(V, sample_min_max)     # normalisation
        VFI = transformation_verts_by_fiber(VFI, vfbounds)
        V1 = randomstretching(Vo.double()).to(self.device)            #random stretching
        V2 = randomstretching(Vo.double()).to(self.device)
        VFI1 = randomstretching(VFI.double()).to(self.device)
        VFI2 = randomstretching(VFI.double()).to(self.device)
        V1 = randomrot(V1).to(self.device)          #random rotation
        V2 = randomrot(V2).to(self.device)
        VFI1 = randomrot(VFI1).to(self.device)
        VFI2 = randomrot(VFI2).to(self.device)
        x, proj_test, x1, x2 = self((V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB))    #.shape = (batch_size, num classes)
        loss_contrastive = self.loss_contrastive(x1, x2)
        
        self.log('val_loss', loss_contrastive.item(), batch_size=self.batch_size)
        predictions = torch.argmax(x, dim=1)
        self.val_accuracy(predictions.reshape(-1,1), labels.reshape(-1,1))
        
        self.log('val_accuracy', self.val_accuracy, batch_size=self.batch_size)

    def test_step(self, test_batch, test_batch_idx):
        V, F, FF, labels, VFI, FFI, FFFI, labelsFI, VB, FB, FFB, vfbounds, sample_min_max = test_batch
        V = V.to(self.device)
        Vo = torch.detach(V).to(self.device)
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
        V = transformation_verts(V, sample_min_max) # normalisation
        VFI = transformation_verts_by_fiber(VFI, vfbounds)
        V1 = randomstretching(Vo.double()).to(self.device)            #random stretching
        V2 = randomstretching(Vo.double()).to(self.device)
        VFI1 = randomstretching(VFI.double()).to(self.device)
        VFI2 = randomstretching(VFI.double()).to(self.device)
        V1 = randomrot(V1).to(self.device)                         #random rotation
        V2 = randomrot(V2).to(self.device)
        VFI1 = randomrot(VFI1).to(self.device)
        VFI2 = randomrot(VFI2).to(self.device)
        x,  proj_test, x1, x2 = self((V, V1, V2, F, FF, VFI, VFI1, VFI2, FFI, FFFI, VB, FB, FFB))
        tot = np.random.randint(0, 1000000)
        labels2 = labels.unsqueeze(dim = 1)
        proj_test = torch.cat((proj_test, labels2), dim=1)
        lab = torch.unique(labels)
        lab = lab.cpu()
        lab = np.array(lab)
        torch.save(proj_test, f"/CMF/data/timtey/results_contrastive_learning/results2_pretrained_true_t_04/proj_test_{lab[0]}_{tot}.pt") # path where to save the results


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