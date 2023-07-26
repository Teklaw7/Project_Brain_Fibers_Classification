import numpy as np
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pytorch3d.transforms as T3d
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence

def transformation_verts_by_fiber(verts, mean_f, scale_f):
    va = verts - mean_f
    scale_f = scale_f*0.6
    for i in range(va.shape[0]):
        va[i,:,:] = va[i,:,:]*scale_f[i]
    return va

def transformation_verts(verts, mean_v, scale_v):
    va = verts - mean_v
    for i in range(va.shape[0]):
        va[i,:,:] = va[i,:,:]*scale_v[i]
    return va

class RotationTransform:
    def __call__(self, verts, rotation_matrix):
        b = torch.transpose(verts,0,1)
        a= torch.mm(rotation_matrix,torch.transpose(verts,0,1))
        verts = torch.transpose(torch.mm(rotation_matrix,torch.transpose(verts,0,1)),0,1)
        return verts
    
def randomrotation(verts):
    verts_device = verts.get_device()
    rotation_matrix = T3d.random_rotation().to(verts_device)
    rotation_transform = RotationTransform()
    verts = rotation_transform(verts,rotation_matrix)
    return verts


def randomrot(verts):
    verts_i = verts.clone()
    lim = 5*np.pi/180
    gauss_law = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([lim/3]))
    for i in range(verts_i.shape[0]):
        x_sample = torch.tensor(gauss_law.sample().item())
        Rx = torch.tensor([[1,0,0],[0,torch.cos(x_sample),-torch.sin(x_sample)],[0,torch.sin(x_sample),torch.cos(x_sample)]]).to(verts.get_device()).double()
        y_sample = torch.tensor(gauss_law.sample().item())
        Ry = torch.tensor([[torch.cos(y_sample),0,torch.sin(y_sample)],[0,1,0],[-torch.sin(y_sample),0,torch.cos(y_sample)]]).to(verts.get_device()).double()
        z_sample = torch.tensor(gauss_law.sample().item())
        Rz = torch.tensor([[torch.cos(z_sample),-torch.sin(z_sample),0],[torch.sin(z_sample),torch.cos(z_sample),0],[0,0,1]]).to(verts.get_device()).double()
        verts_i[i,:,:] = verts[i,:,:]@Rx #multiplication btw the 2 matrix
        verts_i[i,:,:] = verts[i,:,:]@Ry
        verts_i[i,:,:] = verts[i,:,:]@Rz
    return verts_i


def randomstretching(verts):
    verts_i = verts.clone()
    gauss_law = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([0.1]))
    for i in range(verts_i.shape[0]):
        M = torch.tensor([[gauss_law.sample().item(),0,0],[0,gauss_law.sample().item(),0],[0,0,gauss_law.sample().item()]]).to(verts.get_device())
        M = M.to(torch.float64)
        verts_i[i,:,:] = verts[i,:,:]@M #multiplication btw the 2 matrix
    return verts_i

def get_mean_scale_factor(bounds):
    bounds = np.array(bounds)
    mean_f = [0.0]*3
    bounds_max_f = [0.0]*3
    mean_f[0] = (bounds[0]+bounds[1])/2.0
    mean_f[1] = (bounds[2]+bounds[3])/2.0
    mean_f[2] = (bounds[4]+bounds[5])/2.0
    mean_f = np.array(mean_f)
    bounds_max_f[0] = max(bounds[0],bounds[1])
    bounds_max_f[1] = max(bounds[2],bounds[3])
    bounds_max_f[2] = max(bounds[4],bounds[5])
    bounds_max_f = np.array(bounds_max_f)
    scale_f = 1/np.linalg.norm(bounds_max_f-mean_f)

    return mean_f, scale_f