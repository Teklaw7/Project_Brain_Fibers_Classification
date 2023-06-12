import math 
import torch

def cos_sim(v):
    v_inner = inner_product(v)
    v_size = vec_size(v)
    v_cos = v_inner / torch.mm(v_size, v_size.t())
    return v_cos


def vec_size(v):
    return v.norm(dim=-1, keepdim=True)


def inner_product(v):
    return torch.mm(v, v.t())


def euclidean_dist(v, eps=1e-10):
    v_norm = (v**2).sum(-1, keepdim=True)
    dist = v_norm + v_norm.t() - 2.0 * torch.mm(v, v.t())
    dist = torch.sqrt(torch.abs(dist) + eps)
    return dist


def theta(v, eps=1e-5):
    v_cos = cos_sim(v).clamp(-1. + eps, 1. - eps)
    x = torch.acos(v_cos) + math.radians(10)
    return x


def triangle(v):
    theta_ = theta(v)
    theta_rad = theta_ * math.pi / 180.
    vs = vec_size(v)
    x = (vs.mm(vs.t())) * torch.sin(theta_rad)
    return x / 2.


def magnitude_dif(v):
    vs = vec_size(v)
    return (vs - vs.t()).abs()


def sector(v):
    ed = euclidean_dist(v)
    md = magnitude_dif(v)
    sec = math.pi * torch.pow((ed + md), 2) * theta(v)/360.
    return sec


def ts_ss(v):
    tri = triangle(v)
    sec = sector(v)
    return tri * sec


def ts_ss_(v, eps=1e-15, eps2=1e-4):
    # reusable compute
    print("tsss")
    print(v.shape) #20,128
    v_inner = torch.mm(v, v.t())
    print(v_inner.shape) #20,20
    vs = v.norm(dim=-1, keepdim=True)
    print(vs.shape) #20,1
    vs_dot = vs.mm(vs.t())
    print(vs_dot.shape) #20,20
    # compute triangle(v)
    v_cos = v_inner / vs_dot #20,20
    print(v_cos.shape)
    v_cos = v_cos.clamp(-1. + eps2, 1. - eps2)  # clamp to avoid backprop instability
    print(v_cos.shape)
    theta_ = torch.acos(v_cos) + math.radians(10)
    print(theta_.shape)
    theta_rad = theta_ * math.pi / 180.
    print(theta_rad.shape)
    tri = (vs_dot * torch.sin(theta_rad)) / 2.
    print(tri.shape)
    # compute sector(v)
    v_norm = (v ** 2).sum(-1, keepdim=True)
    print("v_norm",v_norm.shape)
    euc_dist = v_norm + v_norm.t() - 2.0 * v_inner
    print(euc_dist.shape)
    euc_dist = torch.sqrt(torch.abs(euc_dist) + eps)  # add epsilon to avoid srt(0.)
    print(euc_dist.shape)
    magnitude_diff = (vs - vs.t()).abs()
    print(magnitude_diff.shape)
    sec = math.pi * (euc_dist + magnitude_diff) ** 2 * theta_ / 360.
    print(sec.shape)
    a = tri * sec
    # return tri * sec
    print("fin",a.shape)
    return a 
vec1 = torch.rand(1,3)#.unsqueeze(0)
vec2 = torch.rand(1,3)#.unsqueeze(0)
# print(vec1.shape)
# print(vec2.shape)
# v = torch.tensor([vec1, vec2])
v = torch.cat((vec1, vec2), 0)  # torch.
# print(euclidean_dist(v), euclidean_dist(v).shape)
# print(cos_sim(v), cos_sim(v).shape)
a = ts_ss(v)
b = ts_ss_(v)
print(a)
print(a.shape)
print(b)
print(b.shape)
# print(ts_ss_(v), ts_ss_(v).shape)
