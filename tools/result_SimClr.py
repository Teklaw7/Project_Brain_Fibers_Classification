import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl 
import torchvision.models as models
from torch.nn.functional import softmax
import torchmetrics
# import utils
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
import os

liste = os.listdir("/CMF/data/timtey/results_contrastive_learning")
print(liste[0])
liste = liste[:-4]
print("l",len(liste))
matrix = [] #should be shape = (56*100,128)
for i in range(len(liste)):
    matrix.append(torch.load(f"/CMF/data/timtey/results_contrastive_learning/{liste[i]}"))
print("m",len(matrix))
MATR = torch.cat(matrix, dim=0)
print(MATR.shape)
MATR = MATR.cpu()

tsne = TSNE(n_components=2)
tsne_results_test = tsne.fit_transform(MATR) #be careful matrix should be on cpu for this transformation
plt.scatter(tsne_results_test[:, 0], tsne_results_test[:, 1], cmap='viridis')
plt.colorbar()
plt.show()

kmean_lab = KMeans(n_clusters=57, random_state=0)
kmeans_labels = kmean_lab.fit_predict(tsne_results_test)
centers  = kmean_lab.cluster_centers_
uniq = torch.unique(torch.tensor(kmeans_labels))
uniq = uniq.numpy()
print("unique",uniq)
plt.scatter(tsne_results_test[:, 0], tsne_results_test[:, 1], c=kmeans_labels,s=50, cmap='viridis')
print(str(uniq[0]))
for i in range(len(centers)):
    plt.text(centers[i, 0], centers[i, 1], str(uniq[i]), fontsize=12)
plt.colorbar()
plt.show()
