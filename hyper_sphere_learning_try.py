import argparse

import math
import os
import pandas as pd
import numpy as np 

# from nets.hyper_sphere import LightHouse

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from sklearn.manifold import TSNE
import pickle

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
def main(args):

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=args.out,
    #     filename='{epoch}-{train_loss:.2f}',
    #     save_top_k=1,
    #     monitor='train_loss'
    # )

    # model = LightHouse(args)

    # early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=10, verbose=True, mode="min")

    # if args.tb_dir:
    #     logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    # trainer = Trainer(
    #     logger=logger,
    #     max_epochs=args.epochs,        
    #     callbacks=[early_stop_callback, checkpoint_callback],
    #     accelerator='gpu', 
    #     devices=torch.cuda.device_count(),
    #     strategy=DDPStrategy(find_unused_parameters=False)
    # )
    
    # trainer.fit(model, ckpt_path=args.model)

    # light_house = model.light_house.detach().cpu().numpy()
    tsne = TSNE(n_components=3, perplexity=205)
    
    lights = np.random.normal(loc=0, scale=1, size=(args.init_points, args.emb_dim))
    # fig1 = plt.figure()
    b = 0
    print("lights.shape",lights.shape)
    for i in range(lights.shape[0]):
        # print("point :", lights[i])
        if np.linalg.norm(lights[i]) == 1:
            b+=1
    print("b",lights.shape[0])
    print("b",b)
    # print("lights", lights[0])
    # print("lights", lights[0][1])
    # print("lights", lights[0][1]**2)
    # for i in range(lights.shape[0]):
    #     norm = 0 
    #     for j in range(128):
    #         norm += lights[i][j]**2
    #     print("norm", norm)
    #     norm = np.sqrt(norm)
    #     # print("norm", norm)
    # print(askjhfdkjh)
    lights = np.abs(lights/np.linalg.norm(lights, axis=1, keepdims=True))
    # lights2 = tsne.fit_transform(lights)
    # ax1 = fig1.add_subplot(projection='3d')
    # ax1.scatter(lights2[:,0], lights2[:,1], lights2[:,2], linewidths=5)
    # ax1.set_xlim([-150,150])
    # ax1.set_ylim([-150,150])
    # ax1.set_zlim([-150,150])
    # plt.show()
    print("lights.shape",lights.shape)
    a = 0
    for i in range(lights.shape[0]):
        # print("point :", lights[i])
        if np.linalg.norm(lights[i]) == 1:
            a+=1
    print("a",lights.shape[0])
    print("a",a)


    # fit KMeans++ model to the data
    kmeans = KMeans(n_clusters=args.n_lights, init='k-means++').fit(lights)

    # get the cluster centroids
    # fig2 = plt.figure()
    centroids = kmeans.cluster_centers_
    # ax2 = fig2.add_subplot(projection='3d')
    # ax2.scatter(centroids[:,0], centroids[:,1], centroids[:,2], linewidths=5)
    # ax2.set_xlim([-150,150])
    # plt.show()
    print("centroids.shape",centroids.shape)
    for i in range(centroids.shape[0]):
        # print("point :", centroids[i])
        print("norme :", np.linalg.norm(centroids[i]))
    print("centroids.shape",centroids.shape)
    lights = np.abs(centroids/np.linalg.norm(centroids, axis=1, keepdims=True))    #shape (57,128), on est normalises
    print("lights.shape",lights.shape)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    with open(os.path.join(args.out, "Lights.pickle"), 'wb') as f:
        pickle.dump(lights, f)
    pca = PCA(n_components=3)
    lights_2 = pca.fit_transform(lights)
    print("lights_2.shape",lights_2.shape)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection='3d')
    ax3.scatter(lights_2[:,0], lights_2[:,1], lights_2[:,2], linewidths=5)
    plt.show()
    print(dkfjhkdfjh)
    # for i in range(lights.shape[0]):
    #     norm = 0
    #     for j in range(lights.shape[1]):
    #         norm+=lights[i][j]**2
    #     norm = np.sqrt(norm)
    #     print("norm", norm)
    # print(dkjhfkdjh)
    min_l = 999999999
    for idx, l in enumerate(lights):
        lights_ex = np.concatenate([lights[:idx], lights[idx+1:]])
        min_l = min(min_l, np.min(np.sum(np.square(l - lights_ex), axis=1)))

    print(min_l)
    # print("centroids", centroids)

    # centroids = np.abs(centroids/np.linalg.norm(centroids, axis=1, keepdims=True))
    # print("centroids.shape",centroids.shape)
    # print("centroids", centroids[0], centroids[1], centroids[2])
    # centroids =np.abs(centroids)
    print("centroids", type(centroids))
    centroids_f =centroids
    for i in range(centroids.shape[0]):
        norm = 0
        for j in range(centroids.shape[1]):
            norm += centroids[i][j]**2
        centroids_f[i] = centroids[i]/norm
    print("centroids_f : ",centroids_f.shape)
    # print("centroids_f", centroids_f[0], centroids_f[1], centroids_f[2])
    # for i in range(centroids_f.shape[0]):
    #     norm = 0
    #     for j in range(centroids_f.shape[1]):
    #         norm += centroids_f[i][j]**2
    #     print("norme :", np.sqrt(norm))

    # print(lkdfhkjsdgh)

    centroids2 = tsne.fit_transform(centroids) # use pca instead of tsne
    print("centroids2.shape",centroids2.shape)
    for i in range(centroids2.shape[0]):
        # print("point :", centroids[i])
        print("norme :", np.linalg.norm(centroids2[i]))
        norm = np.linalg.norm(centroids2[i])
        centroids2[i] = centroids2[i]/norm
        print("norme :", np.linalg.norm(centroids2[i]))
    # print("centroids", centroids)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    r=1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi/2:100j, 0.0:pi/2:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='c', alpha=0.1, linewidth=0)
    ax.scatter(centroids2[:,0], centroids2[:,1], centroids2[:,2], linewidths=5)
    for i in range(centroids2.shape[0]):
        ax.text(centroids2[i,0], centroids2[i,1], centroids2[i,2], str("C"+f"{i}"), size=8, zorder=1, color='k')
    phi = 0
    theta = 0
    ax.scatter(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), cos(theta), linewidths=5)
    ax.text(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), cos(theta), str("Cr"), size=8, zorder=1, color='r')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect('equal')
    # plt.show()


    for i in range(centroids2.shape[0]):
        centroids2[i,0] = (centroids2[i,0] - np.min(centroids2[:,0]))/(np.max(centroids2[:,0]) - np.min(centroids2[:,0]))
        centroids2[i,1] = (centroids2[i,1] - np.min(centroids2[:,1]))/(np.max(centroids2[:,1]) - np.min(centroids2[:,1]))
        centroids2[i,2] = (centroids2[i,2] - np.min(centroids2[:,2]))/(np.max(centroids2[:,2]) - np.min(centroids2[:,2]))
    for i in range(centroids2.shape[0]):
        # print("point :", centroids[i])
        print("norme :", np.linalg.norm(centroids2[i]))
        norm = np.linalg.norm(centroids2[i])
        centroids2[i] = centroids2[i]/norm
        print("norme :", np.linalg.norm(centroids2[i]))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    # to get positive quadrant of the sphere between 0 and pi/2 for both, phi and theta
    phi, theta = np.mgrid[0.0:pi/2:100j, 0.0:pi/2:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='c', alpha=0.1, linewidth=0)
    ax.scatter(centroids2[:,0], centroids2[:,1], centroids2[:,2], linewidths=5)
    for i in range(centroids2.shape[0]):
        ax.text(centroids2[i,0], centroids2[i,1], centroids2[i,2], str("C"+f"{i}"), size=8, zorder=1, color='k')
    phi = 0
    theta = 0
    ax.scatter(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), cos(theta), linewidths=5)
    ax.text(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), cos(theta), str("Cr"), size=8, zorder=1, color='r')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect("equal")

    plt.show()
    complete = torch.tensor([])
    for z in range(125):
        con = torch.tensor([0])
        complete = torch.cat((complete, con))
    print("complete.shape", complete.shape)
    # print(sdkfhfkjh)
    pn = torch.tensor([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), cos(theta)])
    pn = torch.cat((pn, complete))
    print("pn.shape", pn.shape)
    pn = pn.unsqueeze(0)
    print("pn.shape", pn.shape)
    print("pn", pn)
    lights = torch.tensor(lights)
    print("lights.shape", lights.shape)
    Lights = torch.cat((lights,pn ))
    print("Lights.shape", Lights.shape)
    # print(ksdjhfkdjh)

    # Cr = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), cos(theta)])
    # print("Cr.shape", Cr.shape)
    # tsne2 = TSNE(n_components=128, perplexity=205)
    # Cr2 = tsne2.fit_transform(Cr)
    # print("Cr2.shape", Cr2.shape)
    



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Find optimal lights')
    parser.add_argument('--n_lights', default=64, type=int, help='Number of points')
    parser.add_argument('--init_points', default=1000, type=int, help='Number of initial random points')
    parser.add_argument('--emb_dim', help='Embeding dimension', type=int, default=128)
    # parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    # parser.add_argument('--momentum', help='Momentum', type=float, default=0.1)
    # parser.add_argument('--weight_decay', help='Weight decay', type=float, default=0)
    # parser.add_argument('--lr', help='Learning rate', type=float, default=1e-4)    
    # parser.add_argument('--model', help='Model to continue training', type=str, default=None)
    
    parser.add_argument('--out', help='Output', type=str, default="./")
    # parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    # parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="lattice")

    args = parser.parse_args()

    main(args)

