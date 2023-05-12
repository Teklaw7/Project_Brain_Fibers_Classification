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

import pickle

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    lights = np.random.normal(loc=0, scale=1, size=(args.init_points, args.emb_dim))
    lights = np.abs(lights/np.linalg.norm(lights, axis=1, keepdims=True))


    # fit KMeans++ model to the data
    kmeans = KMeans(n_clusters=args.n_lights, init='k-means++').fit(lights)

    # get the cluster centroids
    centroids = kmeans.cluster_centers_
    print("centroids.shape",centroids.shape)
    lights = np.abs(centroids/np.linalg.norm(centroids, axis=1, keepdims=True))    
    print("lights.shape",lights.shape)
    min_l = 999999999
    for idx, l in enumerate(lights):
        lights_ex = np.concatenate([lights[:idx], lights[idx+1:]])
        min_l = min(min_l, np.min(np.sum(np.square(l - lights_ex), axis=1)))

    print(min_l)
    centroids = np.abs(centroids/np.linalg.norm(centroids, axis=1, keepdims=True))
    print("centroids.shape",centroids.shape)
    
    ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], linewidths=5)
    for i in range(centroids.shape[0]):
        ax.text(centroids[i,0], centroids[i,1], centroids[i,2], str("C"+f"{i}"), size=10, zorder=1, color='k')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    with open(os.path.join(args.out, "lights.pickle"), 'wb') as f:
        pickle.dump(lights, f)



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
