import argparse
import os
import numpy as np 
import torch
import pickle
from sklearn.cluster import KMeans

def main(args):

    noise = np.random.uniform(low=0.0, high=1.0, size=(args.init_points, args.emb_dim))
    noise = np.abs(noise)
    # fit KMeans++ model to the data
    kmeans = KMeans(n_clusters=args.n_lights, init='k-means++', verbose=True).fit(noise)

    # get the cluster centroids
    lights = kmeans.cluster_centers_
    lights = np.abs(lights)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    with open(os.path.join(args.out, "lights_57_3d_on_positive_sphere_without_norm.pickle"), 'wb') as f:
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
