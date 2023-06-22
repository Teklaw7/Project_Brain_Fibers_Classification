import numpy as np
import torch
import pandas as pd
import os
import pickle
import argparse

def find_closest_point(lights):
    closest_lights = torch.tensor([])
    for i in range(lights.shape[0]):
        center = lights[i, :]
        dist = torch.sqrt(torch.sum((lights - center) ** 2, dim=1))
        print(dist.shape)
        print(dist)
        values, indices = dist.sort(stable=True)
        print(values)
        print(indices)
        closest_lights = torch.cat((closest_lights, indices[1].unsqueeze(0)), dim=0)
    print(closest_lights.shape)
    print(closest_lights)

    return closest_lights




lights = torch.tensor(pd.read_pickle(r'lights_57_3d_on_positive_sphere.pickle'))
print(lights.shape)
close = find_closest_point(lights)

torch.save(close, 'closest_lights_57_3d_on_positive_sphere.pt')