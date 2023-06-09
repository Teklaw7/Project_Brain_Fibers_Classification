import pandas as pd
import utils
# from main import csv_changes, bounding_box
from os import listdir
from dipy.io.image import load_nifti, load_nifti_data
from warnings import warn
def csv_test(path):
    dt = pd.read_csv(path)
    


    D=[]
    for i in range(len(dt)):
        dp = pd.read_csv("/home/timtey/Documents/Projet/dataset3/tracts_filtered_train_test_label_to_number_copy.csv")
        x = dt.loc[i]
        id = x['id']
        classe = x['class']
        chemin = f"/CMF/data/timtey/tracts/archives/{id}_tracts/{classe}.vtp"
        bundle = utils.ReadSurf(chemin)
        cells = bundle.GetNumberOfCells()
        dp['number'] = dt['id']
        for j in range(cells):
            dp.loc[j] = dt.loc[i]
            dp['number'][j] = int(j)
        dp['number'] = dp['number'].astype(int)
        D.append(dp)
    D = tuple(D)
    df = pd.concat(D)
    df.to_csv("/home/timtey/Documents/Projet/dataset3/tracts_filtered_train_test_label_to_number_copy_f.csv")

path = "/home/timtey/Documents/Projet/dataset3/tracts_filtered_train_test_label_to_number.csv"

def divideby2(path):
    df = pd.read_csv(path)
    dt = df.copy()
    df['num_cells']=0
    dt['num_cells']=1
    D= [df,dt]
    D = tuple(D)
    dp = pd.concat(D)
    dp.to_csv("/home/timtey/Documents/Projet/dataset3/tracts_filtered_train_test_label_to_number_divide_by_2.csv")
    
def get_nb_fibers(path):
    df = pd.read_csv(path)
    df['num_cells']=0
    for i in range(len(df)):
        id = df['id'][i]
        classe = df['class'][i]
        chemin = f"/CMF/data/timtey/tracts/archives/{id}_tracts/{classe}.vtp"
        bundle = utils.ReadSurf(chemin)
        cells = bundle.GetNumberOfCells()
        df['num_cells'][i] = cells

    print(df)
    df.to_csv("/home/timtey/Documents/Projet/dataset3/tracts_filtered_train_test_label_to_number_nb_cells.csv")

def bounding_box(tractography_path):
    bundle = utils.ReadSurf(tractography_path)
    min_max = bundle.GetBounds()
    return min_max

bundle = utils.ReadSurf("/MEDUSA_STOR/timtey/tractography/training/tractogram_deterministic_103515_dg.vtp")
min_max = bundle.GetBounds()


path = "/MEDUSA_STOR/timtey/tractography/training/tractogram_deterministic_102008_dg.vtp"
l_train = listdir("/MEDUSA_STOR/timtey/tractography/training")
l_valid = listdir("/MEDUSA_STOR/timtey/tractography/validation")
l_test = listdir("/MEDUSA_STOR/timtey/tractography/test")
l = [l_train,l_valid,l_test]
l = [item for sublist in l for item in sublist]
MIN_MAX = []
for i in range(len(l_train)):
    min_max_train = list(bounding_box(f"/MEDUSA_STOR/timtey/tractography/training/{l[i]}"))
    MIN_MAX.append(min_max_train)
    print(min_max_train)
for i in range(len(l_valid)):
    min_max_valid = list(bounding_box(f"/MEDUSA_STOR/timtey/tractography/validation/{l[i]}"))
    print(min_max_valid)
    MIN_MAX.append(min_max_valid)
if l_test !=[]:
    for i in range(len(l_test)):
        min_max_test = list(bounding_box(f"/MEDUSA_STOR/timtey/tractography/test/{l[i]}"))
    MIN_MAX.append(min_max_test)
x_min = []
x_max = []
y_min = []
y_max = []
z_min = []
z_max = []
for i in range(len(MIN_MAX)):
    x_min.append(MIN_MAX[i][0])
    x_max.append(MIN_MAX[i][1])
    y_min.append(MIN_MAX[i][2])
    y_max.append(MIN_MAX[i][3])
    z_min.append(MIN_MAX[i][4])
    z_max.append(MIN_MAX[i][5])

brainmask_path = "/CMF/data/timtey/tractography/validation/103515/103515_brain_mask.nii"
brainlabels = load_nifti_data(brainmask_path)

def bounding_box(vol):
    """Compute the bounding box of nonzero intensity voxels in the volume.
    Parameters
    ----------
    vol : ndarray
        Volume to compute bounding box on.
    Returns
    -------
    npmins : list
        Array containg minimum index of each dimension
    npmaxs : list
        Array containg maximum index of each dimension
    """
    # Find bounds on first dimension
    temp = vol
    for i in range(vol.ndim - 1):
        temp = temp.any(-1)
    mins = [temp.argmax()]
    maxs = [len(temp) - temp[::-1].argmax()]
    # Check that vol is not all 0
    if mins[0] == 0 and temp[0] == 0:
        warn('No data found in volume to bound. Returning empty bounding box.')
        return [0] * vol.ndim, [0] * vol.ndim
    # Find bounds on remaining dimensions
    if vol.ndim > 1:
        a, b = bounding_box(vol.any(0))
        mins.extend(a)
        maxs.extend(b)
    return mins, maxs


informations = {
    'surf': l,
    'class': 'whole_brain',
    'id': ["102008", "113215", "119833", "121618", "124826", "103515", "139233", "108525", "124220"],
    'label': 100,
    'x_min': x_min,
    'x_max': x_max,
    'y_min': y_min,
    'y_max': y_max,
    'z_min': z_min,
    'z_max': z_max,
}

df = pd.DataFrame(informations)
print(df)

l_valid = listdir("/MEDUSA_STOR/timtey/tractography/validation")
informations2 = {
    'surf': l_valid,
    'class': 'whole_brain',
    'id': ["103515", "139233", "108525", "124220"],
    'label': 100,
    'x_min': [0,0,0,0],
    'x_max': [0,0,0,0],
    'y_min': [0,0,0,0],
    'y_max': [0,0,0,0],
    'z_min': [0,0,0,0],
}

l_name = ['103515', '139233', '108525', '124220']

for i in range(5,9):
    brainmask_path = f"/CMF/data/timtey/tractography/validation/{l_name[i-5]}/{l_name[i-5]}_brain_mask.nii"
    brainlabels = load_nifti_data(brainmask_path)
    mins, maxs = bounding_box(brainlabels)
    df['x_min'][i] = mins[0]
    df['x_max'][i] = maxs[0]
    df['y_min'][i] = mins[1]
    df['y_max'][i] = maxs[1]
    df['z_min'][i] = mins[2]
    df['z_max'][i] = maxs[2]

dt = pd.DataFrame(informations2)
# df.to_csv("/home/timtey/Documents/Projet/dataset3/whole_brain_tractography.csv")
