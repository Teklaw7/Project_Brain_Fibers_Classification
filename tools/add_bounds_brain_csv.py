from library import utils_lib
from os import listdir
import pandas as pd
bundles = utils_lib.ReadSurf("/MEDUSA_STOR/timtey/tractography/validation/tractogram_deterministic_103515_dg.vtp")
a = bundles.GetBounds()
bundles2 = utils_lib.ReadSurf("/MEDUSA_STOR/timtey/tractography/validation/tractogram_deterministic_139233_dg.vtp")
b = bundles2.GetBounds()
bundles3 = utils_lib.ReadSurf("/MEDUSA_STOR/timtey/tractography/validation/tractogram_deterministic_108525_dg.vtp")
c = bundles3.GetBounds()
bundles4 = utils_lib.ReadSurf("/MEDUSA_STOR/timtey/tractography/validation/tractogram_deterministic_124220_dg.vtp")
d = bundles4.GetBounds()
l_valid = listdir("/MEDUSA_STOR/timtey/tractography/validation")

x_min = [a[0],b[0],c[0],d[0]]
x_max = [a[1],b[1],c[1],d[1]]
y_min = [a[2],b[2],c[2],d[2]]
y_max = [a[3],b[3],c[3],d[3]]
z_min = [a[4],b[4],c[4],d[4]]
z_max = [a[5],b[5],c[5],d[5]]


informations2 = {
    'surf': l_valid,
    'class': 'whole_brain',
    'id': ["103515", "139233", "108525", "124220"],
    'label': 100,
    'x_min': x_min,
    'x_max': x_max,
    'y_min': y_min,
    'y_max': y_max,
    'z_min': z_min,
    'z_max': z_max,
}

df2 = pd.DataFrame(informations2)
df2.to_csv("/home/timtey/Documents/Projet/dataset3/whole_brain_tractography_validation.csv")