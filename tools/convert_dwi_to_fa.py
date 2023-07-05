from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import fractional_anisotropy, color_fa
import numpy as np
from dipy.io.image import load_nifti, save_nifti

path_dwi = "/CMF/data/timtey/DWI/nii/validation/139233-dwi_b3000.nii"
hardi_bval_fname = "/CMF/data/timtey/DWI/nii/validation/139233-dwi_b3000.bval"
hardi_bvec_fname = "/CMF/data/timtey/DWI/nii/validation/139233-dwi_b3000.bvec"


data, affine = load_nifti(path_dwi)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)

gtab = gradient_table(bvals, bvecs)
tenmodel = dti.TensorModel(gtab)
maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True, dilate=2)
tenfit = tenmodel.fit(maskdata)
FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0
MD = dti.mean_diffusivity(tenfit.evals)
MD[np.isnan(MD)] = 0
print("before save")
save_nifti('/CMF/data/timtey/tractography/validation/139233/139233_fa_b3000.nii', FA.astype(np.float32), affine)
save_nifti('/CMF/data/timtey/tractography/validation/139233/139233_md_b3000.nii', MD.astype(np.float32), affine)
print("after save")