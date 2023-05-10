from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames, small_sphere
from dipy.direction import DeterministicMaximumDirectionGetter, BootDirectionGetter
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk, save_vtp
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.viz import window, actor, colormap, has_fury
dwi_path = "/CMF/data/timtey/tractography/training/140420/140420_dwi_b3000.nii"

whitemattermask_path = "/CMF/data/timtey/tractography/training/140420/whiteMatterFA_smooth_mask.nii.gz"
brainmask_path = "/CMF/data/timtey/tractography/training/140420/140420_brain_mask.nii"

# hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
# label_fname = get_fnames('stanford_labels')
hardi_bval_fname = "/CMF/data/timtey/DWI/nii/training/140420-dwi_b3000.bval"
hardi_bvec_fname = "/CMF/data/timtey/DWI/nii/training/140420-dwi_b3000.bvec"
# labels_name = get_fnames(labels_path)
labels =load_nifti_data(whitemattermask_path)# use whi102008_tensor_fa.niite matter
brainlabels = load_nifti_data(brainmask_path)
data, affine, hardi_img = load_nifti(dwi_path, return_img=True)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
# print(data.shape)
# print(affine.shape)
# print(hardi_img.shape)
# bvals, bvecs = read_bvals_bvecs("/CMF/data/timtey/DWI/nii/training/102008-dwi_b3000.bval", "102008-dwi_b3000.bvec")
# print(hardi_bval_fname)
# print(hardi_bvec_fname)
# print(dwi_path)
# print(label_fname)
#print("bvals", bvals)
# print("bvecs", bvecs)
# print(labels_path)
# print("labels",labels)
gtab = gradient_table(bvals, bvecs)
# print("gtab", gtab)
seed_mask = labels == 1
# print("seed mask", seed_mask)
brain_mask = brainlabels == 1
# print("white matter", white_matter)
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
print("seeds", seeds)
response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
# print("response", response)
# print("ratio", ratio)

csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
# print("csd model", csd_model)
csd_fit = csd_model.fit(data, mask=brain_mask)
# print("csd fit", csd_fit)

csa_model = CsaOdfModel(gtab, sh_order=6)
# print("csa model", csa_model)
gfa = csa_model.fit(data, mask=brain_mask).gfa
# print("gfa", gfa)
stopping_criterion = ThresholdStoppingCriterion(gfa, .25)
# print("stopping criterion", stopping_criterion)
detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
   csd_fit.shm_coeff, max_angle=30., sphere=default_sphere)
# print("detmax dg", detmax_dg)
streamline_generator = LocalTracking(detmax_dg, stopping_criterion, seeds,
                                    affine, step_size=.5)
# print("streamline generator", streamline_generator)
print("streamlines")
streamlines = Streamlines(streamline_generator)
print("streamlines", len(streamlines))
print("type streamlines", type(streamlines))
print("streamlines", streamlines)
l = []
for i in range(len(streamlines)):
    if len(streamlines[i]) < 5:
        l.append(i)
# print(len(l))

    

print("streamlines", len(streamlines))
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
# print("sft", sft)
save_trk(sft, "/HELIOS_STOR/training/tractogram_deterministic_140420_dg_ex_fa.trk")
save_vtp(sft, "/HELIOS_STOR/training/tractogram_deterministic_140420_dg_ex_fa.vtp")
print("Saving deterministic tractogram")
interactive = False
if has_fury:
   scene = window.Scene()
   scene.add(actor.line(streamlines))#, colormap.line_colors(streamlines)))
   window.record(scene, out_path='tractogram_deterministic_140420_dg_smooth_ex_22.png',
                 size=(800, 800))
   if interactive:
       window.show(scene)

