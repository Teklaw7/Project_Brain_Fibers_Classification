from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.viz import window, actor, colormap, has_fury

# Enables/disables interactive visualization
interactive = False


hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
label_fname = get_fnames('stanford_labels')

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
print("bvals", bvals)
print("bvecs", bvecs)
gtab = gradient_table(bvals, bvecs)
print("gtab", gtab)
print("labels", labels)
seed_mask = labels == 2
print("seed mask", seed_mask)
white_matter = (labels == 1) | (labels == 2)
print("white matter", white_matter)
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
print("seeds", seeds)
response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
print("response", response)
print("ratio", ratio)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
print("csd_model", csd_model)
csd_fit = csd_model.fit(data, mask=white_matter)
print("csd_fit", csd_fit)
csa_model = CsaOdfModel(gtab, sh_order=6)
print("csa_model", csa_model)
gfa = csa_model.fit(data, mask=white_matter).gfa
print("gfa", gfa)
stopping_criterion = ThresholdStoppingCriterion(gfa, .25)
print("stopping_criterion", stopping_criterion)

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
    csd_fit.shm_coeff, max_angle=30., sphere=default_sphere)
print("detmax_dg", detmax_dg)
streamline_generator = LocalTracking(detmax_dg, stopping_criterion, seeds,
                                     affine, step_size=.5)
# print("streamline_generator", len(streamline_generator))
streamlines = Streamlines(streamline_generator)
print("streamlines", len(streamlines))
print("type", type(streamlines))
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
print("sft", sft)
save_trk(sft, "tractogram_deterministic_dg.trk")

# if has_fury:
#     scene = window.Scene()
#     scene.add(actor.line(streamlines, colormap.line_colors(streamlines)))
#     window.record(scene, out_path='tractogram_deterministic_dg.png',
#                   size=(800, 800))
#     if interactive:
#         window.show(scene)

