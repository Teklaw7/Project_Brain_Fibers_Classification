import vtk
import utils


path = "/HELIOS_STOR/validation/tractogram_deterministic_139233_dg_ex.vtp"
bundle = utils.ReadSurf(path)
print(bundle)