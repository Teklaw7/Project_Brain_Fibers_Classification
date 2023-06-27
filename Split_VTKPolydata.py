import argparse
from tools import utils
import vtk
# from tools import utils
import os


def main(args):
    obj = utils.ReadSurf(args.path)


    nb_cells = obj.GetNumberOfCells()
    n_parts = args.n_parts
    l_obj = []
    list_id = []
    for i in range(n_parts):
        l_obj.append(vtk.vtkPolyData())
        list_id.append([])
    
    # for i in range(nb_cells):
        # list_id[i%n_parts].append(i)
        # 
    for j in range(n_parts):
        for i in range(nb_cells):
            if int(nb_cells/n_parts)*j<= i <int(nb_cells/n_parts)*(j+1):
                list_id[j].append(i)

    # for i in range(nb_cells):
    

    L_tube = []
    for i in range(n_parts):
        L_tube.append(utils.ExtractPart(obj, list_id[i]))
        print("extract_part_done")
    
    for i in range(n_parts):
        m = utils.Merge(L_tube[i])
        print("Merge_done")
        if args.extension == "vtp":
            writer = vtk.vtkXMLPolyDataWriter()
        if args.extension == "vtk":
            writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(os.path.splitext(args.path)+"_"+str(i+1)+".vtp")
        writer.SetInputData(m)
        writer.Write()
        print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a VTK polydata into n parts')
    parser.add_argument('--path', type=str, help='Path Input VTK polydata')
    parser.add_argument('--n', type=int, help='Number of parts to divide the polydata')
    parser.add_argument('--output', type=str, help='Path Output VTK polydata')
    parser.add_argument('--extension', type=str, help='Extension of the output file')
    args = parser.parse_args()

    main(args)

