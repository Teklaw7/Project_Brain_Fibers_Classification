import argparse
from library import utils_lib
import vtk
import os


def main(args):
    obj = utils_lib.ReadSurf(args.path)

    nb_cells = obj.GetNumberOfCells()
    n_parts = args.n_parts
    list_id = []
    for i in range(n_parts):
        list_id.append([])
    
    for j in range(n_parts):
        for i in range(nb_cells):
            if int(nb_cells/n_parts)*j<= i <int(nb_cells/n_parts)*(j+1):
                list_id[j].append(i)

    L_tube = []
    for i in range(n_parts):
        L_tube.append(utils_lib.ExtractPart(obj, list_id[i]))
        print("extract_part_done")
    
    for i in range(n_parts):
        if args.extension == "vtp":
            writer = vtk.vtkXMLPolyDataWriter()
        if args.extension == "vtk":
            writer = vtk.vtkPolyDataWriter()
        path =f'{os.path.splitext(args.path)[0]}_{str(i+1)}.{args.extension}'
        print("path : ",path)
        writer.SetFileName(path)
        writer.SetInputData(L_tube[i])
        writer.Write()
        print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a VTK polydata into n parts')
    parser.add_argument('--path', type=str, help='Path Input VTK polydata')
    parser.add_argument('--n_parts', type=int, help='Number of parts to divide the polydata')
    parser.add_argument('--output', type=str, help='Path Output VTK polydata')
    parser.add_argument('--extension', type=str, help='Extension of the output file')
    args = parser.parse_args()

    main(args)