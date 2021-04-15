from firedrake import *
import pytest
from petsc4py import PETSc
from pyop2.mpi import COMM_WORLD
from os.path import abspath, dirname, join

cwd = abspath(dirname(__file__))


@pytest.mark.parametrize('cell_type', ["triangle", "quad"])
@pytest.mark.parametrize('family_degree', [("CG", 5), ])
def test_io_orientation_simplex(cell_type, family_degree):
    """
    >>> mesh.topology_dm.viewFromOptions("-dm_view")
    Cones:
    [0] Max cone size: 3
    [0]: 0 <---- 6 (0)
    [0]: 0 <---- 7 (0)
    [0]: 0 <---- 8 (0)
    [0]: 1 <---- 9 (0)
    [0]: 1 <---- 10 (0)
    [0]: 1 <---- 7 (-2)
    [0]: 6 <---- 2 (0)
    [0]: 6 <---- 3 (0)
    [0]: 7 <---- 3 (0)
    [0]: 7 <---- 4 (0)
    [0]: 8 <---- 4 (0)
    [0]: 8 <---- 2 (0)
    [0]: 9 <---- 3 (0)
    [0]: 9 <---- 5 (0)
    [0]: 10 <---- 5 (0)
    [0]: 10 <---- 4 (0)
    >>> mesh.cell_closure
    [[ 3  5  4 10  7  9  1]
     [ 3  4  2  8  6  7  0]]
    >>> mesh.cell_orientation
    [[ 0  0  0  0  0  0  1]
     [ 0  0  0  0 -2  0  2]]
    >>> V.finat_element.entity_dofs()
    {0: {0: [0], 1: [1], 2: [2]},
     1: {0: [3, 4, 5, 6], 1: [7, 8, 9, 10], 2: [11, 12, 13, 14]},
     2: {0: [15, 16, 17, 18, 19, 20]}}
    >>> V.cell_node_list
    [[18 19 20 10 11 12 13 14 15 16 17  6  7  8  9  0  1  2  3  4  5]
     [18 20 35 31 32 33 34 30 29 28 27 14 15 16 17 21 22 23 24 25 26]]
    """
    # Parameters
    family, degree = family_degree
    if cell_type == "triangle":
        mesh = UnitSquareMesh(1, 1, quadrilateral=False)
    elif cell_type == "quad":
        mesh = UnitSquareMesh(1, 1, quadrilateral=True)
    elif cell_type == "ext_quad":
        m = UnitIntervalMesh(1)
        mesh = ExtrudedMesh(m, layers=1, layer_height=10) 
    mesh.init()
    V = FunctionSpace(mesh, family, degree)


    mesh.topology_dm.viewFromOptions("-dm_view")
    print("cell_closure     :", mesh.cell_closure)
    print("cell_orientations:", mesh.entity_orientation)
    print(V.cell_node_list)
    print("finat_element:     ", V.finat_element)
    print("finat_element.cell:", V.finat_element.cell)
    print("entity_dofs :", V.finat_element.entity_dofs())
    print("permutations:", V.finat_element.permutations())


if __name__ == "__main__":
    # mpiexec -n 1 python tests/output/test_io_orientation.py -dm_view ascii::ascii_info_detail
    test_io_orientation_simplex("ext_quad", ("DPC", 0))
