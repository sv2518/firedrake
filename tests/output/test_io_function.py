from firedrake import *
import pytest
from petsc4py import PETSc
from petsc4py.PETSc import ViewerHDF5
from pyop2.mpi import COMM_WORLD
from pyop2 import RW
import os
from os.path import abspath, dirname, join

cwd = abspath(dirname(__file__))


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('cell_type', ["triangle", ])
@pytest.mark.parametrize('family_degree', [("CG", 1),
                                           ("CG", 2),
                                           ("CG", 3),
                                           ("CG", 4),
                                           ("CG", 5),
                                           ("DG", 0),
                                           ("DG", 1),
                                           ("DG", 2),
                                           ("DG", 3),
                                           ("DG", 4),
                                           ("DG", 5)])
@pytest.mark.parametrize('format', [ViewerHDF5.Format.HDF5_PETSC, ])
def test_io_function_simplex(cell_type, family_degree, format, tmpdir):
    # Parameters
    family, degree = family_degree
    filename = os.path.join(str(tmpdir), "test_io_function_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    ntimes = 3
    meshname = "exampleDMPlex"
    fs_name = "example_function_space"
    func_name = "example_function"
    # Initially, load an existing triangular mesh.
    comm = COMM_WORLD
    if cell_type == "triangle":
        meshA = Mesh("./docs/notebooks/stokes-control.msh", name=meshname, comm=comm)
    elif cell_type == "quad":
        meshA = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"),
                     name=meshname, comm=comm)
    meshA.init()
    meshA.name = meshname
    plexA = meshA.topology.topology_dm
    VA = FunctionSpace(meshA, family, degree, name=fs_name)
    x, y = SpatialCoordinate(meshA)
    fA = Function(VA, name=func_name)
    fA.interpolate(x * y * y)
    with CheckpointFile(filename, 'w', comm=comm) as afile:
        afile.save_mesh(meshA, format=format)
        afile.save_function(fA)
    volA = assemble(fA * x * y * dx)
    # Load -> View cycle
    grank = COMM_WORLD.rank
    for i in range(ntimes):
        mycolor = (grank > ntimes - i)
        comm = COMM_WORLD.Split(color=mycolor, key=grank)
        if mycolor == 0:
            # Load
            with CheckpointFile(filename, "r", comm=comm) as afile:
                #meshB = afile.load_mesh(name=meshname)
                fB = afile.load_function(func_name, mesh_name=meshname)
            meshB = fB.function_space().mesh()
            x, y = SpatialCoordinate(meshB)
            volB = assemble(fB * x * y * dx)
            # Check
            print("i = ", i)
            print("volA = ", volA)
            print("volB = ", volB)
            assert abs(volB - volA) < 1.e-7
            VB = fB.function_space()
            fBe = Function(VB).interpolate(x * y * y)
            assert assemble(inner(fB - fBe, fB - fBe) * dx) < 1.e-16
            
            # Save
            with CheckpointFile(filename, 'w', comm=comm) as afile:
                afile.save_mesh(meshB, format=format)
                afile.save_function(fB)


if __name__ == "__main__":
    test_io_function_simplex("triangle", ("CG", 5), ViewerHDF5.Format.HDF5_PETSC, "./")
