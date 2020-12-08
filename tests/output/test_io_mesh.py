from firedrake import *
import pytest
from petsc4py import PETSc
from petsc4py.PETSc import ViewerHDF5
from pyop2.mpi import COMM_WORLD
import os


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('outformat', [ViewerHDF5.Format.HDF5_XDMF,
                                       ViewerHDF5.Format.HDF5_PETSC])
@pytest.mark.parametrize('heterogeneous', [False, True])
def test_io_mesh(outformat, heterogeneous, tmpdir):
    # Parameters
    fname = os.path.join(str(tmpdir), "test_io_mesh_dump.h5")
    fname = COMM_WORLD.bcast(fname, root=0)
    ntimes = 3
    # Create mesh.
    mesh = RectangleMesh(4, 1, 4., 1.)
    mesh.init()
    meshname = mesh.name
    plex = mesh.topology_dm
    plex.setOptionsPrefix("original_")
    plex.viewFromOptions("-dm_view")
    # Save mesh.
    with CheckpointFile(fname, "w", comm=COMM_WORLD) as afile:
        afile.save_mesh(mesh, format=outformat)
    # Load -> Save -> Load ...
    grank = COMM_WORLD.rank
    for i in range(ntimes):
        if heterogeneous:
            mycolor = (grank > ntimes - i)
        else:
            mycolor = 0
        comm = COMM_WORLD.Split(color=mycolor, key=grank)
        if mycolor == 0:
            # Load.
            with CheckpointFile(fname, "r", comm=comm) as afile:
                mesh = afile.load_mesh(name=meshname)
            mesh.init()
            flg = mesh.topology_dm.isDistributed()
            # Save.
            with CheckpointFile(fname, "w", comm=comm) as afile:
                afile.save_mesh(mesh, format=outformat)
        COMM_WORLD.Barrier()
