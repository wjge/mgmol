#include <mpi.h>

#include "GridFunc.h"
#include "Laph4.h"
#include "PEenv.h"

#ifdef HAVE_MAGMA
#include "magma_v2.h"
#endif

#include "memory_space.h"

template <typename ScalarType>
using MemoryDev = MemorySpace::Memory<ScalarType, MemorySpace::Device>;

int main(int argc, char* argv[])
{
    int mpirc = MPI_Init(&argc, &argv);
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Initialization failed!!!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    int size;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string name2 = "FD time";
    std::string name3 = "MPI time";

    Timer time_fdtime(name2);
    Timer time_mpitime(name3);

#ifdef HAVE_MAGMA
    magma_int_t magmalog;

    magmalog = magma_init();
    if (magmalog == MAGMA_SUCCESS)
    {
        std::cout << "MAGMA Initialization: success" << std::endl;
    }
#endif

    //warm up
    for(size_t i=0;i<32;i++)
    {
        std::vector<int> val_host(1024,1);
#ifdef HAVE_OPENMP_OFFLOAD
        std::unique_ptr<int, void (*)(int*)> val_dev(
            MemoryDev<int>::allocate(1024),MemoryDev<int>::free);

        MemorySpace::copy_to_dev(val_host, val_dev);

        int* val_alias = val_dev.get();
#else
        int* val_alias = val_host.data();
#endif
        MGMOL_PARALLEL_FOR(val_alias)
        for(size_t j=0; j<1024; ++j)
        {
            val_alias[j]=j;
        }
        MPI_Allreduce(MPI_IN_PLACE, val_host.data(), 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //end of warm up

    {
        const unsigned sizexyz = std::cbrt(size);
        std::cout << " sizexyz = " << sizexyz << std::endl;
        const double origin[3]  = { 0., 0., 0. };
        const double ll         = 2.;
        const double lattice[3] = { ll, ll, ll };
        const unsigned ngpts[3] = { 32*sizexyz, 32*sizexyz, 32*sizexyz };
        const short nghosts     = 2;

        const double h[3] = { ll / static_cast<double>(ngpts[0]),
            ll / static_cast<double>(ngpts[1]),
            ll / static_cast<double>(ngpts[2]) };

        pb::PEenv mype_env(MPI_COMM_WORLD, ngpts[0], ngpts[1], ngpts[2], 1);

        pb::Grid grid(origin, lattice, ngpts, mype_env, nghosts, 0);

        pb::Laph4<double> lap(grid);

        //for strong scaling
        //const size_t numgridfunc = 3000/static_cast<size_t>(size);
        //for weak scaling
        const size_t numgridfunc = 1;

        std::cout<<"size of mpi: "<< size <<
                   ", num of func per mpi rank: " << numgridfunc << std::endl;

        const int endx = nghosts + grid.dim(0);
        const int endy = nghosts + grid.dim(1);
        const int endz = nghosts + grid.dim(2);

        const double coeffx = 2. * M_PI / grid.ll(0);
        const double coeffy = 2. * M_PI / grid.ll(1);
        const double coeffz = 2. * M_PI / grid.ll(2);

        // periodic GridFunc
        pb::GridFunc<double> gf1(grid, 1, 1, 1);
        pb::GridFunc<double> gf2(grid, 1, 1, 1);

        double* u1 = gf1.uu();

        for (int ix = nghosts; ix < endx; ix++)
        {
            int iix  = ix * grid.inc(0);
            double x = grid.start(0) + ix * h[0];

            for (int iy = nghosts; iy < endy; iy++)
            {
                int iiy  = iy * grid.inc(1) + iix;
                double y = grid.start(1) + iy * h[1];

                for (int iz = nghosts; iz < endz; iz++)
                {
                    double z = grid.start(2) + iz * h[2];

                    u1[iiy + iz] = std::sin(x * coeffx) + std::sin(y * coeffy)
                                   + std::sin(z * coeffz);
                }
            }
        }

        //divid the functions into groups
        const size_t nfuncgroups = 1;
 
        const size_t nfuncpergroup = numgridfunc / nfuncgroups;        

        const size_t nfuncpergroupspace = nfuncpergroup * grid.sizeg();        

        std::cout << "sizeg= " << grid.sizeg() <<
                     ", number of function groups: " << nfuncgroups <<
                     ", number of functions per group: " << nfuncpergroup << 
                     ", size of functiongroup: " << nfuncpergroupspace <<std::endl;

        for(size_t i=0; i<nfuncgroups; i++)
        {
            auto arrayofgf1
                = std::unique_ptr<double[]>(new double[nfuncpergroupspace]());

            for (size_t inum = 0; inum < nfuncpergroup; inum++)
            {
                gf1.set_updated_boundaries(false);

                MPI_Barrier(MPI_COMM_WORLD);
                time_mpitime.start();
                // fill ghost values
                gf1.trade_boundaries();

                time_mpitime.stop();

                std::copy(gf1.uu(), gf1.uu() + grid.sizeg(),
                    arrayofgf1.get() + inum * grid.sizeg());
            }

            auto arrayofgf2
                = std::unique_ptr<double[]>(new double[nfuncpergroupspace]());

            time_fdtime.start();

            // apply FD (-Laplacian) operator to arrayofgf1, result in arrayofgf2
            lap.apply(grid, arrayofgf1.get(), arrayofgf2.get(), nfuncpergroup);

            time_fdtime.stop();
        }

        pb::FDoperInterface::printTimers(std::cout);
    }

#ifdef HAVE_MAGMA
    magmalog = magma_finalize();
#endif

    time_mpitime.print(std::cout);
    time_fdtime.print(std::cout);

    mpirc = MPI_Finalize();

    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Finalize failed!!!" << std::endl;
    }

    return 0;
}
