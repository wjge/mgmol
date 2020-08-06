#include <mpi.h>

#include "GridFunc.h"
#include "Laph4.h"
#include "PEenv.h"

#ifdef HAVE_MAGMA
#include "magma_v2.h"
#endif

int main(int argc, char* argv[])
{
    int mpirc = MPI_Init(&argc, &argv);
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Initialization failed!!!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

#ifdef HAVE_MAGMA
    magma_int_t magmalog;

    magmalog = magma_init();
    if (magmalog == MAGMA_SUCCESS)
    {
        std::cout << "MAGMA Initialization: success" << std::endl;
    }
#endif

    {
        const double origin[3]  = { 0., 0., 0. };
        const double ll         = 2.;
        const double lattice[3] = { ll, ll, ll };
        const unsigned ngpts[3] = { 32, 32, 32 };
        const short nghosts     = 2;

        const double h[3] = { ll / static_cast<double>(ngpts[0]),
            ll / static_cast<double>(ngpts[1]),
            ll / static_cast<double>(ngpts[2]) };

        pb::PEenv mype_env(MPI_COMM_WORLD, ngpts[0], ngpts[1], ngpts[2]);

        pb::Grid grid(origin, lattice, ngpts, mype_env, nghosts, 0);

        pb::Laph4<double> lap(grid);

        const size_t numgridfunc = 3000;

        auto arrayofgf1
            = std::unique_ptr<double[]>(new double[numgridfunc * grid.sizeg()]());

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

        gf1.set_updated_boundaries(false);

        // fill ghost values
        gf1.trade_boundaries();

        for (size_t inum = 0; inum < numgridfunc; inum++)
        {
            std::copy(gf1.uu(), gf1.uu() + grid.sizeg(),
                arrayofgf1.get() + inum * grid.sizeg());
        }

        auto arrayofgf2
            = std::unique_ptr<double[]>(new double[numgridfunc * grid.sizeg()]());

        // apply FD (-Laplacian) operator to arrayofgf1, result in arrayofgf2
        lap.apply(grid, arrayofgf1.get(), arrayofgf2.get(), numgridfunc);

        // check values in gf2
        double* u2 = gf2.uu();

        for (size_t inum = 0; inum < numgridfunc; inum++)
        {
            std::copy(arrayofgf2.get() + inum * grid.sizeg(),
                arrayofgf2.get() + (inum + 1) * grid.sizeg(), u2);
        }
    }

#ifdef HAVE_MAGMA
    magmalog = magma_finalize();
#endif

    mpirc = MPI_Finalize();

    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Finalize failed!!!" << std::endl;
    }

    return 0;
}
