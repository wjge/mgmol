#include <mpi.h>

#include "GridFunc.h"
#include "Laph4.h"
#include "PEenv.h"
#include "GridFuncVector.h"

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

    std::string name3 = "MPI time";

    Timer time_mpitime(name3);

    {
        //this is for weak scaling:
        const unsigned sizexyz = std::cbrt(size);
        std::cout << " sizexyz = " << sizexyz << std::endl;
        const double origin[3]  = { 0., 0., 0. };
        const double ll         = 2.;
        const double lattice[3] = { ll, ll, ll };
        //ngpts holds the size of the whole computational domain of the problem
        const unsigned ngpts[3] = { 32*sizexyz, 32*sizexyz, 32*sizexyz };
        const short nghosts     = 2;

        const double h[3] = { ll / static_cast<double>(ngpts[0]),
            ll / static_cast<double>(ngpts[1]),
            ll / static_cast<double>(ngpts[2]) };

        pb::PEenv mype_env(MPI_COMM_WORLD, ngpts[0], ngpts[1], ngpts[2], 1);

        pb::Grid grid(origin, lattice, ngpts, mype_env, nghosts, 0);

        const int nfunc = 3000;
        std::vector<std::vector<int>> gids;
        gids.resize(1);

        for (int i=0; i<nfunc; i++)
        {
           gids[0].push_back(i);
        }
        pb::GridFuncVector<double> gfv(grid, 1, 1, 1, gids);

        gfv.set_updated_boundaries(false);
        
        MPI_Barrier(MPI_COMM_WORLD);
        time_mpitime.start();
 
        gfv.trade_boundaries();

        MPI_Barrier(MPI_COMM_WORLD);
        time_mpitime.stop();
    }

    time_mpitime.print(std::cout);

    mpirc = MPI_Finalize();

    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Finalize failed!!!" << std::endl;
    }

    return 0;
}
