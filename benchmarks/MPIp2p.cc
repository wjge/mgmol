#include <mpi.h>

#include "GridFunc.h"
#include "Laph4.h"
#include "PEenv.h"

int main(int argc, char* argv[])
{
    int mpirc = MPI_Init(&argc, &argv);
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Initialization failed!!!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string name = "MPI time";

    Timer time_mpitime(name);

    std::cout<<"size of mpi: "<< size << std::endl;

    int nooffunc = 3000; 

    int noofcomm = 4;

    int nooffuncpercomm = nooffunc/noofcomm;

    const int NX = 32*32*6*nooffuncpercomm;

    std::vector<double> C(NX);
    std::vector<double> D(NX);

    MPI_Request reqs[2];

    MPI_Barrier(MPI_COMM_WORLD);
    time_mpitime.start();

    for(int i=0; i<noofcomm; i++)
    {
        if(rank==1)
        {
            MPI_Irecv(D.data(), NX, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[1]);

            MPI_Wait(reqs+1, MPI_STATUS_IGNORE);
        }
        if(rank==0)
        {
            MPI_Isend(C.data(), NX, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &reqs[0]);
        }
    }

    time_mpitime.stop();

    time_mpitime.print(std::cout);

    mpirc = MPI_Finalize();

    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Finalize failed!!!" << std::endl;
    }

    return 0;
}
