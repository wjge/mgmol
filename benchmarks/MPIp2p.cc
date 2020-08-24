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

    const int noofdir = 6;

    const int nooffunc = 3000; 

    const int noofcomm = 1;

    int nooffuncpercomm = nooffunc/noofcomm;

    std::cout<<"size of mpi: "<< size << 
    ", total number of functions: " << nooffunc <<
    ", number of functions per mpi send/recv:" << nooffuncpercomm  << 
    std::endl;

    const int NX = 32*32*2*nooffuncpercomm;

    std::vector<double> C(NX);
    std::vector<double> D(NX);

    int LEFT = (rank-1+size)%size;
    int RIGHT = (rank+1+size)%size;

    MPI_Request reqs[2];

    MPI_Barrier(MPI_COMM_WORLD);
    time_mpitime.start();

    for(int j=0; j<noofdir; j++)
    {
        for(int i=0; i<noofcomm; i++)
        {
            MPI_Irecv(D.data(), NX, MPI_DOUBLE, RIGHT, 2423, MPI_COMM_WORLD, &reqs[1]);

            MPI_Isend(C.data(), NX, MPI_DOUBLE, LEFT, 2423, MPI_COMM_WORLD, &reqs[0]);
      
            MPI_Wait(reqs+1, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    time_mpitime.stop();

    time_mpitime.print(std::cout);

    mpirc = MPI_Finalize();

    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Finalize failed!!!" << std::endl;
    }

    return 0;
}
