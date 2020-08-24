add_executable(batchFD
               ${CMAKE_SOURCE_DIR}/benchmarks/batchFD.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph4.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Lap.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph2.cc
               ${CMAKE_SOURCE_DIR}/src/pb/FDoper.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Grid.cc
               ${CMAKE_SOURCE_DIR}/src/pb/PEenv.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFunc.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFuncVector.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc)

add_executable(MPIp2p
               ${CMAKE_SOURCE_DIR}/benchmarks/MPIp2p.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph4.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Lap.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph2.cc
               ${CMAKE_SOURCE_DIR}/src/pb/FDoper.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Grid.cc
               ${CMAKE_SOURCE_DIR}/src/pb/PEenv.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFunc.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFuncVector.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc)

if(${MAGMA_FOUND})
  target_link_libraries(batchFD PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(MPIp2p PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
else()
  target_link_libraries(batchFD PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX)
  target_link_libraries(MPIp2p PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()
