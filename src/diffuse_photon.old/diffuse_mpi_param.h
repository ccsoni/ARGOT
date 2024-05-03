#ifndef __DIFFUSE_PHOTON_MPI__
#define __DIFFUSE_PHOTON_MPI__

#include <mpi.h>

struct diffuse_mpi_param {
  MPI_Datatype ray_info_type;
};

#endif  //__DIFFUSE_PHOTON_MPI__
