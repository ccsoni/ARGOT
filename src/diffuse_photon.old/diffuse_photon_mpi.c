#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "diffuse_photon.h"
#include "diffuse_photon_mpi.h"

void set_dp_mpi_type(struct dp_mpi_param *this_dp_mpi)
{
  /* define the MPI data type for ther ray_info structure */
  int blockcount[1];
  MPI_Datatype type[1];
  MPI_Aint adr[1];
  blockcount[0] = 4;
#ifdef __HELIUM__
  blockcount[0] += 2;
#endif

  type[0]=MPI_FLOAT;
  adr[0]=0;

  MPI_Type_struct(1,blockcount, adr, type, &(this_dp_mpi->ray_info_type));
  MPI_Type_commit(&(this_dp_mpi->ray_info_type));
}
