#ifndef __FFTW_MPI_PARAM__
#define __FFTW_MPI_PARAM__

#include <fftw3-mpi.h>

struct fftw_mpi_param {
  MPI_Comm grav_fftw_comm;
  MPI_Comm dens_reduction_comm;

  fftwf_plan forward_plan, backward_plan;

#ifdef __ISOLATED_GRAV__
  fftwf_plan green_func_plan;
  ptrdiff_t local_size_green, ix_start_green, ix_length_green;
#endif

  ptrdiff_t local_size, ix_start, ix_end, ix_length;
};

#endif /* __FFTW_MPI_PARAM__ */
