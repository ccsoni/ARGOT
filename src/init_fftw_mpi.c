#include <stdlib.h>
#include <assert.h>

#include "run_param.h"
#include "fftw_mpi_param.h"

void init_fftw_mpi(struct run_param *this_run, 
		   struct fftw_mpi_param *this_fftw_mpi, float **gk)
{
  int mpi_color;

  fftwf_mpi_init();

  /* MPI Communicator for FFTW MPI transform */
  mpi_color = this_run->rank_z + NNODE_Z*this_run->rank_y;
  MPI_Comm_split(MPI_COMM_WORLD, mpi_color, this_run->rank_x, 
		 &(this_fftw_mpi->grav_fftw_comm));

  /* MPI Communicator for reduction of mesh density */
  MPI_Comm_split(MPI_COMM_WORLD, this_run->rank_x, this_run->mpi_rank, 
		 &(this_fftw_mpi->dens_reduction_comm));

  /* Setting the local size of the density mesh grids */
  this_fftw_mpi->local_size = 
    fftwf_mpi_local_size_3d(NMESH_X_POTEN, NMESH_Y_POTEN, NMESH_Z_POTEN_P2,
			    this_fftw_mpi->grav_fftw_comm,
			    &(this_fftw_mpi->ix_length),
			    &(this_fftw_mpi->ix_start));

#ifdef __ISOLATED_GRAV__
  if(this_fftw_mpi->ix_start != 2*this_run->rank_x*NMESH_X_LOCAL) {
    fprintf(stderr, "# inconsistent ix_start (%d).\n",
	    this_fftw_mpi->ix_start);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if(this_fftw_mpi->ix_length != 2*NMESH_X_LOCAL) {
    fprintf(stderr, "# inconsistent ix_length (%d).\n", 
	    this_fftw_mpi->ix_length);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if(this_fftw_mpi->local_size 
     != 2*NMESH_X_LOCAL*NMESH_Y_POTEN*NMESH_Z_POTEN_P2) {
    fprintf(stderr, 
	    "# inconsistent local size (%d) of the FFTW array.\n",
	    this_fftw_mpi->local_size);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
#else
  if(this_fftw_mpi->ix_start != this_run->rank_x*NMESH_X_LOCAL) {
    fprintf(stderr, "# inconsistent ix_start (%d).\n",
	    this_fftw_mpi->ix_start);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if(this_fftw_mpi->ix_length != NMESH_X_LOCAL) {
    fprintf(stderr, "# inconsistent ix_length (%d).\n", 
	    this_fftw_mpi->ix_length);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if(this_fftw_mpi->local_size 
     != NMESH_X_LOCAL*NMESH_Y_TOTAL*NMESH_Z_TOTAL_P2) {
    fprintf(stderr, 
	    "# inconsistent local size (%d) of the FFTW array.\n",
	    this_fftw_mpi->local_size);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
#endif

  /* preparing the array for the Green function */
#ifdef __ISOLATED_GRAV__
  this_fftw_mpi->local_size_green =
    fftwf_mpi_local_size_3d(NMESH_X_GREEN, NMESH_Y_GREEN, NMESH_Z_GREEN_P2,
			    this_fftw_mpi->grav_fftw_comm,
			    &(this_fftw_mpi->ix_length_green),
			    &(this_fftw_mpi->ix_start_green));

  if(this_fftw_mpi->local_size_green != 
     NMESH_X_GREEN*NMESH_Y_GREEN*NMESH_Z_GREEN_P2/NNODE_X){
    fprintf(stderr,"# inconsistent local size (%d) of the Green function.\n",
	    this_fftw_mpi->local_size_green);
  }

  *gk = (float *)malloc(sizeof(float)*
			this_fftw_mpi->ix_length_green*NMESH_Y_GREEN*NMESH_Z_GREEN_P2);
#else
  *gk = (float *)malloc(sizeof(float)*
			this_fftw_mpi->ix_length*NMESH_Y_GREEN*NMESH_Z_GREEN);
#endif
  assert(*gk != NULL);
}
