#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <omp.h>

#include "diffuse_photon.h"
#include "diffuse_photon_mpi.h"


void set_mpi_window(struct ray_face *recv_ray, MPI_Win *win_mwf, MPI_Info info)
{
  MPI_Win_create( recv_ray->xy, NMESH_MAX_FACE*sizeof(struct ray_info), sizeof(struct ray_info),
  		  info, MPI_COMM_WORLD, &win_mwf[0] );
  MPI_Win_fence( MPI_MODE_NOPRECEDE, win_mwf[0] );
  
  MPI_Win_create( recv_ray->yz, NMESH_MAX_FACE*sizeof(struct ray_info), sizeof(struct ray_info),
  		  info, MPI_COMM_WORLD, &win_mwf[1] );
  MPI_Win_fence( MPI_MODE_NOPRECEDE, win_mwf[1] );
  
  MPI_Win_create( recv_ray->zx, NMESH_MAX_FACE*sizeof(struct ray_info), sizeof(struct ray_info),
  		  info, MPI_COMM_WORLD, &win_mwf[2] );
  MPI_Win_fence( MPI_MODE_NOPRECEDE, win_mwf[2] );
}


void free_mpi_window(MPI_Win *win_mwf)
{
  MPI_Win_free(&win_mwf[0]);
  MPI_Win_free(&win_mwf[1]);
  MPI_Win_free(&win_mwf[2]);
}


void mpi_win_fence(int assert_type, MPI_Win *win_mwf) 
{
  MPI_Win_fence(assert_type, win_mwf[0]);
  MPI_Win_fence(assert_type, win_mwf[1]);
  MPI_Win_fence(assert_type, win_mwf[2]); 
}


void mpi_put_to_target(struct ray_face *start_ray, int *target_rank,
		       MPI_Win *win_mwf, struct dp_mpi_param *this_dp_mpi)
{
  if(target_rank[0] >= 0)
    MPI_Put(start_ray->xy, NMESH_MAX_FACE, this_dp_mpi->ray_info_type,
	    target_rank[0], 0, NMESH_MAX_FACE, this_dp_mpi->ray_info_type,
	    win_mwf[0]);
    
  if(target_rank[1] >= 0)
    MPI_Put(start_ray->yz, NMESH_MAX_FACE, this_dp_mpi->ray_info_type,
	    target_rank[1], 0, NMESH_MAX_FACE, this_dp_mpi->ray_info_type,
	    win_mwf[1]);
    
  if(target_rank[2] >= 0)
    MPI_Put(start_ray->zx, NMESH_MAX_FACE, this_dp_mpi->ray_info_type,
	    target_rank[2], 0, NMESH_MAX_FACE, this_dp_mpi->ray_info_type,
	    win_mwf[2]);
}



void mpi_win_recv(struct ray_face *start_ray, struct ray_face *recv_ray, 
		  int *source_rank)
{
  int ix,iy,iz;
  
  if(source_rank[0] >= 0) {
#pragma omp parallel for schedule(auto)
    for(iz=0; iz<NMESH_MAX_FACE; iz++){
      start_ray->xy[iz] = recv_ray->xy[iz];
    }
  }
  
  if(source_rank[1] >= 0) {
#pragma omp parallel for schedule(auto)
    for(ix=0; ix<NMESH_MAX_FACE; ix++){
      start_ray->yz[ix] = recv_ray->yz[ix];
    }
  }
  
  if(source_rank[2] >= 0) {
#pragma omp parallel for schedule(auto)
    for(iy=0; iy<NMESH_MAX_FACE; iy++){
      start_ray->zx[iy] = recv_ray->zx[iy];
    }
  }
  
}
