#ifndef __DIFFUSE_PHOTON_MPI_H__
#define __DIFFUSE_PHOTON_MPI_H__

#include <mpi.h>

struct dp_mpi_param {
  MPI_Datatype ray_info_type;
};

int  mpi_rank(int,int,int);
void set_dp_mpi_type(struct dp_mpi_param*);

void set_mpi_window(struct ray_face*, MPI_Win*, MPI_Info);
void free_mpi_window(MPI_Win*);

void mpi_win_fence(int, MPI_Win*) ;
void mpi_put_to_target(struct ray_face*, int*, MPI_Win*, struct dp_mpi_param*);
void mpi_win_recv(struct ray_face*, struct ray_face*, int*);

#endif  // __DIFFUSE_PHOTON_MPI_H__
