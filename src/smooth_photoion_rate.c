#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <sys/times.h>

#include "constants.h"
#include "run_param.h"
#include "mpi_param.h"
#include "radiation.h"

#include "prototype.h"

#define __SMOOTHING__

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])
#define SMOOTH(ix,iy,iz) (smooth[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

#define GAMMA_ION(ix,iy,iz) (gamma[(ix)+1][(iy)+1][(iz)+1])

void send_photoion_data(struct photoion_rate*, struct cuda_mem_space*, 
			struct cuda_param*, struct run_param*);
void recv_photoion_data(struct photoion_rate*, struct cuda_mem_space*,
			struct cuda_param*, struct run_param*);

void smooth_photoion_rate(struct fluid_mesh *mesh, 
			  struct run_param *this_run, 
#ifdef __USE_GPU__
			  struct cuda_mem_space *cuda_mem,
			  struct cuda_param *this_cuda,
#endif /* __USE_GPU__*/
			  struct mpi_param *this_mpi)
{
  MPI_Win win_gamma;
  MPI_Info info;

  int target_rank, source_rank;

  static struct photoion_rate smooth[NMESH_LOCAL];

  static struct photoion_rate gamma[NMESH_X_LOCAL+2][NMESH_Y_LOCAL+2][NMESH_Z_LOCAL+2];
  static struct photoion_rate gamma_xy_send[NMESH_X_LOCAL][NMESH_Y_LOCAL];
  static struct photoion_rate gamma_xy_recv[NMESH_X_LOCAL][NMESH_Y_LOCAL];
  static struct photoion_rate gamma_yz_send[NMESH_Y_LOCAL][NMESH_Z_LOCAL];
  static struct photoion_rate gamma_yz_recv[NMESH_Y_LOCAL][NMESH_Z_LOCAL];
  static struct photoion_rate gamma_xz_send[NMESH_X_LOCAL][NMESH_Z_LOCAL];
  static struct photoion_rate gamma_xz_recv[NMESH_X_LOCAL][NMESH_Z_LOCAL];

  static struct photoion_rate gamma_xside_send[NMESH_X_LOCAL];
  static struct photoion_rate gamma_yside_send[NMESH_Y_LOCAL];
  static struct photoion_rate gamma_zside_send[NMESH_Z_LOCAL];
  static struct photoion_rate gamma_xside_recv[NMESH_X_LOCAL];
  static struct photoion_rate gamma_yside_recv[NMESH_Y_LOCAL];
  static struct photoion_rate gamma_zside_recv[NMESH_Z_LOCAL];

  struct photoion_rate gamma_apex_send, gamma_apex_recv;

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

  MPI_Info_create(&info);
  MPI_Info_set(info, "no_locks", "true");

  /* zero out the photoion_rate array */
#pragma omp parallel for
  for(int ix=0;ix<NMESH_X_LOCAL+2;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL+2;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL+2;iz++) {
	gamma[ix][iy][iz].GammaHI=0.0;
	gamma[ix][iy][iz].HeatHI=0.0;

#ifdef __HELIUM__
	gamma[ix][iy][iz].GammaHeI=0.0;
	gamma[ix][iy][iz].GammaHeII=0.0;
	gamma[ix][iy][iz].HeatHeI=0.0;
	gamma[ix][iy][iz].HeatHeII=0.0;
#endif //__HELIUM__

#ifdef __HYDROGEN_MOL__
	gamma[ix][iy][iz].GammaHM=0.0;
	gamma[ix][iy][iz].GammaH2I_I=0.0;
	gamma[ix][iy][iz].GammaH2I_II=0.0;
	gamma[ix][iy][iz].GammaH2II_I=0.0;
	gamma[ix][iy][iz].GammaH2II_II=0.0;
	gamma[ix][iy][iz].HeatHM=0.0;
	gamma[ix][iy][iz].HeatH2I_I=0.0;
	gamma[ix][iy][iz].HeatH2I_II=0.0;
	gamma[ix][iy][iz].HeatH2II_I=0.0;
	gamma[ix][iy][iz].HeatH2II_II=0.0;
#endif
      }
    }
  }

#ifdef __USE_GPU__
  /* receive the photoion rate array from the GPUs to the "smooth" array. */
  recv_photoion_data(smooth, cuda_mem, this_cuda, this_run);
  /* copy the received data to the photoion_rate array */
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	GAMMA_ION(ix,iy,iz) = SMOOTH(ix,iy,iz);
      }
    }
  }
#else /* !__USE_GPU__ */
#pragma omp parallel for
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	GAMMA_ION(ix,iy,iz).GammaHI = MESH(ix, iy, iz).prev_chem.GammaHI;
	GAMMA_ION(ix,iy,iz).HeatHI  = MESH(ix, iy, iz).prev_chem.HeatHI;
#ifdef __HELIUM__
	GAMMA_ION(ix,iy,iz).GammaHeI  = MESH(ix, iy, iz).prev_chem.GammaHeI;
	GAMMA_ION(ix,iy,iz).GammaHeII = MESH(ix, iy, iz).prev_chem.GammaHeII;
	GAMMA_ION(ix,iy,iz).HeatHeI   = MESH(ix, iy, iz).prev_chem.HeatHeI;
	GAMMA_ION(ix,iy,iz).HeatHeII  = MESH(ix, iy, iz).prev_chem.HeatHeII;
#endif //__HELIUM__
#ifdef __HYDROGEN_MOL__
	GAMMA_ION(ix,iy,iz).GammaHM      = MESH(ix, iy, iz).prev_chem.GammaHM;
	GAMMA_ION(ix,iy,iz).GammaH2I_I   = MESH(ix, iy, iz).prev_chem.GammaH2I_I;
	GAMMA_ION(ix,iy,iz).GammaH2I_II  = MESH(ix, iy, iz).prev_chem.GammaH2I_II;
	GAMMA_ION(ix,iy,iz).GammaH2II_I  = MESH(ix, iy, iz).prev_chem.GammaH2II_I;
	GAMMA_ION(ix,iy,iz).GammaH2II_II = MESH(ix, iy, iz).prev_chem.GammaH2II_II;
	GAMMA_ION(ix,iy,iz).HeatHM       = MESH(ix, iy, iz).prev_chem.HeatHM;
	GAMMA_ION(ix,iy,iz).HeatH2I_I    = MESH(ix, iy, iz).prev_chem.HeatH2I_I;
	GAMMA_ION(ix,iy,iz).HeatH2I_II   = MESH(ix, iy, iz).prev_chem.HeatH2I_II;
	GAMMA_ION(ix,iy,iz).HeatH2II_I   = MESH(ix, iy, iz).prev_chem.HeatH2II_I;
	GAMMA_ION(ix,iy,iz).HeatH2II_II  = MESH(ix, iy, iz).prev_chem.HeatH2II_II;
#endif
      }
    }
  }
#endif /* __USE_GPU__ */

  // along X-direction
  MPI_Win_create(gamma_yz_recv,NMESH_Y_LOCAL*NMESH_Z_LOCAL*sizeof(struct photoion_rate), 
		 sizeof(struct photoion_rate),
		 info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  // toward +X-direction
  target_rank = mpi_rank(this_run->rank_x+1, 
			 this_run->rank_y,
			 this_run->rank_z);
  
  if(target_rank >= 0) {
#pragma omp parallel for
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	gamma_yz_send[iy][iz] = GAMMA_ION(NMESH_X_LOCAL-1,iy,iz);
      }
    }

    MPI_Put(gamma_yz_send, NMESH_Y_LOCAL*NMESH_Z_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_Y_LOCAL*NMESH_Z_LOCAL, this_mpi->photoion_rate_type,
	    win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  source_rank = mpi_rank(this_run->rank_x-1, 
			 this_run->rank_y,
			 this_run->rank_z);

  if(source_rank >= 0) {
#pragma omp parallel for
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	GAMMA_ION(-1, iy, iz) = gamma_yz_recv[iy][iz];
      }
    }
  }else{
#pragma omp parallel for
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	GAMMA_ION(-1, iy, iz) = GAMMA_ION(0, iy, iz);
      }
    }
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  // toward -X-direction
  target_rank = mpi_rank(this_run->rank_x-1, 
			 this_run->rank_y,
			 this_run->rank_z);

  if(target_rank >= 0) {
#pragma omp parallel for
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	gamma_yz_send[iy][iz] = GAMMA_ION(0,iy,iz);
      }
    }
    
    MPI_Put(gamma_yz_send, NMESH_Y_LOCAL*NMESH_Z_LOCAL, this_mpi->photoion_rate_type, 
	    target_rank, 0, NMESH_Y_LOCAL*NMESH_Z_LOCAL, this_mpi->photoion_rate_type,
	    win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x+1, 
			 this_run->rank_y,
			 this_run->rank_z);
  
  if(source_rank >= 0) {
#pragma omp parallel for
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	GAMMA_ION(NMESH_X_LOCAL, iy, iz) = gamma_yz_recv[iy][iz];
      }
    }
  }else{
#pragma omp parallel for
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	GAMMA_ION(NMESH_X_LOCAL, iy, iz) = GAMMA_ION(NMESH_X_LOCAL-1, iy, iz);
      }
    }
  }

  MPI_Win_free(&win_gamma);


  //along Y-direction
  MPI_Win_create(gamma_xz_recv,NMESH_X_LOCAL*NMESH_Z_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate),
		 info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  // toward +Y-direction
  target_rank = mpi_rank(this_run->rank_x, 
			 this_run->rank_y+1,
			 this_run->rank_z);

  if(target_rank >= 0) {
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	gamma_xz_send[ix][iz] = GAMMA_ION(ix, NMESH_Y_LOCAL-1, iz);
      }
    }

    MPI_Put(gamma_xz_send, NMESH_X_LOCAL*NMESH_Z_LOCAL, this_mpi->photoion_rate_type, 
	    target_rank, 0, NMESH_X_LOCAL*NMESH_Z_LOCAL, this_mpi->photoion_rate_type, 
	    win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  source_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y-1,
			 this_run->rank_z);

  if(source_rank >= 0) {
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	GAMMA_ION(ix, -1, iz) = gamma_xz_recv[ix][iz];
      }
    }
  }else{
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	GAMMA_ION(ix, -1, iz) = GAMMA_ION(ix, 0, iz);
      }
    } 
  }

  // toward -Y-direction
  target_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y-1,
			 this_run->rank_z);

  if(target_rank >= 0) {
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	gamma_xz_send[ix][iz] = GAMMA_ION(ix, 0, iz);
      }
    }

    MPI_Put(gamma_xz_send, NMESH_X_LOCAL*NMESH_Z_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_X_LOCAL*NMESH_Z_LOCAL, this_mpi->photoion_rate_type,
	    win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y+1,
			 this_run->rank_z);

  if(source_rank >= 0) {
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	GAMMA_ION(ix, NMESH_Y_LOCAL, iz) = gamma_xz_recv[ix][iz];
      }
    }
  }else{
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	GAMMA_ION(ix, NMESH_Y_LOCAL, iz) = GAMMA_ION(ix, NMESH_Y_LOCAL-1, iz);
      }
    }
  }

  MPI_Win_free(&win_gamma);


  // along Z-direction
  MPI_Win_create(gamma_xy_recv,NMESH_X_LOCAL*NMESH_Y_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate),
		 info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  // toward +Z-direction
  target_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y,
			 this_run->rank_z+1);

  if(target_rank >= 0) {
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
	gamma_xy_send[ix][iy] = GAMMA_ION(ix, iy, NMESH_Z_LOCAL-1);
      }
    }
    
    MPI_Put(gamma_xy_send, NMESH_X_LOCAL*NMESH_Y_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_X_LOCAL*NMESH_Y_LOCAL, this_mpi->photoion_rate_type, 
	    win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  source_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y,
			 this_run->rank_z-1);

  if(source_rank >= 0) {
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
	GAMMA_ION(ix, iy, -1) = gamma_xy_recv[ix][iy];
      }
    }
  }else{
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
	GAMMA_ION(ix, iy, -1) = GAMMA_ION(ix, iy, 0);
      }
    }
  }

  // toward -Z-direction
  target_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y,
			 this_run->rank_z-1);

  if(target_rank >= 0) {
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
	gamma_xy_send[ix][iy] = GAMMA_ION(ix, iy, 0);
      }
    }

    MPI_Put(gamma_xy_send, NMESH_X_LOCAL*NMESH_Y_LOCAL, this_mpi->photoion_rate_type, 
	    target_rank, 0, NMESH_X_LOCAL*NMESH_Y_LOCAL, this_mpi->photoion_rate_type, 
	    win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y,
			 this_run->rank_z+1);

  if(source_rank >= 0) {
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
	GAMMA_ION(ix, iy, NMESH_Z_LOCAL) = gamma_xy_recv[ix][iy];
      }
    }
  }else{
#pragma omp parallel for
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
	GAMMA_ION(ix, iy, NMESH_Z_LOCAL) = GAMMA_ION(ix, iy, NMESH_Z_LOCAL-1);
      }
    }
  }

  MPI_Win_free(&win_gamma);

  // Communication of data at the sides of the domains
  
  // X-side: +Y and +Z direction
  MPI_Win_create(&gamma_xside_recv, NMESH_X_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y+1,
			 this_run->rank_z+1);

  if(target_rank >= 0) {
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      gamma_xside_send[ix] = GAMMA_ION(ix, NMESH_Y_LOCAL-1, NMESH_Z_LOCAL-1);
    }
    MPI_Put(gamma_xside_send, NMESH_X_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_X_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y-1,
			 this_run->rank_z-1);

  if(source_rank >= 0) {
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      GAMMA_ION(ix, -1, -1) = gamma_xside_recv[ix];
    }
  }else{
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      GAMMA_ION(ix, -1, -1) = GAMMA_ION(ix, 0, 0);
    }
  }

  MPI_Win_free(&win_gamma);
  
  // X-side: +Y and -Z direction
  MPI_Win_create(&gamma_xside_recv, NMESH_X_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y+1,
			 this_run->rank_z-1);

  if(target_rank >= 0) {
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      gamma_xside_send[ix] = GAMMA_ION(ix, NMESH_Y_LOCAL-1, 0);
    }
    MPI_Put(gamma_xside_send, NMESH_X_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_X_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y-1,
			 this_run->rank_z+1);

  if(source_rank >= 0) {
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      GAMMA_ION(ix, -1, NMESH_Z_LOCAL) = gamma_xside_recv[ix];
    }
  }else{
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      GAMMA_ION(ix, -1, NMESH_Z_LOCAL) = GAMMA_ION(ix, 0, NMESH_Z_LOCAL-1);
    }
  }

  MPI_Win_free(&win_gamma);

  // X-side: -Y and +Z direction
  MPI_Win_create(&gamma_xside_recv, NMESH_X_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y-1,
			 this_run->rank_z+1);

  if(target_rank >= 0) {
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      gamma_xside_send[ix] = GAMMA_ION(ix, 0, NMESH_Z_LOCAL-1);
    }
    MPI_Put(gamma_xside_send, NMESH_X_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_X_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y+1,
			 this_run->rank_z-1);

  if(source_rank >= 0) {
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      GAMMA_ION(ix, NMESH_Y_LOCAL, -1) = gamma_xside_recv[ix];
    }
  }else{
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      GAMMA_ION(ix, NMESH_Y_LOCAL, -1) = GAMMA_ION(ix, NMESH_Y_LOCAL-1, 0);
    }
  }

  MPI_Win_free(&win_gamma);

  // X-side: -Y and -Z direction
  MPI_Win_create(&gamma_xside_recv, NMESH_X_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y-1,
			 this_run->rank_z-1);

  if(target_rank >= 0) {
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      gamma_xside_send[ix] = GAMMA_ION(ix, 0, 0);
    }
    MPI_Put(gamma_xside_send, NMESH_X_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_X_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x,
			 this_run->rank_y+1,
			 this_run->rank_z+1);

  if(source_rank >= 0) {
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      GAMMA_ION(ix, NMESH_Y_LOCAL, NMESH_Z_LOCAL) = gamma_xside_recv[ix];
    }
  }else{
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      GAMMA_ION(ix, NMESH_Y_LOCAL, NMESH_Z_LOCAL) = GAMMA_ION(ix, NMESH_Y_LOCAL-1, NMESH_Z_LOCAL-1);
    }
  }

  MPI_Win_free(&win_gamma);

  // Y-side: +X and +Z direction
  MPI_Win_create(&gamma_yside_recv, NMESH_Y_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y,
			 this_run->rank_z+1);

  if(target_rank >= 0) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      gamma_yside_send[iy] = GAMMA_ION(NMESH_X_LOCAL-1, iy, NMESH_Z_LOCAL-1);
    }
    MPI_Put(gamma_yside_send, NMESH_Y_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_Y_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y,
			 this_run->rank_z-1);

  if(source_rank >= 0) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      GAMMA_ION(-1, iy, -1) = gamma_yside_recv[iy];
    }
  }else{
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      GAMMA_ION(-1, iy, -1) = GAMMA_ION(0, iy, 0);
    }
  }

  MPI_Win_free(&win_gamma);

  // Y-side: +X and -Z direction
  MPI_Win_create(&gamma_yside_recv, NMESH_Y_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y,
			 this_run->rank_z-1);

  if(target_rank >= 0) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      gamma_yside_send[iy] = GAMMA_ION(NMESH_X_LOCAL-1, iy, 0);
    }
    MPI_Put(gamma_yside_send, NMESH_Y_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_Y_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y,
			 this_run->rank_z+1);

  if(source_rank >= 0) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      GAMMA_ION(-1, iy, NMESH_Z_LOCAL) = gamma_yside_recv[iy];
    }
  }else{
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      GAMMA_ION(-1, iy, NMESH_Z_LOCAL) = GAMMA_ION(0, iy, NMESH_Z_LOCAL-1);
    }
  }

  MPI_Win_free(&win_gamma);

  // Y-side: -X and +Z direction
  MPI_Win_create(&gamma_yside_recv, NMESH_Y_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y,
			 this_run->rank_z+1);

  if(target_rank >= 0) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      gamma_yside_send[iy] = GAMMA_ION(0, iy, NMESH_Z_LOCAL-1);
    }
    MPI_Put(gamma_yside_send, NMESH_Y_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_Y_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y,
			 this_run->rank_z-1);

  if(source_rank >= 0) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      GAMMA_ION(NMESH_X_LOCAL, iy, -1) = gamma_yside_recv[iy];
    }
  }else{
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      GAMMA_ION(NMESH_X_LOCAL, iy, -1) = GAMMA_ION(NMESH_X_LOCAL-1, iy, 0);
    }
  }

  MPI_Win_free(&win_gamma);

  // Y-side: -X and -Z direction
  MPI_Win_create(&gamma_yside_recv, NMESH_Y_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y,
			 this_run->rank_z-1);

  if(target_rank >= 0) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      gamma_yside_send[iy] = GAMMA_ION(0, iy, 0);
    }
    MPI_Put(gamma_yside_send, NMESH_Y_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_Y_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y,
			 this_run->rank_z+1);

  if(source_rank >= 0) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      GAMMA_ION(NMESH_X_LOCAL, iy, NMESH_Z_LOCAL) = gamma_yside_recv[iy];
    }
  }else{
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      GAMMA_ION(NMESH_X_LOCAL, iy, NMESH_Z_LOCAL) = GAMMA_ION(NMESH_X_LOCAL-1, iy, NMESH_Z_LOCAL-1);
    }
  }

  MPI_Win_free(&win_gamma);

  // Z-side: +X and +Y direction
  MPI_Win_create(&gamma_zside_recv, NMESH_Z_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y+1,
			 this_run->rank_z);

  if(target_rank >= 0) {
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      gamma_zside_send[iz] = GAMMA_ION(NMESH_X_LOCAL-1, NMESH_Y_LOCAL-1, iz);
    }
    MPI_Put(gamma_zside_send, NMESH_Z_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_Z_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y-1,
			 this_run->rank_z);

  if(source_rank >= 0) {
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      GAMMA_ION(-1, -1, iz) = gamma_zside_recv[iz];
    }
  }else{
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      GAMMA_ION(-1, -1, iz) = GAMMA_ION(0, 0, iz);
    }
  }

  MPI_Win_free(&win_gamma);

  // Z-side: +X and -Y direction
  MPI_Win_create(&gamma_zside_recv, NMESH_Z_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y-1,
			 this_run->rank_z);

  if(target_rank >= 0) {
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      gamma_zside_send[iz] = GAMMA_ION(NMESH_X_LOCAL-1, 0, iz);
    }
    MPI_Put(gamma_zside_send, NMESH_Z_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_Z_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y+1,
			 this_run->rank_z);

  if(source_rank >= 0) {
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      GAMMA_ION(-1, NMESH_Y_LOCAL, iz) = gamma_zside_recv[iz];
    }
  }else{
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      GAMMA_ION(-1, NMESH_Y_LOCAL, iz) = GAMMA_ION(0, NMESH_Y_LOCAL-1, iz);
    }
  }

  MPI_Win_free(&win_gamma);

  // Z-side: -X and +Y direction
  MPI_Win_create(&gamma_zside_recv, NMESH_Z_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y+1,
			 this_run->rank_z);

  if(target_rank >= 0) {
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      gamma_zside_send[iz] = GAMMA_ION(0, NMESH_Y_LOCAL-1, iz);
    }
    MPI_Put(gamma_zside_send, NMESH_Z_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_Z_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y-1,
			 this_run->rank_z);

  if(source_rank >= 0) {
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      GAMMA_ION(NMESH_X_LOCAL, -1, iz) = gamma_zside_recv[iz];
    }
  }else{
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      GAMMA_ION(NMESH_X_LOCAL, -1, iz) = GAMMA_ION(NMESH_X_LOCAL-1, 0, iz);
    }
  }

  MPI_Win_free(&win_gamma);

  // Z-side: -X and -Y direction
  MPI_Win_create(&gamma_zside_recv, NMESH_Z_LOCAL*sizeof(struct photoion_rate),
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y-1,
			 this_run->rank_z);

  if(target_rank >= 0) {
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      gamma_zside_send[iz] = GAMMA_ION(0, 0, iz);
    }
    MPI_Put(gamma_zside_send, NMESH_Z_LOCAL, this_mpi->photoion_rate_type,
	    target_rank, 0, NMESH_Z_LOCAL, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_gamma);

  source_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y+1,
			 this_run->rank_z);

  if(source_rank >= 0) {
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      GAMMA_ION(NMESH_X_LOCAL, NMESH_Y_LOCAL, iz) = gamma_zside_recv[iz];
    }
  }else{
    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
      GAMMA_ION(NMESH_X_LOCAL, NMESH_Y_LOCAL, iz) = GAMMA_ION(NMESH_X_LOCAL-1, NMESH_Y_LOCAL-1, iz);
    }
  }

  MPI_Win_free(&win_gamma);

  // Communication of data at the apexes of the domains

  // sending to +++ direction 
  MPI_Win_create(&gamma_apex_recv, sizeof(struct photoion_rate), 
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y+1,
			 this_run->rank_z+1);

  if(target_rank >= 0) {
    gamma_apex_send = GAMMA_ION(NMESH_X_LOCAL-1,NMESH_Y_LOCAL-1,NMESH_Z_LOCAL-1);
    MPI_Put(&gamma_apex_send, 1, this_mpi->photoion_rate_type, 
	    target_rank, 0, 1, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  source_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y-1,
			 this_run->rank_z-1);
  if(source_rank >= 0) {
    GAMMA_ION(-1,-1,-1) = gamma_apex_recv;
  }else{
    GAMMA_ION(-1,-1,-1) = GAMMA_ION(0,0,0);
  }
  
  MPI_Win_free(&win_gamma);

  // sending to ++- direction 
  MPI_Win_create(&gamma_apex_recv, sizeof(struct photoion_rate), 
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y+1,
			 this_run->rank_z-1);

  if(target_rank >= 0) {
    gamma_apex_send = GAMMA_ION(NMESH_X_LOCAL-1,NMESH_Y_LOCAL-1,0);
    MPI_Put(&gamma_apex_send, 1, this_mpi->photoion_rate_type, 
	    target_rank, 0, 1, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  source_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y-1,
			 this_run->rank_z+1);
  if(source_rank >= 0) {
    GAMMA_ION(-1,-1,NMESH_Z_LOCAL) = gamma_apex_recv;
  }else{
    GAMMA_ION(-1,-1,NMESH_Z_LOCAL) = GAMMA_ION(0,0,NMESH_Z_LOCAL-1);
  }
  
  MPI_Win_free(&win_gamma);

  // sending to +-+ direction 
  MPI_Win_create(&gamma_apex_recv, sizeof(struct photoion_rate), 
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y-1,
			 this_run->rank_z+1);

  if(target_rank >= 0) {
    gamma_apex_send = GAMMA_ION(NMESH_X_LOCAL-1,0,NMESH_Z_LOCAL-1);
    MPI_Put(&gamma_apex_send, 1, this_mpi->photoion_rate_type, 
	    target_rank, 0, 1, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  source_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y+1,
			 this_run->rank_z-1);
  if(source_rank >= 0) {
    GAMMA_ION(-1,NMESH_Y_LOCAL,-1) = gamma_apex_recv;
  }else{
    GAMMA_ION(-1,NMESH_Y_LOCAL,-1) = GAMMA_ION(0,NMESH_Y_LOCAL-1,0);
  }
  
  MPI_Win_free(&win_gamma);

  // sending to -++ direction 
  MPI_Win_create(&gamma_apex_recv, sizeof(struct photoion_rate), 
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y+1,
			 this_run->rank_z+1);

  if(target_rank >= 0) {
    gamma_apex_send = GAMMA_ION(0,NMESH_Y_LOCAL-1,NMESH_Z_LOCAL-1);
    MPI_Put(&gamma_apex_send, 1, this_mpi->photoion_rate_type, 
	    target_rank, 0, 1, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  source_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y-1,
			 this_run->rank_z-1);
  if(source_rank >= 0) {
    GAMMA_ION(NMESH_X_LOCAL,-1,-1) = gamma_apex_recv;
  }else{
    GAMMA_ION(NMESH_X_LOCAL,-1,-1) = GAMMA_ION(NMESH_X_LOCAL-1,0,0);
  }
  
  MPI_Win_free(&win_gamma);

  // sending to +-- direction 
  MPI_Win_create(&gamma_apex_recv, sizeof(struct photoion_rate), 
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y-1,
			 this_run->rank_z-1);

  if(target_rank >= 0) {
    gamma_apex_send = GAMMA_ION(NMESH_X_LOCAL-1,0,0);
    MPI_Put(&gamma_apex_send, 1, this_mpi->photoion_rate_type, 
	    target_rank, 0, 1, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  source_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y+1,
			 this_run->rank_z+1);
  if(source_rank >= 0) {
    GAMMA_ION(-1,NMESH_Y_LOCAL,NMESH_Z_LOCAL) = gamma_apex_recv;
  }else{
    GAMMA_ION(-1,NMESH_Y_LOCAL,NMESH_Z_LOCAL) = GAMMA_ION(0,NMESH_Y_LOCAL-1,NMESH_Z_LOCAL-1);
  }
  
  MPI_Win_free(&win_gamma);

  // sending to -+- direction 
  MPI_Win_create(&gamma_apex_recv, sizeof(struct photoion_rate), 
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y+1,
			 this_run->rank_z-1);

  if(target_rank >= 0) {
    gamma_apex_send = GAMMA_ION(0,NMESH_Y_LOCAL-1,0);
    MPI_Put(&gamma_apex_send, 1, this_mpi->photoion_rate_type, 
	    target_rank, 0, 1, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  source_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y-1,
			 this_run->rank_z+1);
  if(source_rank >= 0) {
    GAMMA_ION(NMESH_X_LOCAL,-1,NMESH_Z_LOCAL) = gamma_apex_recv;
  }else{
    GAMMA_ION(NMESH_X_LOCAL,-1,NMESH_Z_LOCAL) = GAMMA_ION(NMESH_X_LOCAL-1,0,NMESH_Z_LOCAL-1);
  }
  
  MPI_Win_free(&win_gamma);

  // sending to --+ direction 
  MPI_Win_create(&gamma_apex_recv, sizeof(struct photoion_rate), 
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y-1,
			 this_run->rank_z+1);

  if(target_rank >= 0) {
    gamma_apex_send = GAMMA_ION(0,0,NMESH_Z_LOCAL-1);
    MPI_Put(&gamma_apex_send, 1, this_mpi->photoion_rate_type, 
	    target_rank, 0, 1, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  source_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y+1,
			 this_run->rank_z-1);
  if(source_rank >= 0) {
    GAMMA_ION(NMESH_X_LOCAL,NMESH_Y_LOCAL,-1) = gamma_apex_recv;
  }else{
    GAMMA_ION(NMESH_X_LOCAL,NMESH_Y_LOCAL,-1) = GAMMA_ION(NMESH_X_LOCAL-1,NMESH_Y_LOCAL-1,0);
  }
  
  MPI_Win_free(&win_gamma);

  // sending to --- direction 
  MPI_Win_create(&gamma_apex_recv, sizeof(struct photoion_rate), 
		 sizeof(struct photoion_rate), info, MPI_COMM_WORLD, &win_gamma);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_gamma);

  target_rank = mpi_rank(this_run->rank_x-1,
			 this_run->rank_y-1,
			 this_run->rank_z-1);

  if(target_rank >= 0) {
    gamma_apex_send = GAMMA_ION(0,0,0);
    MPI_Put(&gamma_apex_send, 1, this_mpi->photoion_rate_type, 
	    target_rank, 0, 1, this_mpi->photoion_rate_type, win_gamma);
  }

  MPI_Win_fence(MPI_MODE_NOSTORE, win_gamma);

  source_rank = mpi_rank(this_run->rank_x+1,
			 this_run->rank_y+1,
			 this_run->rank_z+1);
  if(source_rank >= 0) {
    GAMMA_ION(NMESH_X_LOCAL,NMESH_Y_LOCAL,NMESH_Z_LOCAL) = gamma_apex_recv;
  }else{
    GAMMA_ION(NMESH_X_LOCAL,NMESH_Y_LOCAL,NMESH_Z_LOCAL) = GAMMA_ION(NMESH_X_LOCAL-1,NMESH_Y_LOCAL-1,NMESH_Z_LOCAL-1);
  }
  
  MPI_Win_free(&win_gamma);

  MPI_Info_free(&info);

  // smooth the photo-ionization rate
#define CENTRAL_WEIGHT (30.0)
#define NORMAL (1.0/(CENTRAL_WEIGHT+26.0))
#pragma omp parallel for
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {

#ifdef __SMOOTHING__
	SMOOTH(ix,iy,iz).GammaHI = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).GammaHI+
					   GAMMA_ION(ix-1,iy-1,iz  ).GammaHI+
					   GAMMA_ION(ix-1,iy-1,iz+1).GammaHI+
					   GAMMA_ION(ix-1,iy  ,iz-1).GammaHI+
					   GAMMA_ION(ix-1,iy  ,iz  ).GammaHI+
					   GAMMA_ION(ix-1,iy  ,iz+1).GammaHI+
					   GAMMA_ION(ix-1,iy+1,iz-1).GammaHI+
					   GAMMA_ION(ix-1,iy+1,iz  ).GammaHI+
					   GAMMA_ION(ix-1,iy+1,iz+1).GammaHI+
					   GAMMA_ION(ix  ,iy-1,iz-1).GammaHI+
					   GAMMA_ION(ix  ,iy-1,iz  ).GammaHI+
					   GAMMA_ION(ix  ,iy-1,iz+1).GammaHI+
					   GAMMA_ION(ix  ,iy  ,iz-1).GammaHI+
   			    CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).GammaHI+
					   GAMMA_ION(ix  ,iy  ,iz+1).GammaHI+
					   GAMMA_ION(ix  ,iy+1,iz-1).GammaHI+
					   GAMMA_ION(ix  ,iy+1,iz  ).GammaHI+
					   GAMMA_ION(ix  ,iy+1,iz+1).GammaHI+
					   GAMMA_ION(ix+1,iy-1,iz-1).GammaHI+
					   GAMMA_ION(ix+1,iy-1,iz  ).GammaHI+
					   GAMMA_ION(ix+1,iy-1,iz+1).GammaHI+
					   GAMMA_ION(ix+1,iy  ,iz-1).GammaHI+
					   GAMMA_ION(ix+1,iy  ,iz  ).GammaHI+
					   GAMMA_ION(ix+1,iy  ,iz+1).GammaHI+
					   GAMMA_ION(ix+1,iy+1,iz-1).GammaHI+
					   GAMMA_ION(ix+1,iy+1,iz  ).GammaHI+
					   GAMMA_ION(ix+1,iy+1,iz+1).GammaHI);

	SMOOTH(ix,iy,iz).HeatHI = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).HeatHI+
					  GAMMA_ION(ix-1,iy-1,iz  ).HeatHI+
					  GAMMA_ION(ix-1,iy-1,iz+1).HeatHI+
					  GAMMA_ION(ix-1,iy  ,iz-1).HeatHI+
					  GAMMA_ION(ix-1,iy  ,iz  ).HeatHI+
					  GAMMA_ION(ix-1,iy  ,iz+1).HeatHI+
					  GAMMA_ION(ix-1,iy+1,iz-1).HeatHI+
					  GAMMA_ION(ix-1,iy+1,iz  ).HeatHI+
					  GAMMA_ION(ix-1,iy+1,iz+1).HeatHI+
					  GAMMA_ION(ix  ,iy-1,iz-1).HeatHI+
					  GAMMA_ION(ix  ,iy-1,iz  ).HeatHI+
					  GAMMA_ION(ix  ,iy-1,iz+1).HeatHI+
					  GAMMA_ION(ix  ,iy  ,iz-1).HeatHI+
			   CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).HeatHI+
					  GAMMA_ION(ix  ,iy  ,iz+1).HeatHI+
					  GAMMA_ION(ix  ,iy+1,iz-1).HeatHI+
					  GAMMA_ION(ix  ,iy+1,iz  ).HeatHI+
					  GAMMA_ION(ix  ,iy+1,iz+1).HeatHI+
					  GAMMA_ION(ix+1,iy-1,iz-1).HeatHI+
					  GAMMA_ION(ix+1,iy-1,iz  ).HeatHI+
					  GAMMA_ION(ix+1,iy-1,iz+1).HeatHI+
					  GAMMA_ION(ix+1,iy  ,iz-1).HeatHI+
					  GAMMA_ION(ix+1,iy  ,iz  ).HeatHI+
					  GAMMA_ION(ix+1,iy  ,iz+1).HeatHI+
					  GAMMA_ION(ix+1,iy+1,iz-1).HeatHI+
					  GAMMA_ION(ix+1,iy+1,iz  ).HeatHI+
					  GAMMA_ION(ix+1,iy+1,iz+1).HeatHI);
#ifdef __HELIUM__
	SMOOTH(ix,iy,iz).GammaHeI = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).GammaHeI+
					    GAMMA_ION(ix-1,iy-1,iz  ).GammaHeI+
					    GAMMA_ION(ix-1,iy-1,iz+1).GammaHeI+
					    GAMMA_ION(ix-1,iy  ,iz-1).GammaHeI+
					    GAMMA_ION(ix-1,iy  ,iz  ).GammaHeI+
					    GAMMA_ION(ix-1,iy  ,iz+1).GammaHeI+
					    GAMMA_ION(ix-1,iy+1,iz-1).GammaHeI+
					    GAMMA_ION(ix-1,iy+1,iz  ).GammaHeI+
					    GAMMA_ION(ix-1,iy+1,iz+1).GammaHeI+
					    GAMMA_ION(ix  ,iy-1,iz-1).GammaHeI+
					    GAMMA_ION(ix  ,iy-1,iz  ).GammaHeI+
					    GAMMA_ION(ix  ,iy-1,iz+1).GammaHeI+
					    GAMMA_ION(ix  ,iy  ,iz-1).GammaHeI+
		             CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).GammaHeI+
					    GAMMA_ION(ix  ,iy  ,iz+1).GammaHeI+
					    GAMMA_ION(ix  ,iy+1,iz-1).GammaHeI+
					    GAMMA_ION(ix  ,iy+1,iz  ).GammaHeI+
					    GAMMA_ION(ix  ,iy+1,iz+1).GammaHeI+
					    GAMMA_ION(ix+1,iy-1,iz-1).GammaHeI+
					    GAMMA_ION(ix+1,iy-1,iz  ).GammaHeI+
					    GAMMA_ION(ix+1,iy-1,iz+1).GammaHeI+
					    GAMMA_ION(ix+1,iy  ,iz-1).GammaHeI+
					    GAMMA_ION(ix+1,iy  ,iz  ).GammaHeI+
					    GAMMA_ION(ix+1,iy  ,iz+1).GammaHeI+
					    GAMMA_ION(ix+1,iy+1,iz-1).GammaHeI+
					    GAMMA_ION(ix+1,iy+1,iz  ).GammaHeI+
					    GAMMA_ION(ix+1,iy+1,iz+1).GammaHeI);

	SMOOTH(ix,iy,iz).GammaHeII = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).GammaHeII+
					     GAMMA_ION(ix-1,iy-1,iz  ).GammaHeII+
					     GAMMA_ION(ix-1,iy-1,iz+1).GammaHeII+
					     GAMMA_ION(ix-1,iy  ,iz-1).GammaHeII+
					     GAMMA_ION(ix-1,iy  ,iz  ).GammaHeII+
					     GAMMA_ION(ix-1,iy  ,iz+1).GammaHeII+
					     GAMMA_ION(ix-1,iy+1,iz-1).GammaHeII+
					     GAMMA_ION(ix-1,iy+1,iz  ).GammaHeII+
					     GAMMA_ION(ix-1,iy+1,iz+1).GammaHeII+
					     GAMMA_ION(ix  ,iy-1,iz-1).GammaHeII+
					     GAMMA_ION(ix  ,iy-1,iz  ).GammaHeII+
					     GAMMA_ION(ix  ,iy-1,iz+1).GammaHeII+
					     GAMMA_ION(ix  ,iy  ,iz-1).GammaHeII+
			      CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).GammaHeII+
					     GAMMA_ION(ix  ,iy  ,iz+1).GammaHeII+
					     GAMMA_ION(ix  ,iy+1,iz-1).GammaHeII+
					     GAMMA_ION(ix  ,iy+1,iz  ).GammaHeII+
					     GAMMA_ION(ix  ,iy+1,iz+1).GammaHeII+
					     GAMMA_ION(ix+1,iy-1,iz-1).GammaHeII+
					     GAMMA_ION(ix+1,iy-1,iz  ).GammaHeII+
					     GAMMA_ION(ix+1,iy-1,iz+1).GammaHeII+
					     GAMMA_ION(ix+1,iy  ,iz-1).GammaHeII+
					     GAMMA_ION(ix+1,iy  ,iz  ).GammaHeII+
					     GAMMA_ION(ix+1,iy  ,iz+1).GammaHeII+
					     GAMMA_ION(ix+1,iy+1,iz-1).GammaHeII+
					     GAMMA_ION(ix+1,iy+1,iz  ).GammaHeII+
					     GAMMA_ION(ix+1,iy+1,iz+1).GammaHeII);

	SMOOTH(ix,iy,iz).HeatHeI = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).HeatHeI+
					   GAMMA_ION(ix-1,iy-1,iz  ).HeatHeI+
					   GAMMA_ION(ix-1,iy-1,iz+1).HeatHeI+
					   GAMMA_ION(ix-1,iy  ,iz-1).HeatHeI+
					   GAMMA_ION(ix-1,iy  ,iz  ).HeatHeI+
					   GAMMA_ION(ix-1,iy  ,iz+1).HeatHeI+
					   GAMMA_ION(ix-1,iy+1,iz-1).HeatHeI+
					   GAMMA_ION(ix-1,iy+1,iz  ).HeatHeI+
					   GAMMA_ION(ix-1,iy+1,iz+1).HeatHeI+
					   GAMMA_ION(ix  ,iy-1,iz-1).HeatHeI+
					   GAMMA_ION(ix  ,iy-1,iz  ).HeatHeI+
					   GAMMA_ION(ix  ,iy-1,iz+1).HeatHeI+
					   GAMMA_ION(ix  ,iy  ,iz-1).HeatHeI+
 			    CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).HeatHeI+
					   GAMMA_ION(ix  ,iy  ,iz+1).HeatHeI+
					   GAMMA_ION(ix  ,iy+1,iz-1).HeatHeI+
					   GAMMA_ION(ix  ,iy+1,iz  ).HeatHeI+
					   GAMMA_ION(ix  ,iy+1,iz+1).HeatHeI+
					   GAMMA_ION(ix+1,iy-1,iz-1).HeatHeI+
					   GAMMA_ION(ix+1,iy-1,iz  ).HeatHeI+
					   GAMMA_ION(ix+1,iy-1,iz+1).HeatHeI+
					   GAMMA_ION(ix+1,iy  ,iz-1).HeatHeI+
					   GAMMA_ION(ix+1,iy  ,iz  ).HeatHeI+
					   GAMMA_ION(ix+1,iy  ,iz+1).HeatHeI+
					   GAMMA_ION(ix+1,iy+1,iz-1).HeatHeI+
					   GAMMA_ION(ix+1,iy+1,iz  ).HeatHeI+
					   GAMMA_ION(ix+1,iy+1,iz+1).HeatHeI);

	SMOOTH(ix,iy,iz).HeatHeII = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).HeatHeII+
					    GAMMA_ION(ix-1,iy-1,iz  ).HeatHeII+
					    GAMMA_ION(ix-1,iy-1,iz+1).HeatHeII+
					    GAMMA_ION(ix-1,iy  ,iz-1).HeatHeII+
					    GAMMA_ION(ix-1,iy  ,iz  ).HeatHeII+
					    GAMMA_ION(ix-1,iy  ,iz+1).HeatHeII+
					    GAMMA_ION(ix-1,iy+1,iz-1).HeatHeII+
					    GAMMA_ION(ix-1,iy+1,iz  ).HeatHeII+
					    GAMMA_ION(ix-1,iy+1,iz+1).HeatHeII+
					    GAMMA_ION(ix  ,iy-1,iz-1).HeatHeII+
					    GAMMA_ION(ix  ,iy-1,iz  ).HeatHeII+
					    GAMMA_ION(ix  ,iy-1,iz+1).HeatHeII+
					    GAMMA_ION(ix  ,iy  ,iz-1).HeatHeII+
			     CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).HeatHeII+
					    GAMMA_ION(ix  ,iy  ,iz+1).HeatHeII+
					    GAMMA_ION(ix  ,iy+1,iz-1).HeatHeII+
					    GAMMA_ION(ix  ,iy+1,iz  ).HeatHeII+
					    GAMMA_ION(ix  ,iy+1,iz+1).HeatHeII+
					    GAMMA_ION(ix+1,iy-1,iz-1).HeatHeII+
					    GAMMA_ION(ix+1,iy-1,iz  ).HeatHeII+
					    GAMMA_ION(ix+1,iy-1,iz+1).HeatHeII+
					    GAMMA_ION(ix+1,iy  ,iz-1).HeatHeII+
					    GAMMA_ION(ix+1,iy  ,iz  ).HeatHeII+
					    GAMMA_ION(ix+1,iy  ,iz+1).HeatHeII+
					    GAMMA_ION(ix+1,iy+1,iz-1).HeatHeII+
					    GAMMA_ION(ix+1,iy+1,iz  ).HeatHeII+
					    GAMMA_ION(ix+1,iy+1,iz+1).HeatHeII);
#endif /* __HELIUM__ */

#ifdef __HYDROGEN_MOL__
	SMOOTH(ix,iy,iz).GammaHM = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).GammaHM+
					   GAMMA_ION(ix-1,iy-1,iz  ).GammaHM+
					   GAMMA_ION(ix-1,iy-1,iz+1).GammaHM+
					   GAMMA_ION(ix-1,iy  ,iz-1).GammaHM+
					   GAMMA_ION(ix-1,iy  ,iz  ).GammaHM+
					   GAMMA_ION(ix-1,iy  ,iz+1).GammaHM+
					   GAMMA_ION(ix-1,iy+1,iz-1).GammaHM+
					   GAMMA_ION(ix-1,iy+1,iz  ).GammaHM+
					   GAMMA_ION(ix-1,iy+1,iz+1).GammaHM+
					   GAMMA_ION(ix  ,iy-1,iz-1).GammaHM+
					   GAMMA_ION(ix  ,iy-1,iz  ).GammaHM+
					   GAMMA_ION(ix  ,iy-1,iz+1).GammaHM+
					   GAMMA_ION(ix  ,iy  ,iz-1).GammaHM+
			    CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).GammaHM+
					   GAMMA_ION(ix  ,iy  ,iz+1).GammaHM+
					   GAMMA_ION(ix  ,iy+1,iz-1).GammaHM+
					   GAMMA_ION(ix  ,iy+1,iz  ).GammaHM+
					   GAMMA_ION(ix  ,iy+1,iz+1).GammaHM+
					   GAMMA_ION(ix+1,iy-1,iz-1).GammaHM+
					   GAMMA_ION(ix+1,iy-1,iz  ).GammaHM+
					   GAMMA_ION(ix+1,iy-1,iz+1).GammaHM+
					   GAMMA_ION(ix+1,iy  ,iz-1).GammaHM+
					   GAMMA_ION(ix+1,iy  ,iz  ).GammaHM+
					   GAMMA_ION(ix+1,iy  ,iz+1).GammaHM+
					   GAMMA_ION(ix+1,iy+1,iz-1).GammaHM+
					   GAMMA_ION(ix+1,iy+1,iz  ).GammaHM+
					   GAMMA_ION(ix+1,iy+1,iz+1).GammaHM);

	SMOOTH(ix,iy,iz).GammaH2I_I = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).GammaH2I_I+
					      GAMMA_ION(ix-1,iy-1,iz  ).GammaH2I_I+
					      GAMMA_ION(ix-1,iy-1,iz+1).GammaH2I_I+
					      GAMMA_ION(ix-1,iy  ,iz-1).GammaH2I_I+
					      GAMMA_ION(ix-1,iy  ,iz  ).GammaH2I_I+
					      GAMMA_ION(ix-1,iy  ,iz+1).GammaH2I_I+
					      GAMMA_ION(ix-1,iy+1,iz-1).GammaH2I_I+
					      GAMMA_ION(ix-1,iy+1,iz  ).GammaH2I_I+
					      GAMMA_ION(ix-1,iy+1,iz+1).GammaH2I_I+
					      GAMMA_ION(ix  ,iy-1,iz-1).GammaH2I_I+
					      GAMMA_ION(ix  ,iy-1,iz  ).GammaH2I_I+
					      GAMMA_ION(ix  ,iy-1,iz+1).GammaH2I_I+
					      GAMMA_ION(ix  ,iy  ,iz-1).GammaH2I_I+
			       CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).GammaH2I_I+
					      GAMMA_ION(ix  ,iy  ,iz+1).GammaH2I_I+
					      GAMMA_ION(ix  ,iy+1,iz-1).GammaH2I_I+
					      GAMMA_ION(ix  ,iy+1,iz  ).GammaH2I_I+
					      GAMMA_ION(ix  ,iy+1,iz+1).GammaH2I_I+
					      GAMMA_ION(ix+1,iy-1,iz-1).GammaH2I_I+
					      GAMMA_ION(ix+1,iy-1,iz  ).GammaH2I_I+
					      GAMMA_ION(ix+1,iy-1,iz+1).GammaH2I_I+
					      GAMMA_ION(ix+1,iy  ,iz-1).GammaH2I_I+
					      GAMMA_ION(ix+1,iy  ,iz  ).GammaH2I_I+
					      GAMMA_ION(ix+1,iy  ,iz+1).GammaH2I_I+
					      GAMMA_ION(ix+1,iy+1,iz-1).GammaH2I_I+
					      GAMMA_ION(ix+1,iy+1,iz  ).GammaH2I_I+
					      GAMMA_ION(ix+1,iy+1,iz+1).GammaH2I_I);

	SMOOTH(ix,iy,iz).GammaH2I_II = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).GammaH2I_II+
					       GAMMA_ION(ix-1,iy-1,iz  ).GammaH2I_II+
					       GAMMA_ION(ix-1,iy-1,iz+1).GammaH2I_II+
					       GAMMA_ION(ix-1,iy  ,iz-1).GammaH2I_II+
					       GAMMA_ION(ix-1,iy  ,iz  ).GammaH2I_II+
					       GAMMA_ION(ix-1,iy  ,iz+1).GammaH2I_II+
					       GAMMA_ION(ix-1,iy+1,iz-1).GammaH2I_II+
					       GAMMA_ION(ix-1,iy+1,iz  ).GammaH2I_II+
					       GAMMA_ION(ix-1,iy+1,iz+1).GammaH2I_II+
					       GAMMA_ION(ix  ,iy-1,iz-1).GammaH2I_II+
					       GAMMA_ION(ix  ,iy-1,iz  ).GammaH2I_II+
					       GAMMA_ION(ix  ,iy-1,iz+1).GammaH2I_II+
					       GAMMA_ION(ix  ,iy  ,iz-1).GammaH2I_II+
			        CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).GammaH2I_II+
					       GAMMA_ION(ix  ,iy  ,iz+1).GammaH2I_II+
					       GAMMA_ION(ix  ,iy+1,iz-1).GammaH2I_II+
					       GAMMA_ION(ix  ,iy+1,iz  ).GammaH2I_II+
					       GAMMA_ION(ix  ,iy+1,iz+1).GammaH2I_II+
					       GAMMA_ION(ix+1,iy-1,iz-1).GammaH2I_II+
					       GAMMA_ION(ix+1,iy-1,iz  ).GammaH2I_II+
					       GAMMA_ION(ix+1,iy-1,iz+1).GammaH2I_II+
					       GAMMA_ION(ix+1,iy  ,iz-1).GammaH2I_II+
					       GAMMA_ION(ix+1,iy  ,iz  ).GammaH2I_II+
					       GAMMA_ION(ix+1,iy  ,iz+1).GammaH2I_II+
					       GAMMA_ION(ix+1,iy+1,iz-1).GammaH2I_II+
					       GAMMA_ION(ix+1,iy+1,iz  ).GammaH2I_II+
					       GAMMA_ION(ix+1,iy+1,iz+1).GammaH2I_II);

	SMOOTH(ix,iy,iz).GammaH2II_I = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).GammaH2II_I+
					       GAMMA_ION(ix-1,iy-1,iz  ).GammaH2II_I+
					       GAMMA_ION(ix-1,iy-1,iz+1).GammaH2II_I+
					       GAMMA_ION(ix-1,iy  ,iz-1).GammaH2II_I+
					       GAMMA_ION(ix-1,iy  ,iz  ).GammaH2II_I+
					       GAMMA_ION(ix-1,iy  ,iz+1).GammaH2II_I+
					       GAMMA_ION(ix-1,iy+1,iz-1).GammaH2II_I+
					       GAMMA_ION(ix-1,iy+1,iz  ).GammaH2II_I+
					       GAMMA_ION(ix-1,iy+1,iz+1).GammaH2II_I+
					       GAMMA_ION(ix  ,iy-1,iz-1).GammaH2II_I+
					       GAMMA_ION(ix  ,iy-1,iz  ).GammaH2II_I+
					       GAMMA_ION(ix  ,iy-1,iz+1).GammaH2II_I+
					       GAMMA_ION(ix  ,iy  ,iz-1).GammaH2II_I+
			        CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).GammaH2II_I+
					       GAMMA_ION(ix  ,iy  ,iz+1).GammaH2II_I+
					       GAMMA_ION(ix  ,iy+1,iz-1).GammaH2II_I+
					       GAMMA_ION(ix  ,iy+1,iz  ).GammaH2II_I+
					       GAMMA_ION(ix  ,iy+1,iz+1).GammaH2II_I+
					       GAMMA_ION(ix+1,iy-1,iz-1).GammaH2II_I+
					       GAMMA_ION(ix+1,iy-1,iz  ).GammaH2II_I+
					       GAMMA_ION(ix+1,iy-1,iz+1).GammaH2II_I+
					       GAMMA_ION(ix+1,iy  ,iz-1).GammaH2II_I+
					       GAMMA_ION(ix+1,iy  ,iz  ).GammaH2II_I+
					       GAMMA_ION(ix+1,iy  ,iz+1).GammaH2II_I+
					       GAMMA_ION(ix+1,iy+1,iz-1).GammaH2II_I+
					       GAMMA_ION(ix+1,iy+1,iz  ).GammaH2II_I+
					       GAMMA_ION(ix+1,iy+1,iz+1).GammaH2II_I);

	SMOOTH(ix,iy,iz).GammaH2II_II = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).GammaH2II_II+
						GAMMA_ION(ix-1,iy-1,iz  ).GammaH2II_II+
						GAMMA_ION(ix-1,iy-1,iz+1).GammaH2II_II+
						GAMMA_ION(ix-1,iy  ,iz-1).GammaH2II_II+
						GAMMA_ION(ix-1,iy  ,iz  ).GammaH2II_II+
						GAMMA_ION(ix-1,iy  ,iz+1).GammaH2II_II+
						GAMMA_ION(ix-1,iy+1,iz-1).GammaH2II_II+
						GAMMA_ION(ix-1,iy+1,iz  ).GammaH2II_II+
						GAMMA_ION(ix-1,iy+1,iz+1).GammaH2II_II+
						GAMMA_ION(ix  ,iy-1,iz-1).GammaH2II_II+
						GAMMA_ION(ix  ,iy-1,iz  ).GammaH2II_II+
						GAMMA_ION(ix  ,iy-1,iz+1).GammaH2II_II+
						GAMMA_ION(ix  ,iy  ,iz-1).GammaH2II_II+
			 	 CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).GammaH2II_II+
						GAMMA_ION(ix  ,iy  ,iz+1).GammaH2II_II+
						GAMMA_ION(ix  ,iy+1,iz-1).GammaH2II_II+
						GAMMA_ION(ix  ,iy+1,iz  ).GammaH2II_II+
						GAMMA_ION(ix  ,iy+1,iz+1).GammaH2II_II+
						GAMMA_ION(ix+1,iy-1,iz-1).GammaH2II_II+
						GAMMA_ION(ix+1,iy-1,iz  ).GammaH2II_II+
						GAMMA_ION(ix+1,iy-1,iz+1).GammaH2II_II+
						GAMMA_ION(ix+1,iy  ,iz-1).GammaH2II_II+
						GAMMA_ION(ix+1,iy  ,iz  ).GammaH2II_II+
						GAMMA_ION(ix+1,iy  ,iz+1).GammaH2II_II+
						GAMMA_ION(ix+1,iy+1,iz-1).GammaH2II_II+
						GAMMA_ION(ix+1,iy+1,iz  ).GammaH2II_II+
						GAMMA_ION(ix+1,iy+1,iz+1).GammaH2II_II);

	SMOOTH(ix,iy,iz).HeatHM = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).HeatHM+
					  GAMMA_ION(ix-1,iy-1,iz  ).HeatHM+
					  GAMMA_ION(ix-1,iy-1,iz+1).HeatHM+
					  GAMMA_ION(ix-1,iy  ,iz-1).HeatHM+
					  GAMMA_ION(ix-1,iy  ,iz  ).HeatHM+
					  GAMMA_ION(ix-1,iy  ,iz+1).HeatHM+
					  GAMMA_ION(ix-1,iy+1,iz-1).HeatHM+
					  GAMMA_ION(ix-1,iy+1,iz  ).HeatHM+
					  GAMMA_ION(ix-1,iy+1,iz+1).HeatHM+
					  GAMMA_ION(ix  ,iy-1,iz-1).HeatHM+
					  GAMMA_ION(ix  ,iy-1,iz  ).HeatHM+
					  GAMMA_ION(ix  ,iy-1,iz+1).HeatHM+
					  GAMMA_ION(ix  ,iy  ,iz-1).HeatHM+
			   CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).HeatHM+
					  GAMMA_ION(ix  ,iy  ,iz+1).HeatHM+
					  GAMMA_ION(ix  ,iy+1,iz-1).HeatHM+
					  GAMMA_ION(ix  ,iy+1,iz  ).HeatHM+
					  GAMMA_ION(ix  ,iy+1,iz+1).HeatHM+
					  GAMMA_ION(ix+1,iy-1,iz-1).HeatHM+
					  GAMMA_ION(ix+1,iy-1,iz  ).HeatHM+
					  GAMMA_ION(ix+1,iy-1,iz+1).HeatHM+
					  GAMMA_ION(ix+1,iy  ,iz-1).HeatHM+
					  GAMMA_ION(ix+1,iy  ,iz  ).HeatHM+
					  GAMMA_ION(ix+1,iy  ,iz+1).HeatHM+
					  GAMMA_ION(ix+1,iy+1,iz-1).HeatHM+
					  GAMMA_ION(ix+1,iy+1,iz  ).HeatHM+
					  GAMMA_ION(ix+1,iy+1,iz+1).HeatHM);

	SMOOTH(ix,iy,iz).HeatH2I_I = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).HeatH2I_I+
					     GAMMA_ION(ix-1,iy-1,iz  ).HeatH2I_I+
					     GAMMA_ION(ix-1,iy-1,iz+1).HeatH2I_I+
					     GAMMA_ION(ix-1,iy  ,iz-1).HeatH2I_I+
					     GAMMA_ION(ix-1,iy  ,iz  ).HeatH2I_I+
					     GAMMA_ION(ix-1,iy  ,iz+1).HeatH2I_I+
					     GAMMA_ION(ix-1,iy+1,iz-1).HeatH2I_I+
					     GAMMA_ION(ix-1,iy+1,iz  ).HeatH2I_I+
					     GAMMA_ION(ix-1,iy+1,iz+1).HeatH2I_I+
					     GAMMA_ION(ix  ,iy-1,iz-1).HeatH2I_I+
					     GAMMA_ION(ix  ,iy-1,iz  ).HeatH2I_I+
					     GAMMA_ION(ix  ,iy-1,iz+1).HeatH2I_I+
					     GAMMA_ION(ix  ,iy  ,iz-1).HeatH2I_I+
			      CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).HeatH2I_I+
					     GAMMA_ION(ix  ,iy  ,iz+1).HeatH2I_I+
					     GAMMA_ION(ix  ,iy+1,iz-1).HeatH2I_I+
					     GAMMA_ION(ix  ,iy+1,iz  ).HeatH2I_I+
					     GAMMA_ION(ix  ,iy+1,iz+1).HeatH2I_I+
					     GAMMA_ION(ix+1,iy-1,iz-1).HeatH2I_I+
					     GAMMA_ION(ix+1,iy-1,iz  ).HeatH2I_I+
					     GAMMA_ION(ix+1,iy-1,iz+1).HeatH2I_I+
					     GAMMA_ION(ix+1,iy  ,iz-1).HeatH2I_I+
					     GAMMA_ION(ix+1,iy  ,iz  ).HeatH2I_I+
					     GAMMA_ION(ix+1,iy  ,iz+1).HeatH2I_I+
					     GAMMA_ION(ix+1,iy+1,iz-1).HeatH2I_I+
					     GAMMA_ION(ix+1,iy+1,iz  ).HeatH2I_I+
					     GAMMA_ION(ix+1,iy+1,iz+1).HeatH2I_I);

	SMOOTH(ix,iy,iz).HeatH2I_II = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).HeatH2I_II+
					      GAMMA_ION(ix-1,iy-1,iz  ).HeatH2I_II+
					      GAMMA_ION(ix-1,iy-1,iz+1).HeatH2I_II+
					      GAMMA_ION(ix-1,iy  ,iz-1).HeatH2I_II+
					      GAMMA_ION(ix-1,iy  ,iz  ).HeatH2I_II+
					      GAMMA_ION(ix-1,iy  ,iz+1).HeatH2I_II+
					      GAMMA_ION(ix-1,iy+1,iz-1).HeatH2I_II+
					      GAMMA_ION(ix-1,iy+1,iz  ).HeatH2I_II+
					      GAMMA_ION(ix-1,iy+1,iz+1).HeatH2I_II+
					      GAMMA_ION(ix  ,iy-1,iz-1).HeatH2I_II+
					      GAMMA_ION(ix  ,iy-1,iz  ).HeatH2I_II+
					      GAMMA_ION(ix  ,iy-1,iz+1).HeatH2I_II+
					      GAMMA_ION(ix  ,iy  ,iz-1).HeatH2I_II+
			       CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).HeatH2I_II+
					      GAMMA_ION(ix  ,iy  ,iz+1).HeatH2I_II+
					      GAMMA_ION(ix  ,iy+1,iz-1).HeatH2I_II+
					      GAMMA_ION(ix  ,iy+1,iz  ).HeatH2I_II+
					      GAMMA_ION(ix  ,iy+1,iz+1).HeatH2I_II+
					      GAMMA_ION(ix+1,iy-1,iz-1).HeatH2I_II+
					      GAMMA_ION(ix+1,iy-1,iz  ).HeatH2I_II+
					      GAMMA_ION(ix+1,iy-1,iz+1).HeatH2I_II+
					      GAMMA_ION(ix+1,iy  ,iz-1).HeatH2I_II+
					      GAMMA_ION(ix+1,iy  ,iz  ).HeatH2I_II+
					      GAMMA_ION(ix+1,iy  ,iz+1).HeatH2I_II+
					      GAMMA_ION(ix+1,iy+1,iz-1).HeatH2I_II+
					      GAMMA_ION(ix+1,iy+1,iz  ).HeatH2I_II+
					      GAMMA_ION(ix+1,iy+1,iz+1).HeatH2I_II);

	SMOOTH(ix,iy,iz).HeatH2II_I = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).HeatH2II_I+
					      GAMMA_ION(ix-1,iy-1,iz  ).HeatH2II_I+
					      GAMMA_ION(ix-1,iy-1,iz+1).HeatH2II_I+
					      GAMMA_ION(ix-1,iy  ,iz-1).HeatH2II_I+
					      GAMMA_ION(ix-1,iy  ,iz  ).HeatH2II_I+
					      GAMMA_ION(ix-1,iy  ,iz+1).HeatH2II_I+
					      GAMMA_ION(ix-1,iy+1,iz-1).HeatH2II_I+
					      GAMMA_ION(ix-1,iy+1,iz  ).HeatH2II_I+
					      GAMMA_ION(ix-1,iy+1,iz+1).HeatH2II_I+
					      GAMMA_ION(ix  ,iy-1,iz-1).HeatH2II_I+
					      GAMMA_ION(ix  ,iy-1,iz  ).HeatH2II_I+
					      GAMMA_ION(ix  ,iy-1,iz+1).HeatH2II_I+
					      GAMMA_ION(ix  ,iy  ,iz-1).HeatH2II_I+
			       CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).HeatH2II_I+
					      GAMMA_ION(ix  ,iy  ,iz+1).HeatH2II_I+
					      GAMMA_ION(ix  ,iy+1,iz-1).HeatH2II_I+
					      GAMMA_ION(ix  ,iy+1,iz  ).HeatH2II_I+
					      GAMMA_ION(ix  ,iy+1,iz+1).HeatH2II_I+
					      GAMMA_ION(ix+1,iy-1,iz-1).HeatH2II_I+
					      GAMMA_ION(ix+1,iy-1,iz  ).HeatH2II_I+
					      GAMMA_ION(ix+1,iy-1,iz+1).HeatH2II_I+
					      GAMMA_ION(ix+1,iy  ,iz-1).HeatH2II_I+
					      GAMMA_ION(ix+1,iy  ,iz  ).HeatH2II_I+
					      GAMMA_ION(ix+1,iy  ,iz+1).HeatH2II_I+
					      GAMMA_ION(ix+1,iy+1,iz-1).HeatH2II_I+
					      GAMMA_ION(ix+1,iy+1,iz  ).HeatH2II_I+
					      GAMMA_ION(ix+1,iy+1,iz+1).HeatH2II_I);

	SMOOTH(ix,iy,iz).HeatH2II_II = NORMAL*(GAMMA_ION(ix-1,iy-1,iz-1).HeatH2II_II+
					       GAMMA_ION(ix-1,iy-1,iz  ).HeatH2II_II+
					       GAMMA_ION(ix-1,iy-1,iz+1).HeatH2II_II+
					       GAMMA_ION(ix-1,iy  ,iz-1).HeatH2II_II+
					       GAMMA_ION(ix-1,iy  ,iz  ).HeatH2II_II+
					       GAMMA_ION(ix-1,iy  ,iz+1).HeatH2II_II+
					       GAMMA_ION(ix-1,iy+1,iz-1).HeatH2II_II+
					       GAMMA_ION(ix-1,iy+1,iz  ).HeatH2II_II+
					       GAMMA_ION(ix-1,iy+1,iz+1).HeatH2II_II+
					       GAMMA_ION(ix  ,iy-1,iz-1).HeatH2II_II+
					       GAMMA_ION(ix  ,iy-1,iz  ).HeatH2II_II+
					       GAMMA_ION(ix  ,iy-1,iz+1).HeatH2II_II+
					       GAMMA_ION(ix  ,iy  ,iz-1).HeatH2II_II+
			        CENTRAL_WEIGHT*GAMMA_ION(ix  ,iy  ,iz  ).HeatH2II_II+
					       GAMMA_ION(ix  ,iy  ,iz+1).HeatH2II_II+
					       GAMMA_ION(ix  ,iy+1,iz-1).HeatH2II_II+
					       GAMMA_ION(ix  ,iy+1,iz  ).HeatH2II_II+
					       GAMMA_ION(ix  ,iy+1,iz+1).HeatH2II_II+
					       GAMMA_ION(ix+1,iy-1,iz-1).HeatH2II_II+
					       GAMMA_ION(ix+1,iy-1,iz  ).HeatH2II_II+
					       GAMMA_ION(ix+1,iy-1,iz+1).HeatH2II_II+
					       GAMMA_ION(ix+1,iy  ,iz-1).HeatH2II_II+
					       GAMMA_ION(ix+1,iy  ,iz  ).HeatH2II_II+
					       GAMMA_ION(ix+1,iy  ,iz+1).HeatH2II_II+
					       GAMMA_ION(ix+1,iy+1,iz-1).HeatH2II_II+
					       GAMMA_ION(ix+1,iy+1,iz  ).HeatH2II_II+
					       GAMMA_ION(ix+1,iy+1,iz+1).HeatH2II_II);
#endif /* __HYDROGEN_MOL__ */

#else /* !__SMOOTHING__ */
        SMOOTH(ix,iy,iz).GammaHI = GAMMA_ION(ix,iy,iz).GammaHI;
        SMOOTH(ix,iy,iz).HeatHI  = GAMMA_ION(ix,iy,iz).HeatHI;
#ifdef __HELIUM__
        SMOOTH(ix,iy,iz).GammaHeI  = GAMMA_ION(ix,iy,iz).GammaHeI;
        SMOOTH(ix,iy,iz).GammaHeII = GAMMA_ION(ix,iy,iz).GammaHeII;
        SMOOTH(ix,iy,iz).HeatHeI   = GAMMA_ION(ix,iy,iz).HeatHeI;
        SMOOTH(ix,iy,iz).HeatHeII  = GAMMA_ION(ix,iy,iz).HeatHeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
        SMOOTH(ix,iy,iz).GammaHM      = GAMMA_ION(ix,iy,iz).GammaHM;
        SMOOTH(ix,iy,iz).GammaH2I_I   = GAMMA_ION(ix,iy,iz).GammaH2I_I;
        SMOOTH(ix,iy,iz).GammaH2I_II  = GAMMA_ION(ix,iy,iz).GammaH2I_II;
        SMOOTH(ix,iy,iz).GammaH2II_I  = GAMMA_ION(ix,iy,iz).GammaH2II_I;
        SMOOTH(ix,iy,iz).GammaH2II_II = GAMMA_ION(ix,iy,iz).GammaH2II_II;
        SMOOTH(ix,iy,iz).HeatHM      = GAMMA_ION(ix,iy,iz).HeatHM;
        SMOOTH(ix,iy,iz).HeatH2I_I   = GAMMA_ION(ix,iy,iz).HeatH2I_I;
        SMOOTH(ix,iy,iz).HeatH2I_II  = GAMMA_ION(ix,iy,iz).HeatH2I_II;
        SMOOTH(ix,iy,iz).HeatH2II_I  = GAMMA_ION(ix,iy,iz).HeatH2II_I;
        SMOOTH(ix,iy,iz).HeatH2II_II = GAMMA_ION(ix,iy,iz).HeatH2II_II;
#endif /* __HYDROGEN_MOL__ */
#endif /* __SMOOTHING__ */
      }
    }
  }

#ifdef __USE_GPU__
  send_photoion_data(smooth, cuda_mem, this_cuda, this_run);
#endif /* __USE_GPU__ */
#pragma omp parallel for
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	MESH(ix,iy,iz).prev_chem.GammaHI = SMOOTH(ix,iy,iz).GammaHI;
	MESH(ix,iy,iz).prev_chem.HeatHI  = SMOOTH(ix,iy,iz).HeatHI;
#ifdef __HELIUM__
	MESH(ix,iy,iz).prev_chem.GammaHeI  = SMOOTH(ix,iy,iz).GammaHeI;
	MESH(ix,iy,iz).prev_chem.GammaHeII = SMOOTH(ix,iy,iz).GammaHeII;
	MESH(ix,iy,iz).prev_chem.HeatHeI   = SMOOTH(ix,iy,iz).HeatHeI;
	MESH(ix,iy,iz).prev_chem.HeatHeII  = SMOOTH(ix,iy,iz).HeatHeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
	MESH(ix,iy,iz).prev_chem.GammaHM      = SMOOTH(ix,iy,iz).GammaHM;
	MESH(ix,iy,iz).prev_chem.GammaH2I_I   = SMOOTH(ix,iy,iz).GammaH2I_I;
	MESH(ix,iy,iz).prev_chem.GammaH2I_II  = SMOOTH(ix,iy,iz).GammaH2I_II;
	MESH(ix,iy,iz).prev_chem.GammaH2II_I  = SMOOTH(ix,iy,iz).GammaH2II_I;
	MESH(ix,iy,iz).prev_chem.GammaH2II_II = SMOOTH(ix,iy,iz).GammaH2II_II;
	MESH(ix,iy,iz).prev_chem.HeatHM       = SMOOTH(ix,iy,iz).HeatHM;
	MESH(ix,iy,iz).prev_chem.HeatH2I_I    = SMOOTH(ix,iy,iz).HeatH2I_I;
	MESH(ix,iy,iz).prev_chem.HeatH2I_II   = SMOOTH(ix,iy,iz).HeatH2I_II;
	MESH(ix,iy,iz).prev_chem.HeatH2II_I   = SMOOTH(ix,iy,iz).HeatH2II_I;
	MESH(ix,iy,iz).prev_chem.HeatH2II_II  = SMOOTH(ix,iy,iz).HeatH2II_II;
#endif
      }
    }
  }

#ifdef __ARGOT_PROFILE__
  times(&end_tms);
  gettimeofday(&end_tv, NULL);

  fprintf(this_run->proc_file,
	  "# smooth_photoion_rate : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	  timing(start_tms, end_tms), wallclock_timing(start_tv, end_tv));
  fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */


}
