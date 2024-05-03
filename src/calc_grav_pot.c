#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <fftw3-mpi.h>

#include "run_param.h"
#include "fluid.h"
#include "fftw_mpi_param.h"
#include "prototype.h"

#define RHO(ix,iy,iz) dens[(iz)+NMESH_Z_POTEN_P2*((iy)+NMESH_Y_POTEN*(ix))]
#define RHOK(ix,iy,iz) dens_hat[(iz)+(NMESH_Z_POTEN/2+1)*((iy)+NMESH_Y_POTEN*(ix))]
#define MESH(ix,iy,iz) mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))]

#define cmplx_re(c) ((c)[0])
#define cmplx_im(c) ((c)[1])

void zero_out_mesh_density(float *dens, struct fftw_mpi_param *this_fftw) 
{
#ifdef __ISOLATED_GRAV__
  assert(this_fftw->ix_length == 2*NMESH_X_LOCAL);

#pragma omp parallel for schedule(auto)
  for(int ix=0;ix<this_fftw->ix_length;ix++) {
    for(int iy=0;iy<NMESH_Y_POTEN;iy++) {
      for(int iz=0;iz<NMESH_Z_POTEN_P2;iz++) {
	RHO(ix,iy,iz)=0.0;
      }
    }
  }

#else
  assert(this_fftw->ix_length == NMESH_X_LOCAL);

#pragma omp parallel for schedule(auto)
  for(int ix=0;ix<this_fftw->ix_length;ix++) {
    for(int iy=0;iy<NMESH_Y_POTEN;iy++) {
      for(int iz=0;iz<NMESH_Z_POTEN_P2;iz++) {
	RHO(ix,iy,iz) = 0.0;
      }
    }
  }
#endif

}

void calc_mesh_density(float *dens, struct fluid_mesh *mesh,
		       struct run_param *this_run,
		       struct fftw_mpi_param *this_fftw)
{
#ifdef __ISOLATED_GRAV__

  assert(this_fftw->ix_length == 2*NMESH_X_LOCAL);
  float vol = this_run->delta_x*this_run->delta_y*this_run->delta_z;
  
  /* starting global index along X-coord. where the data should be placed. */
  int jx_start = this_run->rank_x*NMESH_X_LOCAL;
  /* the X MPI rank where the data should be sent. */
  int to_rank_x = jx_start / this_fftw->ix_length;
  /* starting local index */
  int ix_start_local = jx_start - to_rank_x*this_fftw->ix_length;

#pragma omp parallel for schedule(auto)
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    int jx = ix_start_local + ix;
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      int jy = this_run->rank_y*NMESH_Y_LOCAL + iy;
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	int jz = this_run->rank_z*NMESH_Z_LOCAL + iz;
	RHO(jx,jy,jz) = MESH(ix,iy,iz).dens*vol;
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, dens,
		this_fftw->ix_length*NMESH_Y_POTEN*NMESH_Z_POTEN_P2,
		MPI_FLOAT, MPI_SUM, this_fftw->dens_reduction_comm);


  int half_comm_size = (this_fftw->ix_length/2)*NMESH_Y_POTEN*NMESH_Z_POTEN_P2;
  int comm_size = this_fftw->ix_length*NMESH_Y_POTEN*NMESH_Z_POTEN_P2;
  int tag;
  MPI_Status stat;
  
  tag = 0;
  if((this_run->rank_x%2) == 0) {
    int rank_from = mpi_rank(this_run->rank_x+1, 
			     this_run->rank_y, 
			     this_run->rank_z);
    MPI_Recv(&(RHO(this_fftw->ix_length/2, 0,0)), half_comm_size, MPI_FLOAT, 
	     rank_from, tag, MPI_COMM_WORLD, &stat);
  }else{
    int rank_to = mpi_rank(this_run->rank_x-1,
			   this_run->rank_y,
			   this_run->rank_z);
    MPI_Send(&(RHO(this_fftw->ix_length/2, 0,0)), half_comm_size, MPI_FLOAT,
	     rank_to, tag, MPI_COMM_WORLD);
    zero_out_mesh_density(dens, this_fftw);
  }

  tag = 1;
  for(int irank_x=2;irank_x<NNODE_X;irank_x+=2) {
    if(this_run->rank_x == irank_x) {
      int rank_to = mpi_rank(this_run->rank_x/2, 
			     this_run->rank_y,
			     this_run->rank_z);
      MPI_Send(dens, comm_size, MPI_FLOAT, rank_to, tag, MPI_COMM_WORLD);
      zero_out_mesh_density(dens, this_fftw);
    }
    if(this_run->rank_x == irank_x/2) {
      int rank_from = mpi_rank(irank_x, 
			       this_run->rank_y,
			       this_run->rank_z);
      MPI_Recv(dens, comm_size, MPI_FLOAT, rank_from, tag, MPI_COMM_WORLD,
	       &stat);

    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

#else /* !__ISOLATED_GRAV__ */

  assert(this_fftw->ix_length == NMESH_X_LOCAL);
  float vol = this_run->delta_x*this_run->delta_y*this_run->delta_z;

#pragma omp parallel for schedule(auto)
  for(int ix=0;ix<this_fftw->ix_length;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      int jy = this_run->rank_y*NMESH_Y_LOCAL+iy;
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	int jz = this_run->rank_z*NMESH_Z_LOCAL+iz;

	RHO(ix,jy,jz) = MESH(ix,iy,iz).dens*vol;
      }
    }
  }
  
  MPI_Allreduce(MPI_IN_PLACE, dens, 
		NMESH_X_LOCAL*NMESH_Y_POTEN*NMESH_Z_POTEN_P2,
		MPI_FLOAT, MPI_SUM, this_fftw->dens_reduction_comm);
#endif

}

void calc_mesh_grav_pot(float *dens,
			struct fluid_mesh *mesh,
			struct run_param *this_run,
			struct fftw_mpi_param *this_fftw,
			float *gk)
{
  static int forward_plan_created = 0;
  static int backward_plan_created = 0;

#ifdef __ISOLATED_GRAV__
#define GK(ix,iy,iz) (gk_hat[(iz)+(NMESH_Z_GREEN/2+1)*((iy)+NMESH_Y_GREEN*(ix))])

  fftwf_complex *dens_hat, *gk_hat;

  if(forward_plan_created == 0) {
    this_fftw->forward_plan = 
      fftwf_mpi_plan_dft_r2c_3d(NMESH_X_POTEN, NMESH_Y_POTEN, NMESH_Z_POTEN,
				dens, (fftwf_complex *)dens, 
				this_fftw->grav_fftw_comm, FFTW_ESTIMATE);
    forward_plan_created = 1;
  }

  fftwf_execute(this_fftw->forward_plan);

  dens_hat = (fftwf_complex *)dens;
  gk_hat = (fftwf_complex *)gk;

#pragma omp parallel for schedule(auto)
  for(int ix=0;ix<this_fftw->ix_length;ix++) {
    for(int iy=0;iy<NMESH_Y_POTEN;iy++) {
      for(int iz=0;iz<NMESH_Z_POTEN/2+1;iz++) {
	cmplx_re(RHOK(ix,iy,iz)) *= cmplx_re(GK(ix,iy,iz));
	cmplx_im(RHOK(ix,iy,iz)) *= cmplx_re(GK(ix,iy,iz));
      }
    }
  }

  if(backward_plan_created == 0) {
    this_fftw->backward_plan = 
      fftwf_mpi_plan_dft_c2r_3d(NMESH_X_POTEN, NMESH_Y_POTEN, NMESH_Z_POTEN,
				dens_hat, dens, 
				this_fftw->grav_fftw_comm, FFTW_ESTIMATE);
    backward_plan_created = 1;
  }

  fftwf_execute(this_fftw->backward_plan);

#pragma omp parallel for schedule(auto)
  for(int ix=0;ix<this_fftw->ix_length;ix++) {
    for(int iy=0;iy<NMESH_Y_POTEN;iy++) {
      for(int iz=0;iz<NMESH_Z_POTEN;iz++) {
	RHO(ix,iy,iz) /= (float)(NMESH_X_POTEN*NMESH_Y_POTEN*NMESH_Z_POTEN);
      }
    }
  }

  /* sending the potential data to the appropriate ranks */
  int comm_size=this_fftw->ix_length*NMESH_Y_POTEN*NMESH_Z_POTEN_P2;
  int half_comm_size=comm_size/2;
  int tag;
  MPI_Status stat;

  tag=2;
  for(int irank_x=NNODE_X/2-1;irank_x>0;irank_x-=1) {
    if(this_run->rank_x == irank_x) {
      int rank_to = mpi_rank(this_run->rank_x*2,
			     this_run->rank_y,
			     this_run->rank_z);
      MPI_Send(dens, comm_size, MPI_FLOAT, rank_to, tag, MPI_COMM_WORLD);
    }
    if(this_run->rank_x == irank_x*2) {
      int rank_from = mpi_rank(irank_x,
			       this_run->rank_y,
			       this_run->rank_z);
      MPI_Recv(dens, comm_size, MPI_FLOAT, rank_from,  tag, MPI_COMM_WORLD, &stat);
    }
  }

  tag=3;
  if((this_run->rank_x%2) == 0) {
    int rank_to = mpi_rank(this_run->rank_x+1,
			   this_run->rank_y,
			   this_run->rank_z);
    MPI_Send(&(RHO(this_fftw->ix_length/2, 0,0)), half_comm_size, MPI_FLOAT,
	     rank_to, tag, MPI_COMM_WORLD);
  }else{
    int rank_from = mpi_rank(this_run->rank_x-1,
			     this_run->rank_y,
			     this_run->rank_z);
    MPI_Recv(dens, half_comm_size, MPI_FLOAT, 
	     rank_from, tag, MPI_COMM_WORLD, &stat);
  }

  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      int jy=iy+NMESH_Y_LOCAL*this_run->rank_y;
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	int jz=iz+NMESH_Z_LOCAL*this_run->rank_z;
	MESH(ix,iy,iz).pot = RHO(ix,jy,jz);
      }
    }
  }

#undef GK
#else /* !__ISOLATED_GRAV__ */
#define GK(ix,iy,iz) (gk[(iz)+NMESH_Z_GREEN*((iy)+NMESH_Y_GREEN*(ix))])

  fftwf_complex *dens_hat;

  if(forward_plan_created == 0) {
    this_fftw->forward_plan = 
      fftwf_mpi_plan_dft_r2c_3d(NMESH_X_POTEN, NMESH_Y_POTEN, NMESH_Z_POTEN,
				dens, (fftwf_complex *)dens, 
				this_fftw->grav_fftw_comm, FFTW_ESTIMATE);
    forward_plan_created = 1;
  }

  fftwf_execute(this_fftw->forward_plan);

  dens_hat = (fftwf_complex *)dens;

#pragma omp parallel for schedule(auto)
  for(int ix=0;ix<this_fftw->ix_length;ix++) {
    for(int iy=0;iy<NMESH_Y_POTEN/2;iy++) {
      int iyp = NMESH_Y_POTEN/2+iy;
      int iym = NMESH_Y_POTEN/2-iy;
      for(int iz=0;iz<NMESH_Z_POTEN/2+1;iz++) {
	cmplx_re(RHOK(ix,iy,iz))  *= GK(ix,iy,iz);
	cmplx_re(RHOK(ix,iyp,iz)) *= GK(ix,iym,iz);
		 
	cmplx_im(RHOK(ix,iy,iz))  *= GK(ix,iy,iz);
	cmplx_im(RHOK(ix,iyp,iz)) *= GK(ix,iym,iz);
      }
    }
  }


  if(backward_plan_created == 0) {
    this_fftw->backward_plan = 
      fftwf_mpi_plan_dft_c2r_3d(NMESH_X_POTEN, NMESH_Y_POTEN, NMESH_Z_POTEN,
				dens_hat, dens, 
				this_fftw->grav_fftw_comm, FFTW_ESTIMATE);
    backward_plan_created = 1;
  }

  fftwf_execute(this_fftw->backward_plan);

#pragma omp parallel for schedule(auto)
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      int jy = iy+NMESH_Y_LOCAL*this_run->rank_y;
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	int jz = iz+NMESH_Z_LOCAL*this_run->rank_z;
	MESH(ix,iy,iz).pot = RHO(ix,jy,jz);
      }
    }
  }

#undef GK
#endif

}
