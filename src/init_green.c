#include <math.h>
#include <assert.h>

#include <fftw3-mpi.h>

#include "run_param.h"
#include "constants.h"
#include "fftw_mpi_param.h"

#ifdef __ISOLATED_GRAV__
#define GK(ix,iy,iz) (gk[(iz)+NMESH_Z_GREEN_P2*((iy)+NMESH_Y_GREEN*(ix))])

void init_green_isolated(float *gk, 
			 struct run_param *this_run,
			 struct fftw_mpi_param *this_fftw)
{

  assert(this_fftw->ix_length_green*NNODE_X == NMESH_X_GREEN);
  assert(this_fftw->ix_start_green == this_run->rank_x*this_fftw->ix_length_green);

  for(int ix=0;ix<this_fftw->ix_length_green;ix++) {
    int jx = ix+this_fftw->ix_start_green;
    float xpos;

    if(jx<NMESH_X_GREEN/2) {
      xpos = this_run->delta_x*(float)jx;
    }else{
      xpos = this_run->delta_x*(float)(NMESH_X_GREEN-jx);
    }
    for(int jy=0;jy<=NMESH_Y_GREEN;jy++) {
      float ypos;

      if(jy<NMESH_Y_GREEN/2) {
	ypos = this_run->delta_y*(float)jy;
      }else{
	ypos = this_run->delta_y*(float)(NMESH_Y_GREEN-jy);
      }
      for(int jz=0;jz<=NMESH_Z_GREEN;jz++) {
	float zpos;
	if(jz<NMESH_Z_GREEN/2) {
	  zpos = this_run->delta_z*(float)jz;
	}else{
	  zpos = this_run->delta_z*(float)(NMESH_Z_GREEN-jz);
	}

	if(jx == 0 && jy == 0 && jz == 0) {
	  GK(ix,jy,jz) = 0.0;
	}else{
	  GK(ix,jy,jz) = -1.0/sqrtf(SQR(xpos)+SQR(ypos)+SQR(zpos));
	}

      }
    }
  }

  fftwf_plan init_green_plan;
  init_green_plan = 
    fftwf_mpi_plan_dft_r2c_3d(NMESH_X_GREEN, NMESH_Y_GREEN, NMESH_Z_GREEN,
			      gk, (fftwf_complex*)gk, this_fftw->grav_fftw_comm,
			      FFTW_ESTIMATE);

  fftwf_execute(init_green_plan);

  fftwf_destroy_plan(init_green_plan);

}

#undef GK

#else /* !__ISOLATED_GRAV__ */

float periodic_green_function(int ikx, int iky, int ikz) 
{
  float sins_x, sins_y, sins_z;
  float pins;

  /* number of mesh along each axes should be the same */
  assert(NMESH_X_POTEN == NMESH_Y_POTEN);
  assert(NMESH_X_POTEN == NMESH_Z_POTEN);

  assert(0<=ikx); assert(0<=iky); assert(0<=ikz);
  assert(ikx<NMESH_X_GREEN);assert(iky<NMESH_Y_GREEN);assert(ikz<NMESH_Z_GREEN);

  pins = -PI/(float)(NMESH_X_POTEN*NMESH_X_POTEN);

  if(ikx==0 && iky==0 && ikz==0) {
    return 0.0;
  }else{
    sins_x = sin(PI*(float)ikx/(float)NMESH_X_POTEN);
    sins_y = sin(PI*(float)iky/(float)NMESH_Y_POTEN);
    sins_z = sin(PI*(float)ikz/(float)NMESH_Z_POTEN);

    sins_x = SQR(sins_x);
    sins_y = SQR(sins_y);
    sins_z = SQR(sins_z);
    
    return pins/(sins_x+sins_y+sins_z);
  }
}



#define GK(ix,iy,iz) (gk[(iz)+NMESH_Z_GREEN*((iy)+NMESH_Y_GREEN*(ix))])

void init_green_periodic(float *gk, 
			 struct run_param *this_run, 
			 struct fftw_mpi_param *this_fftw)
{
  if(this_fftw->ix_start < NMESH_X_TOTAL/2) {
    for(int ikx=0;ikx<this_fftw->ix_length;ikx++) {
      for(int iky=0;iky<NMESH_Y_GREEN;iky++) {
	for(int ikz=0;ikz<NMESH_Z_GREEN;ikz++) {
	  GK(ikx,iky,ikz)=periodic_green_function(ikx+this_fftw->ix_start,
						  iky,ikz);
	}
      }
    }
  }else{
    for(int ikx=0;ikx<this_fftw->ix_length;ikx++) {
      for(int iky=0;iky<NMESH_Y_GREEN;iky++) {
	for(int ikz=0;ikz<NMESH_Z_GREEN;ikz++) {
	  GK(ikx,iky,ikz)=periodic_green_function(NMESH_X_TOTAL-this_fftw->ix_start-ikx,
						  iky,ikz);
	}
      }
    }    
  }
}

#undef GK
#endif

void init_green(float *gk, 
		struct run_param *this_run, struct fftw_mpi_param *this_fftw)
{
#ifdef __ISOLATED_GRAV__
  init_green_isolated(gk, this_run, this_fftw);
#else
  init_green_periodic(gk, this_run, this_fftw);
#endif
}
