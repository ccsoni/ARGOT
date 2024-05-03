#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>

#include "run_param.h"
#include "constants.h"
#include "fluid.h"
#include "radiation.h"
#include "prototype.h"

#ifndef TINY
#define TINY (1.0e-31)
#endif

#define MESH(ix,iy,iz) mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))]

void calc_optical_depth(struct ray_segment *segment, 
			struct fluid_mesh *mesh,
			struct run_param *this_run)
{
  double nsigma_HI;

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

  uint64_t iseg;

//#pragma omp parallel for schedule(dynamic,32)
#pragma omp parallel for schedule(auto)
  for(iseg=0;iseg<this_run->nseg;iseg++) {
    float nH, nHI;
#ifdef __HELIUM__
    float nHe, nHeI, nHeII;
#endif
#ifdef __HYDROGEN_MOL__
    float nHM, nH2I, nH2II;
#endif

    int x_step, y_step, z_step;
    int x_next, y_next, z_next;

    float dx, dy, dz;
    float dl, dxy;

    float sin_theta, cos_theta;
    float sin_phi,   cos_phi;

    float xovr, yovr, zovr;
    float rmin, rx, ry, rz;

    int   ix_end, iy_end, iz_end;
    int   ix_cur, iy_cur, iz_cur;

    /* current position on the ray segment in the local coordinate */
    float x_cur, y_cur, z_cur; 

    segment[iseg].optical_depth_HI = 0.0;
#ifdef __HELIUM__
    segment[iseg].optical_depth_HeI  = 0.0;
    segment[iseg].optical_depth_HeII = 0.0;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
    segment[iseg].optical_depth_HM = 0.0;
    segment[iseg].optical_depth_H2I = 0.0;
    segment[iseg].optical_depth_H2II = 0.0;
#endif

    ix_end = (int)((segment[iseg].xpos_end-this_run->xmin_local)/this_run->delta_x);
    iy_end = (int)((segment[iseg].ypos_end-this_run->ymin_local)/this_run->delta_y);
    iz_end = (int)((segment[iseg].zpos_end-this_run->zmin_local)/this_run->delta_z);

    ix_end = MAX(MIN(ix_end, NMESH_X_LOCAL-1),0);
    iy_end = MAX(MIN(iy_end, NMESH_Y_LOCAL-1),0);
    iz_end = MAX(MIN(iz_end, NMESH_Z_LOCAL-1),0);

    ix_cur = (int)((segment[iseg].xpos_start-this_run->xmin_local)/this_run->delta_x);
    iy_cur = (int)((segment[iseg].ypos_start-this_run->ymin_local)/this_run->delta_y);
    iz_cur = (int)((segment[iseg].zpos_start-this_run->zmin_local)/this_run->delta_z);

    ix_cur = MAX(MIN(ix_cur, NMESH_X_LOCAL-1),0);
    iy_cur = MAX(MIN(iy_cur, NMESH_Y_LOCAL-1),0);
    iz_cur = MAX(MIN(iz_cur, NMESH_Z_LOCAL-1),0);

    dx = segment[iseg].xpos_end-segment[iseg].xpos_start;
    dy = segment[iseg].ypos_end-segment[iseg].ypos_start;
    dz = segment[iseg].zpos_end-segment[iseg].zpos_start;

    dl = sqrt(SQR(dx)+SQR(dy)+SQR(dz)+TINY);
    dxy = sqrt(SQR(dx)+SQR(dy));

    cos_theta = dz/dl;
    sin_theta = dxy/dl;

    //    sin_theta = sqrt(1.0-SQR(cos_theta));
    //    dxy = dl*sin_theta;
    cos_phi = dx/(dxy+TINY);
    sin_phi = dy/(dxy+TINY);

    xovr = cos_phi*sin_theta;
    if(fabsf(xovr)<TINY) {
      xovr = (xovr >= 0.0 ? TINY : -TINY);
    }

    yovr = sin_phi*sin_theta;
    if(fabs(yovr)<TINY) {
      yovr = (yovr >= 0.0 ? TINY : -TINY);
    }

    zovr = cos_theta;
    if(fabs(zovr)<TINY) {
      zovr = (zovr >= 0.0 ? TINY : -TINY);
    }

    if(xovr > 0.e0) {
      x_step = 1; x_next = 1;
    }else{
      x_step = 0; x_next = -1;
    }

    if(yovr > 0.e0) {
      y_step = 1; y_next = 1;
    }else {
      y_step = 0; y_next = -1;
    }

    if(zovr > 0.e0) {
      z_step = 1; z_next = 1;
    }else{
      z_step = 0; z_next = -1;
    }

    /* local coodinate with respect to each parallel domain */
    x_cur = segment[iseg].xpos_start - this_run->xmin_local;
    y_cur = segment[iseg].ypos_start - this_run->ymin_local;
    z_cur = segment[iseg].zpos_start - this_run->zmin_local;

    while((ix_cur != ix_end || iy_cur != iy_end || iz_cur != iz_end) &&
	  (0<=ix_cur && ix_cur<NMESH_X_LOCAL) &&
	  (0<=iy_cur && iy_cur<NMESH_Y_LOCAL) &&
	  (0<=iz_cur && iz_cur<NMESH_Z_LOCAL) ) {
      
      nH  = MESH(ix_cur, iy_cur, iz_cur).dens;
      nHI = nH*MESH(ix_cur, iy_cur, iz_cur).chem.fHI;
#ifdef __HELIUM__
      nHe   = nH*HELIUM_FACT;
      nHeI  = nHe*MESH(ix_cur, iy_cur, iz_cur).chem.fHeI;
      nHeII = nHe*MESH(ix_cur, iy_cur, iz_cur).chem.fHeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
      nHM   = nH*MESH(ix_cur, iy_cur, iz_cur).chem.fHM;
      nH2I  = nH*MESH(ix_cur, iy_cur, iz_cur).chem.fH2I;
      nH2II = nH*MESH(ix_cur, iy_cur, iz_cur).chem.fH2II;
#endif /* __HYDROGEN_MOL__ */

      dx = this_run->delta_x*(x_step+ix_cur) - x_cur;
      dy = this_run->delta_y*(y_step+iy_cur) - y_cur;
      dz = this_run->delta_z*(z_step+iz_cur) - z_cur;
      
      rx = dx/xovr;
      ry = dy/yovr;
      rz = dz/zovr;

      rmin = fminf(rx, fminf(ry, rz));
      //rmin = MIN(rx, MIN(ry, rz));
      if(rmin == rx) ix_cur += x_next;
      if(rmin == ry) iy_cur += y_next;
      if(rmin == rz) iz_cur += z_next;

      x_cur += rmin*xovr;
      y_cur += rmin*yovr;
      z_cur += rmin*zovr;

      segment[iseg].optical_depth_HI += rmin*nHI;
#ifdef __HELIUM__
      segment[iseg].optical_depth_HeI += rmin*nHeI;
      segment[iseg].optical_depth_HeII += rmin*nHeII;
#endif /* __HELIUM__ */	
#ifdef __HYDROGEN_MOL__
      segment[iseg].optical_depth_HM   += rmin*nHM;
      segment[iseg].optical_depth_H2I  += rmin*nH2I;
      segment[iseg].optical_depth_H2II += rmin*nH2II;
#endif /* __HYDROGEN_MOL__ */

    }

    float depth;

    /* convert to the global coordinates */
    x_cur += this_run->xmin_local;
    y_cur += this_run->ymin_local;
    z_cur += this_run->zmin_local;
 
    depth = sqrt(SQR(segment[iseg].xpos_end-x_cur)+
		 SQR(segment[iseg].ypos_end-y_cur)+
		 SQR(segment[iseg].zpos_end-z_cur));

    nH  = MESH(ix_end, iy_end, iz_end).dens;
    nHI = nH*MESH(ix_end, iy_end, iz_end).chem.fHI;
#ifdef __HELIUM__
    nHe   = nH*HELIUM_FACT;
    nHeI  = nHe*MESH(ix_end, iy_end, iz_end).chem.fHeI;
    nHeII = nHe*MESH(ix_end, iy_end, iz_end).chem.fHeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
    nHM   = nH*MESH(ix_end, iy_end, iz_end).chem.fHM;
    nH2I  = nH*MESH(ix_end, iy_end, iz_end).chem.fH2I;
    nH2II = nH*MESH(ix_end, iy_end, iz_end).chem.fH2II;
#endif /* __HYDROGEN_MOL__ */

    segment[iseg].optical_depth_HI   += depth*nHI;
#ifdef __HELIUM__
    segment[iseg].optical_depth_HeI  += depth*nHeI;
    segment[iseg].optical_depth_HeII += depth*nHeII;
#endif
#ifdef __HYDROGEN_MOL__
    segment[iseg].optical_depth_HM   += depth*nHM;
    segment[iseg].optical_depth_H2I  += depth*nH2I;
    segment[iseg].optical_depth_H2II += depth*nH2II;
#endif /* __HYDROGEN_MOL__ */
  }

#ifdef __ARGOT_PROFILE__
  MPI_Barrier(MPI_COMM_WORLD);
  times(&end_tms);
  gettimeofday(&end_tv, NULL);
    
  fprintf(this_run->proc_file,
	  "# calc_optical_depth : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	  timing(start_tms, end_tms), wallclock_timing(start_tv, end_tv));
  fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */

}
