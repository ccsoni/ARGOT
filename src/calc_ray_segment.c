#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "run_param.h"
#include "constants.h"
#include "radiation.h"
#include "prototype.h"

#ifndef TINY
#define TINY (1.0e-30)
#endif

void calc_ray_segment(struct light_ray *ray, struct run_param *this_run)
{

//#pragma omp parallel for schedule(dynamic, 32)
#pragma omp parallel for schedule(auto)
  for(uint64_t iray=0;iray<this_run->nray;iray++){
    int nseg;

    int x_step, y_step, z_step;
    int x_next, y_next, z_next;

    float x_target, y_target, z_target;
    float dx_domain, dy_domain, dz_domain;
  
    float sin_theta, cos_theta;
    float sin_phi,   cos_phi;

    float dx, dy, dz;
    float dl, dxy;
  
    float xovr, yovr, zovr;
    float rmin, rx, ry, rz;
  
    struct light_ray *this_ray;

    /* domain indices of the target */
    int   ixd_tgt, iyd_tgt, izd_tgt;
    /* domain indices of the source */
    int   ixd_src, iyd_src, izd_src;

    this_ray = ray+iray;

    /* coordinates of the target position */
    x_target = this_run->xmin + ((float)this_ray->ix_target+0.5)*this_run->delta_x;
    y_target = this_run->ymin + ((float)this_ray->iy_target+0.5)*this_run->delta_y;
    z_target = this_run->zmin + ((float)this_ray->iz_target+0.5)*this_run->delta_z;

    /* side lengths of the MPI-parallelized domain */
    dx_domain = (this_run->xmax-this_run->xmin)/(float)NNODE_X;
    dy_domain = (this_run->ymax-this_run->ymin)/(float)NNODE_Y;
    dz_domain = (this_run->zmax-this_run->zmin)/(float)NNODE_Z;

    ixd_tgt = (int)((x_target-this_run->xmin)/dx_domain);
    iyd_tgt = (int)((y_target-this_run->ymin)/dy_domain);
    izd_tgt = (int)((z_target-this_run->zmin)/dz_domain);

    dx = x_target - this_ray->src.xpos;
    dy = y_target - this_ray->src.ypos;
    dz = z_target - this_ray->src.zpos;

    dl = sqrt(SQR(dx)+SQR(dy)+SQR(dz)+TINY);

    cos_theta = dz/dl;
    sin_theta = sqrt(1.0-SQR(cos_theta));
    
    dxy = dl*sin_theta;
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

    /* in the case that the sources' positions are outside of the simulation volume. */
    float outside_len = 0.0;
    if(this_ray->src.xpos < this_run->xmin)  outside_len = (this_run->xmin - this_ray->src.xpos)/xovr;
    if(this_ray->src.xpos > this_run->xmax)  outside_len = (this_run->xmax - this_ray->src.xpos)/xovr;
    
    if(this_ray->src.ypos < this_run->ymin)  outside_len = (this_run->ymin - this_ray->src.ypos)/yovr;
    if(this_ray->src.ypos > this_run->ymax)  outside_len = (this_run->ymax - this_ray->src.ypos)/yovr;
    
    if(this_ray->src.zpos < this_run->zmin)  outside_len = (this_run->zmin - this_ray->src.zpos)/zovr;
    if(this_ray->src.zpos > this_run->zmax)  outside_len = (this_run->zmax - this_ray->src.zpos)/zovr;

    if(outside_len != 0.0) {
      this_ray->segment[0].xpos_start = this_ray->src.xpos + outside_len*xovr;
      this_ray->segment[0].ypos_start = this_ray->src.ypos + outside_len*yovr;
      this_ray->segment[0].zpos_start = this_ray->src.zpos + outside_len*zovr;
    }else{
      this_ray->segment[0].xpos_start = this_ray->src.xpos;
      this_ray->segment[0].ypos_start = this_ray->src.ypos;
      this_ray->segment[0].zpos_start = this_ray->src.zpos;
    }

    ixd_src = (int)((this_ray->segment[0].xpos_start-this_run->xmin)/dx_domain);
    iyd_src = (int)((this_ray->segment[0].ypos_start-this_run->ymin)/dy_domain);
    izd_src = (int)((this_ray->segment[0].zpos_start-this_run->zmin)/dz_domain);

    nseg = 0;
    this_ray->segment[0].target_rank = mpi_rank(ixd_tgt, iyd_tgt, izd_tgt);
    this_ray->segment[0].local_rank = mpi_rank(ixd_src, iyd_src, izd_src);
    this_ray->segment[0].ray_indx = iray;

    while(ixd_src != ixd_tgt || iyd_src != iyd_tgt || izd_src != izd_tgt) {
      //dx = dx_domain*(x_step+ixd_src) - this_ray->segment[nseg].xpos_start;
      //dy = dy_domain*(y_step+iyd_src) - this_ray->segment[nseg].ypos_start;
      //dz = dz_domain*(z_step+izd_src) - this_ray->segment[nseg].zpos_start;
      dx = this_run->xmin + dx_domain*(x_step+ixd_src) - this_ray->segment[nseg].xpos_start;
      dy = this_run->ymin + dy_domain*(y_step+iyd_src) - this_ray->segment[nseg].ypos_start;
      dz = this_run->zmin + dz_domain*(z_step+izd_src) - this_ray->segment[nseg].zpos_start;
     
      rx = dx/xovr;
      ry = dy/yovr;
      rz = dz/zovr;
      
      rmin = fminf(rx, fminf(ry,rz));

      if(rmin == rx) {
	ixd_src += x_next;
      }

      if(rmin == ry) {
	iyd_src += y_next;
      }

      if(rmin == rz) {
	izd_src += z_next;
      }

      this_ray->segment[nseg].xpos_end = 
	this_ray->segment[nseg].xpos_start + rmin*xovr;
      
      this_ray->segment[nseg].ypos_end = 
	this_ray->segment[nseg].ypos_start + rmin*yovr;
      
      this_ray->segment[nseg].zpos_end = 
	this_ray->segment[nseg].zpos_start + rmin*zovr;

      //this_ray->segment[nseg].optical_depth = rmin;
      this_ray->segment[nseg].optical_depth_HI = 0.0;
#ifdef __HELIUM__
      this_ray->segment[nseg].optical_depth_HeI  = 0.0;
      this_ray->segment[nseg].optical_depth_HeII = 0.0;
#endif
#ifdef __HYDROGEN_MOL__
      this_ray->segment[nseg].optical_depth_HM   = 0.0;
      this_ray->segment[nseg].optical_depth_H2I  = 0.0;
      this_ray->segment[nseg].optical_depth_H2II = 0.0;
#endif
      
      this_ray->segment[nseg+1].target_rank = mpi_rank(ixd_tgt, iyd_tgt, izd_tgt);
      this_ray->segment[nseg+1].local_rank  = mpi_rank(ixd_src, iyd_src, izd_src);
      this_ray->segment[nseg+1].xpos_start = this_ray->segment[nseg].xpos_end;
      this_ray->segment[nseg+1].ypos_start = this_ray->segment[nseg].ypos_end;
      this_ray->segment[nseg+1].zpos_start = this_ray->segment[nseg].zpos_end;
      this_ray->segment[nseg+1].ray_indx   = iray;
    
      nseg++;
    }

    this_ray->segment[nseg].xpos_end = x_target;
    this_ray->segment[nseg].ypos_end = y_target;
    this_ray->segment[nseg].zpos_end = z_target;

    //for(int inu=0;inu<NGRID_NU;inu++) this_ray->segment[nseg].optical_depth[inu] = 0.0;
    this_ray->segment[nseg].optical_depth_HI = 0.0;
#ifdef __HELIUM__
    this_ray->segment[nseg].optical_depth_HeI = 0.0;
    this_ray->segment[nseg].optical_depth_HeII = 0.0;
#endif
#ifdef __HYDROGEN_MOL__
    this_ray->segment[nseg].optical_depth_HM   = 0.0;
    this_ray->segment[nseg].optical_depth_H2I  = 0.0;
    this_ray->segment[nseg].optical_depth_H2II = 0.0;
#endif

    this_ray->num_segment = nseg+1;

    assert(this_ray->num_segment < NSEG_PER_RAY);
  }

#if 0
  if(this_ray->num_segment >= NSEG_PER_RAY) {
    fprintf(this_run->proc_file,
	    "### num_segment = %d \n ",this_ray->num_segment);
    fprintf(this_run->proc_file,
	    "### src: (x,y,z) = (%14.6e, %14.6e, %14.6e)\n ",
	    this_ray->src.xpos, this_ray->src.ypos, this_ray->src.zpos);
    fprintf(this_run->proc_file,
	    "### target: (x,y,z) = (%14.6e, %14.6e, %14.6e)\n",
	    this_run->xmin+(this_ray->ix_target+0.5)*this_run->delta_x,
	    this_run->ymin+(this_ray->iy_target+0.5)*this_run->delta_y,
	    this_run->zmin+(this_ray->iz_target+0.5)*this_run->delta_z);
  }
#endif

}
