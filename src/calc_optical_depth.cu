#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>

#include "run_param.h"
#include "constants.h"
#include "fluid.h"
#include "radiation.h"
#include "cuda_mem_space.h"

extern "C" float timing(struct tms, struct tms);
extern "C" float wallclock_timing(struct timeval, struct timeval);

#ifndef TINY
#define TINY (1.0e-31)
#endif

#define MESH(ix,iy,iz) mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))]

__global__ void calc_optical_depth_kernel(const struct fluid_mesh* __restrict__, 
					  struct ray_segment*,
					  const struct run_param* __restrict__,
					  uint64_t);

extern "C" 
void calc_optical_depth(struct ray_segment *seg,
			struct cuda_mem_space *cuda_mem,
			struct cuda_param *this_cuda,
			struct run_param *this_run)
{
  int idev;
  cudaError_t err;

  static uint64_t nseg_to_dev[NMAX_CUDA_DEV];
  static uint64_t offset_seg[NMAX_CUDA_DEV];

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

  uint64_t nseg_remained = this_run->nseg;
  uint64_t nseg_per_dev = this_run->nseg/this_cuda->num_cuda_dev + 1;

  /* Determine the # of ray_segment dealt by each GPU */
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {

    if(nseg_remained < nseg_per_dev) {
      nseg_to_dev[idev] = nseg_remained;
    }else{
      nseg_to_dev[idev] = nseg_per_dev;
    }
    nseg_remained -= nseg_to_dev[idev];

    if(idev==0) { 
      offset_seg[idev]=0;
    }else{
      offset_seg[idev] = offset_seg[idev-1] + nseg_to_dev[idev-1];
    }

    cudaSetDevice(idev);
    err = cudaMalloc((void **) &(cuda_mem[idev].segment_dev),
		     sizeof(struct ray_segment)*nseg_to_dev[idev]);
    assert(err == cudaSuccess);
  }

  /* Issuing the CUDA kernels */
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    uint64_t nseg_waiting, offset;

    cudaSetDevice(idev);
    err = cudaMemcpyAsync(cuda_mem[idev].segment_dev,seg+offset_seg[idev],
			  sizeof(struct ray_segment)*nseg_to_dev[idev],
			  cudaMemcpyHostToDevice, this_cuda->strm[idev]);
    assert(err == cudaSuccess);

    /* determine the kernel geometry and execute the kernels */
    nseg_waiting = nseg_to_dev[idev];
    offset = 0;
    while(nseg_waiting > 0) {
      int nthread, nblock;

      if(nseg_waiting >= NSEG_MAX_PER_DEV) {
	nthread = NSEG_MAX_PER_BLOCK;
	nblock  = NSEG_MAX_PER_DEV/NSEG_MAX_PER_BLOCK;

	nseg_waiting -= (nthread*nblock);
      }else if(nseg_waiting >= NSEG_MAX_PER_BLOCK) {
	nthread = NSEG_MAX_PER_BLOCK;
	nblock  = nseg_waiting/NSEG_MAX_PER_BLOCK;

	nseg_waiting -= (nthread*nblock);
      }else{ /* nseg_waiting < NSEG_MAX_PER_BLOCK */
	nthread = nseg_waiting;
	nblock  = 1;

	nseg_waiting -= (nthread*nblock);
      }
      
      dim3 nthrd(nthread, 1, 1);
      dim3 nblck(nblock, 1, 1);

      calc_optical_depth_kernel<<<nblck, nthrd, 0, this_cuda->strm[idev]>>>
	                 (cuda_mem[idev].mesh_dev, cuda_mem[idev].segment_dev, 
			  cuda_mem[idev].this_run_dev, offset);
      offset += nthread*nblock;
      
    }

    err = cudaMemcpyAsync(seg+offset_seg[idev], cuda_mem[idev].segment_dev,
			  sizeof(struct ray_segment)*nseg_to_dev[idev],
			  cudaMemcpyDeviceToHost,this_cuda->strm[idev]);
    assert(err == cudaSuccess);
  }

  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaStreamSynchronize(this_cuda->strm[idev]);
  }

  /* Free the allocated device memory */
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaFree(cuda_mem[idev].segment_dev);
  }

#ifdef __ARGOT_PROFILE__
    times(&end_tms);
    gettimeofday(&end_tv, NULL);

    fprintf(this_run->proc_file,
	    "# calc_optical_depth : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	    timing(start_tms, end_tms), wallclock_timing(start_tv, end_tv));
    fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */

}

__global__ void calc_optical_depth_kernel(const struct fluid_mesh* __restrict__ mesh, 
					  struct ray_segment *segment,
					  const struct run_param* __restrict__ this_run, 
					  uint64_t offset)
{
  uint64_t tid;

  float rmin, rx, ry, rz;

  int   ix_end, iy_end, iz_end;
  int   ix_cur, iy_cur, iz_cur;

  float dx, dy, dz;
  float dl, dxy;

  float nH, nHI;
#ifdef __HELIUM__
  float nHe, nHeI, nHeII;
#endif
#ifdef __HYDROGEN_MOL__
  float nHM, nH2I, nH2II;
#endif
  
  float xovr, yovr, zovr;
  float x_step, y_step, z_step;
  float x_next, y_next, z_next;

  float cos_phi, sin_phi;
  float cos_theta, sin_theta;

  /* current position on the ray segment in the local coordinate */
  float x_cur, y_cur, z_cur;

  struct ray_segment *tgt_seg;

  tid = blockIdx.x*blockDim.x + threadIdx.x + offset;

  tgt_seg = &(segment[tid]);

  tgt_seg->optical_depth_HI = 0.0;
#ifdef __HELIUM__
  tgt_seg->optical_depth_HeI  = 0.0;
  tgt_seg->optical_depth_HeII = 0.0;
#endif /* __HELIUM__ */  
#ifdef __HYDROGEN_MOL__
  tgt_seg->optical_depth_HM   = 0.0;
  tgt_seg->optical_depth_H2I  = 0.0;
  tgt_seg->optical_depth_H2II = 0.0;
#endif

  
  ix_end = (int)((tgt_seg->xpos_end-this_run->xmin_local)/this_run->delta_x);
  iy_end = (int)((tgt_seg->ypos_end-this_run->ymin_local)/this_run->delta_y);
  iz_end = (int)((tgt_seg->zpos_end-this_run->zmin_local)/this_run->delta_z);

  ix_end = MAX(MIN(ix_end, NMESH_X_LOCAL-1),0);
  iy_end = MAX(MIN(iy_end, NMESH_Y_LOCAL-1),0);
  iz_end = MAX(MIN(iz_end, NMESH_Z_LOCAL-1),0);

  ix_cur = (int)((tgt_seg->xpos_start-this_run->xmin_local)/this_run->delta_x);
  iy_cur = (int)((tgt_seg->ypos_start-this_run->ymin_local)/this_run->delta_y);
  iz_cur = (int)((tgt_seg->zpos_start-this_run->zmin_local)/this_run->delta_z);

  ix_cur = MAX(MIN(ix_cur, NMESH_X_LOCAL-1),0);
  iy_cur = MAX(MIN(iy_cur, NMESH_Y_LOCAL-1),0);
  iz_cur = MAX(MIN(iz_cur, NMESH_Z_LOCAL-1),0);
  
  dx = tgt_seg->xpos_end-tgt_seg->xpos_start;
  dy = tgt_seg->ypos_end-tgt_seg->ypos_start;
  dz = tgt_seg->zpos_end-tgt_seg->zpos_start;

  dl = sqrt(SQR(dx)+SQR(dy)+SQR(dz)+TINY);
  dxy = sqrt(SQR(dx)+SQR(dy));

  cos_theta = dz/dl;
  sin_theta = dxy/dl;

  //  sin_theta = sqrt(1.0-SQR(cos_theta));
  //  dxy = dl*sin_theta;
  cos_phi = dx/(dxy+TINY);
  sin_phi = dy/(dxy+TINY);

  xovr = cos_phi*sin_theta;
  if(fabsf(xovr)<TINY) {
    xovr = (xovr >= 0.0 ? TINY : -TINY);
  }

  yovr = sin_phi*sin_theta;
  if(fabsf(yovr)<TINY) {
    yovr = (yovr >= 0.0 ? TINY : -TINY);
  }

  zovr = cos_theta;
  if(fabsf(zovr)<TINY) {
    zovr = (zovr >= 0.0 ? TINY : -TINY);
  }

  if(xovr > 0.e0) {
    x_step = 1; x_next = 1;
  }else {
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

  /* local coordinate with respect to each parallel domain */
  x_cur = tgt_seg->xpos_start - this_run->xmin_local;
  y_cur = tgt_seg->ypos_start - this_run->ymin_local;
  z_cur = tgt_seg->zpos_start - this_run->zmin_local;

  while( (ix_cur != ix_end || iy_cur != iy_end || iz_cur != iz_end) &&
	 ( 0<=ix_cur && ix_cur<NMESH_X_LOCAL ) &&
	 ( 0<=iy_cur && iy_cur<NMESH_Y_LOCAL ) &&
	 ( 0<=iz_cur && iz_cur<NMESH_Z_LOCAL ) ) {
    
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

    rmin = fminf(rx, fminf(ry,rz));
    //    rmin = MIN(rx, MIN(ry, rz));
    if(rmin == rx) ix_cur += x_next;
    if(rmin == ry) iy_cur += y_next;
    if(rmin == rz) iz_cur += z_next;

    x_cur += rmin*xovr;
    y_cur += rmin*yovr;
    z_cur += rmin*zovr;

    tgt_seg->optical_depth_HI += rmin*nHI;
#ifdef __HELIUM__
    tgt_seg->optical_depth_HeI  += rmin*nHeI;
    tgt_seg->optical_depth_HeII += rmin*nHeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
    tgt_seg->optical_depth_HM   += rmin*nHM;
    tgt_seg->optical_depth_H2I  += rmin*nH2I;
    tgt_seg->optical_depth_H2II += rmin*nH2II;
#endif /* __HYDROGEN_MOL__ */
  }

  float depth;

  /* convert to the global coordinates */
  x_cur += this_run->xmin_local;
  y_cur += this_run->ymin_local;
  z_cur += this_run->zmin_local;

  depth = sqrt(SQR(tgt_seg->xpos_end-x_cur)+
	       SQR(tgt_seg->ypos_end-y_cur)+
	       SQR(tgt_seg->zpos_end-z_cur));

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

  tgt_seg->optical_depth_HI   += depth*nHI;
#ifdef __HELIUM__
  tgt_seg->optical_depth_HeI  += depth*nHeI;
  tgt_seg->optical_depth_HeII += depth*nHeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
  tgt_seg->optical_depth_HM   += depth*nHM;
  tgt_seg->optical_depth_H2I  += depth*nH2I;
  tgt_seg->optical_depth_H2II += depth*nH2II;
#endif /* __HYDROGEN_MOL__ */
  
}
