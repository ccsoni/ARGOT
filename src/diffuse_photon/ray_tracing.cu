#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "diffuse_photon.h"
#include "cuda_mem_space.h"


__global__ void calc_diffuse_optical_depth_kernel(struct ray_info*, struct ray_info*, struct ray_info*,
						  struct radiation_mesh*, const struct angle_info* __restrict__,
						  const struct run_param* __restrict__, int64_t);

__forceinline__ __device__ void ray_cycle(struct ray_info*, struct ray_info*, int*, int, int, float);
__forceinline__ __device__ void reset_I(struct ray_info*);


extern "C" 
void ray_tracing(long ipix,
		 struct ray_face *ray,
		 struct cuda_mem_space *cuda_mem,
		 struct cuda_diffuse_param *cd_param,
		 struct host_diffuse_param *hd_param,
		 cudaStream_t strm, int device_id)
{
  unsigned int N_face;
  short base_id = hd_param->angle[ipix].base_id;
  
  switch(base_id){
  case(0):
  case(1):
    N_face = NMESH_XY_LOCAL;
  break;
    
  case(2):
  case(3):
    N_face = NMESH_YZ_LOCAL;
  break;
    
  case(4):
  case(5):
    N_face = NMESH_ZX_LOCAL;
  break;
  }
  
  int64_t offset;
  
#ifdef __USE_ATOMIC__
  dim3 block(N_face/NMESH_PER_BLOCK_DRT, 1, 1);
  dim3 thread(NMESH_PER_BLOCK_DRT, 1, 1);
  offset = 0;
  
  calc_diffuse_optical_depth_kernel<<<block, thread, 0, strm >>>
    (ray->xy, ray->yz, ray->zx, 
     cd_param->rmesh, &cd_param->angle[ipix],
     cuda_mem->this_run_dev, offset);
  
#else //!__USE_ATOMIC__

  int gid;
  int loop_size = N_face/RAY_GROUP_NUM;
  
  dim3 block(N_face/(RAY_GROUP_NUM*NMESH_PER_BLOCK_DRT), 1, 1);
  dim3 thread(NMESH_PER_BLOCK_DRT, 1, 1);
  
  for(gid=0; gid<RAY_GROUP_NUM; gid++) {
    offset = gid*loop_size;
    
    calc_diffuse_optical_depth_kernel<<<block, thread, 0, strm >>>
      (ray->xy, ray->yz, ray->zx, 
       cd_param->rmesh, &cd_param->angle[ipix],
       cuda_mem->this_run_dev, offset); 
  }
    
#endif //__USE_ATOMIC__

  CUDA_SAFE( cudaStreamSynchronize(strm) );
  
}


__forceinline__ __device__ void calc_intensity(float rmin, struct radiation_mesh *rmesh, struct ray_info *iray)
{
  float tau,etau,etaum1;

  /*** nu1 ***/
  tau    = rmesh->absorption_nu1 * rmin;
  etau   = exp(-tau);
  etaum1 = -expm1(-tau); /* -(exp(-tau)-1.0) = (1.0-exp(-tau)) */
      
#ifdef __USE_ATOMIC__
  atomicAdd(&rmesh->I_nu1, iray->I_in_nu1*etaum1);
#else //!__USE_ATOMIC__    
  rmesh->I_nu1    += iray->I_in_nu1*etaum1;
#endif
  iray->I_in_nu1 = iray->I_in_nu1*etau + rmesh->source_func_nu1*etaum1;  
  
  
#ifdef __HELIUM__
  /*** nu2 ***/
  tau  = rmesh->absorption_nu2 * rmin;
  etau = exp(-tau); etaum1 = -expm1(-tau);

#ifdef __USE_ATOMIC__
  atomicAdd(&rmesh->I_nu2, iray->I_in_nu2*etaum1);
#else //!__USE_ATOMIC__    
  rmesh->I_nu2    += iray->I_in_nu2*etaum1;
#endif
  iray->I_in_nu2 = iray->I_in_nu2*etau + rmesh->source_func_nu2*etaum1;  
  
  /*** nu3 ***/
  tau  = rmesh->absorption_nu3 * rmin;
  etau = exp(-tau); etaum1 = -expm1(-tau);
  
#ifdef __USE_ATOMIC__
  atomicAdd(&rmesh->I_nu3, iray->I_in_nu3*etaum1);
#else //!__USE_ATOMIC__    
  rmesh->I_nu3    += iray->I_in_nu3*etaum1;
#endif
  iray->I_in_nu3 = iray->I_in_nu3*etau + rmesh->source_func_nu3*etaum1;  
  
  
#ifdef __HELIUM_BB__
  /*** nu4 ***/
  tau  = rmesh->absorption_nu4 * rmin;
  etau = exp(-tau); etaum1 = -expm1(-tau);

#ifdef __USE_ATOMIC__
  atomicAdd(&rmesh->I_nu4, iray->I_in_nu4*etaum1);
#else //!__USE_ATOMIC__    
  rmesh->I_nu4    += iray->I_in_nu4*etaum1;
#endif
  iray->I_in_nu4 = iray->I_in_nu4*etau + rmesh->source_func_nu4*etaum1;  
  
  /*** nu5 ***/
  tau  = rmesh->absorption_nu5 * rmin;
  etau = exp(-tau); etaum1 = -expm1(-tau);
  
#ifdef __USE_ATOMIC__
  atomicAdd(&rmesh->I_nu5, iray->I_in_nu5*etaum1);
#else //!__USE_ATOMIC__    
  rmesh->I_nu5    += iray->I_in_nu5*etaum1;
#endif
  iray->I_in_nu5 = iray->I_in_nu5*etau + rmesh->source_func_nu5*etaum1;  
  
  /*** nu6 ***/
  tau  = rmesh->absorption_nu6 * rmin;
  etau = exp(-tau); etaum1 = -expm1(-tau);

#ifdef __USE_ATOMIC__
  atomicAdd(&rmesh->I_nu6, iray->I_in_nu6*etaum1);
#else //!__USE_ATOMIC__    
  rmesh->I_nu6    += iray->I_in_nu6*etaum1;
#endif
  iray->I_in_nu6 = iray->I_in_nu6*etau + rmesh->source_func_nu6*etaum1;  

#endif //__HELIUM_BB__
#endif //__HELIUM__
}


__global__ void calc_diffuse_optical_depth_kernel(struct ray_info *ray_xy,
						  struct ray_info *ray_yz,
						  struct ray_info *ray_zx,
						  struct radiation_mesh *rmesh,
						  const struct angle_info* __restrict__ ang,
						  const struct run_param* __restrict__ this_run,
						  int64_t offset)
{
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x + offset;
  
  float  delta_x = this_run->delta_x;
  float  delta_y = this_run->delta_y;
  float  delta_z = this_run->delta_z;
  
  float  xovr=ang->xovr , yovr=ang->yovr , zovr=ang->zovr;
  short  base_id = ang->base_id;
  
  int x_step,y_step,z_step;
  int x_next,y_next,z_next;

  unsigned int pos;    
  int    ix_cur, iy_cur, iz_cur;
  float  dx,dy,dz;
  float  rdx,rdy,rdz;
  float  rmin; 

  struct ray_info iray;

  if(xovr > 0.0e0f){ 
    x_step = 1;  x_next = 1;
  }else{
    x_step = 0;  x_next = -1;
  }
  
  if(yovr > 0.0e0f){
    y_step = 1;  y_next = 1;
  }else{
    y_step = 0;  y_next = -1;
  }
  
  if(zovr > 0.0e0f){
    z_step = 1;  z_next = 1;
  }else{
    z_step = 0;  z_next = -1;
  }
  
  
  switch(base_id) {
  case(0):  case(1):
    iray = ray_xy[ix];
    break;
    
  case(2):  case(3):
    iray = ray_yz[ix];
    break;
    
  case(4):  case(5):
    iray = ray_zx[ix];
    break;
  }
  
  /* (global_position-offset)/delta) */
  ix_cur = (iray.x-this_run->xmin_local)/delta_x;
  iy_cur = (iray.y-this_run->ymin_local)/delta_y;
  iz_cur = (iray.z-this_run->zmin_local)/delta_z;
   
  //start point is grid point.
  if( xovr < 0.0e0f && ix_cur==NMESH_X_LOCAL) --ix_cur;
  if( yovr < 0.0e0f && iy_cur==NMESH_Y_LOCAL) --iy_cur;
  if( zovr < 0.0e0f && iz_cur==NMESH_Z_LOCAL) --iz_cur;
     
  if ( (ix_cur >= NMESH_X_LOCAL) || 
       (iy_cur >= NMESH_Y_LOCAL) ||
       (iz_cur >= NMESH_Z_LOCAL) ||
       (ix_cur < 0) || 
       (iy_cur < 0) || 
       (iz_cur < 0) )
    return;    ///return or break -> openmp continue

  while(1){
    pos = iz_cur + NMESH_Z_LOCAL*(iy_cur + NMESH_Y_LOCAL*ix_cur);

    /***start geometric calculation***/
    dx = delta_x*(x_step+ix_cur) - (iray.x-this_run->xmin_local); 
    dy = delta_y*(y_step+iy_cur) - (iray.y-this_run->ymin_local);
    dz = delta_z*(z_step+iz_cur) - (iray.z-this_run->zmin_local);
    
    rdx = dx/xovr;
    rdy = dy/yovr;
    rdz = dz/zovr;

    if(rdx<0.0) rdx=0.0;
    if(rdy<0.0) rdy=0.0;
    if(rdz<0.0) rdz=0.0;
    
    rmin = fminf(fminf(rdx,rdy),rdz);
   
    iray.x += rmin*xovr;
    iray.y += rmin*yovr;
    iray.z += rmin*zovr;

    if(rdx==rmin)   ix_cur += x_next;
    if(rdy==rmin)   iy_cur += y_next;
    if(rdz==rmin)   iz_cur += z_next;
    /***end geometric calculation***/


    /* calc total length */
    rmin *= this_run->lunit;
#ifdef __COSMOLOGICAL__
    rmin *= this_run->anow;
#endif
   
#ifdef __USE_ATOMIC__
    atomicAdd(&rmesh[pos].length, rmin);
#else //!__USE_ATOMIC__    
    rmesh[pos].length  += rmin;
#endif
    
    /* calc optical depth and intensity */
    calc_intensity(rmin, &rmesh[pos], &iray);

    
    if ( (ix_cur >= NMESH_X_LOCAL) ||
	 (iy_cur >= NMESH_Y_LOCAL) ||
	 (iz_cur >= NMESH_Z_LOCAL) ||
	 (ix_cur < 0) ||
	 (iy_cur < 0) ||
	 (iz_cur < 0)) {
      
      
      switch(base_id){
      case(0):
      case(1):
	
	if( (iz_cur >= NMESH_Z_LOCAL) || (iz_cur < 0) ) {
	  ray_xy[ix] = iray;
	  return;
	}	  
      
        if(ix_cur >= NMESH_X_LOCAL) {
	  ray_cycle(&ray_yz[ix], &iray, &ix_cur, 0, 0, this_run->xmin_local);
	  if(this_run->rank_x == 0)  reset_I(&iray);
	}
	else if(ix_cur < 0) {
	  ray_cycle(&ray_yz[ix], &iray, &ix_cur, NMESH_X_LOCAL-1, 0, this_run->xmax_local);
	  if(this_run->rank_x == this_run->nnode_x-1)  reset_I(&iray); 
	}
	
	
	if(iy_cur >= NMESH_Y_LOCAL) {
	  ray_cycle(&ray_zx[ix], &iray, &iy_cur, 0, 1, this_run->ymin_local);
	  if(this_run->rank_y == 0)  reset_I(&iray);  /**check**/
	}
	else if(iy_cur < 0) {
	  ray_cycle(&ray_zx[ix], &iray, &iy_cur, NMESH_Y_LOCAL-1, 1, this_run->ymax_local);
	  if(this_run->rank_y == this_run->nnode_y-1)  reset_I(&iray);
	}
	
	break;  //break switch (0)(1)
	
	
      case(2):
      case(3):
	
	if( (ix_cur >= NMESH_X_LOCAL) || (ix_cur < 0) ) {
	  ray_yz[ix] = iray;
	  return;
	}
      
        if(iy_cur >= NMESH_Y_LOCAL) {
	  ray_cycle(&ray_zx[ix], &iray, &iy_cur, 0, 1, this_run->ymin_local);
	  if(this_run->rank_y == 0)  reset_I(&iray);
	}
	else if(iy_cur < 0) {
	  ray_cycle(&ray_zx[ix], &iray, &iy_cur, NMESH_Y_LOCAL-1, 1, this_run->ymax_local);
	  if(this_run->rank_y == this_run->nnode_y-1)  reset_I(&iray);
	}
	
	if(iz_cur >= NMESH_Z_LOCAL) {
	  ray_cycle(&ray_xy[ix], &iray, &iz_cur, 0, 2, this_run->zmin_local);
	  if(this_run->rank_z == 0)  reset_I(&iray);
	}
	else if(iz_cur < 0) {
	  ray_cycle(&ray_xy[ix], &iray, &iz_cur, NMESH_Z_LOCAL-1, 2, this_run->zmax_local);
	  if(this_run->rank_z == this_run->nnode_z-1)  reset_I(&iray);
	}
		
	break;  //break switch (2)(3)
	
	
      case(4):
      case(5):
	
	if( (iy_cur >= NMESH_Y_LOCAL) || (iy_cur < 0) ) {
	  ray_zx[ix] = iray;
	  return;
	}
      
        if(iz_cur >= NMESH_Z_LOCAL) {
	  ray_cycle(&ray_xy[ix], &iray, &iz_cur, 0, 2, this_run->zmin_local);
	  if(this_run->rank_z == 0)  reset_I(&iray);
	}	    
	else if(iz_cur < 0) {
	  ray_cycle(&ray_xy[ix], &iray, &iz_cur, NMESH_Z_LOCAL-1, 2, this_run->zmax_local);
	  if(this_run->rank_z == this_run->nnode_z-1)  reset_I(&iray);
	}
		
	if(ix_cur >= NMESH_X_LOCAL) {
	  ray_cycle(&ray_yz[ix], &iray, &ix_cur, 0, 0, this_run->xmin_local);
	  if(this_run->rank_x == 0)  reset_I(&iray);
	}	    
	else if(ix_cur < 0) {
	  ray_cycle(&ray_yz[ix], &iray, &ix_cur, NMESH_X_LOCAL-1, 0, this_run->xmax_local);
	  if(this_run->rank_x == this_run->nnode_x-1)  reset_I(&iray);
	}      
	
	break;  //break switch (4)(5)
	
      } // end switch
      

      if ( (ix_cur >= NMESH_X_LOCAL) ||
       	   (iy_cur >= NMESH_Y_LOCAL) ||
       	   (iz_cur >= NMESH_Z_LOCAL) ||
       	   (ix_cur < 0) ||
       	   (iy_cur < 0) ||
       	   (iz_cur < 0))  break;

    }
    
  } //while loop
  
}



__forceinline__ __device__ void ray_cycle(struct ray_info *ray, struct ray_info *iray,  
					  int *cur, int next_cur, 
					  int type, float next_pos)
{
  float I_in_tmp_nu1 =  ray->I_in_nu1;
#ifdef __HELIUM__
  float I_in_tmp_nu2 =  ray->I_in_nu2;
  float I_in_tmp_nu3 =  ray->I_in_nu3;
#ifdef __HELIUM_BB__
  float I_in_tmp_nu4 =  ray->I_in_nu4;
  float I_in_tmp_nu5 =  ray->I_in_nu5;
  float I_in_tmp_nu6 =  ray->I_in_nu6;
#endif //__HELIUM_BB__
#endif //__HELIUM__

  *ray = *iray;
  *cur = next_cur;
  
  switch(type) {
  case(0): iray->x = next_pos; break; 
  case(1): iray->y = next_pos; break;   
  case(2): iray->z = next_pos; break;
  }

  iray->I_in_nu1 = I_in_tmp_nu1;
#ifdef __HELIUM__
  iray->I_in_nu2 = I_in_tmp_nu2;
  iray->I_in_nu3 = I_in_tmp_nu3;
#ifdef __HELIUM_BB__
  iray->I_in_nu4 = I_in_tmp_nu4;
  iray->I_in_nu5 = I_in_tmp_nu5;
  iray->I_in_nu6 = I_in_tmp_nu6;
#endif //__HELIUM_BB__
#endif //__HELIUM__
}


__forceinline__ __device__ void reset_I(struct ray_info *ray)
{
  ray->I_in_nu1 = 0.0e0;
#ifdef __HELIUM__
  ray->I_in_nu2 = 0.0e0;
  ray->I_in_nu3 = 0.0e0;
#ifdef __HELIUM_BB__
  ray->I_in_nu4 = 0.0e0;
  ray->I_in_nu5 = 0.0e0;
  ray->I_in_nu6 = 0.0e0;
#endif //__HELIUM_BB__
#endif //__HELIUM__
}
