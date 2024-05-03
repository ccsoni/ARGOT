#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "diffuse_photon.h"

inline void ray_cycle(struct ray_info*, struct ray_info*, int*, int, int, float);
inline void reset_I(struct ray_info*);

void calc_diffuse_optical_depth(struct ray_face*, struct radiation_mesh*,
				struct angle_info*, int, int, struct run_param*);


void ray_tracing(long ipix,
		 struct ray_face *ray,
		 struct host_diffuse_param *hd_param,
		 struct run_param *this_run)
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
  
  int gid;
  int loop_size = N_face/RAY_GROUP_NUM;

#ifdef __USE_ATOMIC__
  calc_diffuse_optical_depth( ray, hd_param->rmesh, &hd_param->angle[ipix], 
			      0, N_face, this_run );

#else //!__USE_ATOMIC__  

  for(gid=0; gid<RAY_GROUP_NUM; gid++) {
    calc_diffuse_optical_depth( ray, hd_param->rmesh, &hd_param->angle[ipix], 
				gid*loop_size,(gid+1)*loop_size, this_run );
  }
#endif

}

void calc_diffuse_optical_depth(struct ray_face *ray,
				struct radiation_mesh *rmesh,
				struct angle_info *ang,
				int loop_start, int loop_end,
				struct run_param *this_run)
{
  float  delta_x = this_run->delta_x;
  float  delta_y = this_run->delta_y;
  float  delta_z = this_run->delta_z;
  
  float  xovr=ang->xovr , yovr=ang->yovr , zovr=ang->zovr;
  short  base_id = ang->base_id;
  
  int x_step,y_step,z_step;
  int x_next,y_next,z_next;
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

#pragma omp parallel for schedule(auto)
  for(int ix=loop_start; ix<loop_end; ix++){
    unsigned int pos;    
    int    ix_cur, iy_cur, iz_cur;
    float  dx,dy,dz;
    float  rdx,rdy,rdz;
    float  rmin;
    float  tau,etau,etaum1;
    
    struct ray_info iray;
   
    switch(base_id) {
    case(0): 
    case(1):
      iray = ray->xy[ix];
      break;
      
    case(2): 
    case(3):
      iray = ray->yz[ix];
      break;

    case(4): 
    case(5):
      iray = ray->zx[ix];
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
      continue;    ///retrun or break -> openmp continue
     
    while(1) {
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
#pragma omp atomic
#endif
      rmesh[pos].length += rmin;

      
      /***calc optical depth and intensity***/
      /* HI : recombination */
      tau    = rmesh[pos].absorptionHI * rmin;
      etau   = exp(-tau);
      etaum1 = -expm1(-tau); /* -(exp(-tau)-1.0) = (1.0-exp(-tau)) */
      
#ifdef __USE_ATOMIC__
#pragma omp atomic
#endif
      rmesh[pos].IHI += iray.I_inHI*etaum1; /* (kappa = L/tau) term in calc_GH_tot.c */

      iray.I_inHI = iray.I_inHI*etau + rmesh[pos].source_funcHI*etaum1;  
      
#ifdef __HELIUM__
      /* HeI : recombination */
      tau    = rmesh[pos].absorptionHeI * rmin;
      etau   = exp(-tau);
      etaum1 = -expm1(-tau); /* -(exp(-tau)-1.0) = (1.0-exp(-tau)) */
      
#ifdef __USE_ATOMIC__
#pragma omp atomic
#endif
      rmesh[pos].IHeI    += iray.I_inHeI*etaum1;
      
      iray.I_inHeI = iray.I_inHeI*etau + rmesh[pos].source_funcHeI*etaum1;
      
      /* HeII : recombination */
      tau   = rmesh[pos].absorptionHeII *  rmin;
      etau   = exp(-tau);
      etaum1 = -expm1(-tau); /* -(exp(-tau)-1.0) = (1.0-exp(-tau)) */
      
#ifdef __USE_ATOMIC__
#pragma omp atomic
#endif
      rmesh[pos].IHeII    += iray.I_inHeII*etaum1;
      
      iray.I_inHeII = iray.I_inHeII*etau + rmesh[pos].source_funcHeII*etaum1;
      
#endif //__HELIUM__

      /***end optical depth and intensity***/

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
	    ray->xy[ix] = iray;
	    break;
	  }	  
	  
	  if(ix_cur >= NMESH_X_LOCAL) {
	    ray_cycle(&ray->yz[ix], &iray, &ix_cur, 0, 0, this_run->xmin_local);
	    if(this_run->rank_x == 0) reset_I(&iray);
	  }
	  else if(ix_cur < 0) {
	    ray_cycle(&ray->yz[ix], &iray, &ix_cur, NMESH_X_LOCAL-1, 0, this_run->xmax_local);
	    if(this_run->rank_x == this_run->nnode_x-1)  reset_I(&iray); 
	  }
	  
	  
	  if(iy_cur >= NMESH_Y_LOCAL) {
	    ray_cycle(&ray->zx[ix], &iray, &iy_cur, 0, 1, this_run->ymin_local);
	    if(this_run->rank_y == 0)  reset_I(&iray);
	  }
	  else if(iy_cur < 0) {
	    ray_cycle(&ray->zx[ix], &iray, &iy_cur, NMESH_Y_LOCAL-1, 1, this_run->ymax_local);
	    if(this_run->rank_y == this_run->nnode_y-1)  reset_I(&iray);
	  }
	  
	
	break;  //break switch (0)(1)
	  
	  
	case(2):
	case(3):

	  if( (ix_cur >= NMESH_X_LOCAL) || (ix_cur < 0) ) {
	    ray->yz[ix] = iray;
	    break;
	  }
	  
	  if(iy_cur >= NMESH_Y_LOCAL) {
	    ray_cycle(&ray->zx[ix], &iray, &iy_cur, 0, 1, this_run->ymin_local);
	    if(this_run->rank_y == 0)  reset_I(&iray);
	  }
	  else if(iy_cur < 0) {
	    ray_cycle(&ray->zx[ix], &iray, &iy_cur, NMESH_Y_LOCAL-1, 1, this_run->ymax_local);
	    if(this_run->rank_y == this_run->nnode_y-1)  reset_I(&iray);
	  }

	  
	  if(iz_cur >= NMESH_Z_LOCAL) {
	    ray_cycle(&ray->xy[ix], &iray, &iz_cur, 0, 2, this_run->zmin_local);
	    if(this_run->rank_z == 0)  reset_I(&iray);
	  }
	  else if(iz_cur < 0) {
	    ray_cycle(&ray->xy[ix], &iray, &iz_cur, NMESH_Z_LOCAL-1, 2, this_run->zmax_local);
	    if(this_run->rank_z == this_run->nnode_z-1)  reset_I(&iray);
	  }
	  
	  
	  break;  //break switch (2)(3)
	  
	  
	case(4):
	case(5):

	  if( (iy_cur >= NMESH_Y_LOCAL) || (iy_cur < 0) ) {
	    ray->zx[ix] = iray;
	    break;
	  }

	  if(iz_cur >= NMESH_Z_LOCAL) {
	    ray_cycle(&ray->xy[ix], &iray, &iz_cur, 0, 2, this_run->zmin_local);
	    if(this_run->rank_z == 0)  reset_I(&iray);
	  }	    
	  else if(iz_cur < 0) {
	    ray_cycle(&ray->xy[ix], &iray, &iz_cur, NMESH_Z_LOCAL-1, 2, this_run->zmax_local);
	    if(this_run->rank_z == this_run->nnode_z-1)  reset_I(&iray);
	  }
	  

	  if(ix_cur >= NMESH_X_LOCAL) {
	    ray_cycle(&ray->yz[ix], &iray, &ix_cur, 0, 0, this_run->xmin_local);
	    if(this_run->rank_x == 0)  reset_I(&iray);
	  }	    
	  else if(ix_cur < 0) {
	    ray_cycle(&ray->yz[ix], &iray, &ix_cur, NMESH_X_LOCAL-1, 0, this_run->xmax_local);
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
  } //ix loop
  
}


inline void ray_cycle(struct ray_info *ray, struct ray_info *iray,  
		      int *cur, int next_cur, 
		      int type, float next_pos)
{
  float I_in_tmpHI   = ray->I_inHI;
#ifdef __HELIUM__
  float I_in_tmpHeI  = ray->I_inHeI;
  float I_in_tmpHeII = ray->I_inHeII;
#endif //__HELIUM__ 
  
  *ray = *iray;
  *cur = next_cur;
  
  switch(type) {
  case(0): iray->x = next_pos; break; 
  case(1): iray->y = next_pos; break;   
  case(2): iray->z = next_pos; break;
  }

  iray->I_inHI	 = I_in_tmpHI;
#ifdef __HELIUM__
  iray->I_inHeI  = I_in_tmpHeI;
  iray->I_inHeII = I_in_tmpHeII;
#endif //__HELIUM__
}


inline void reset_I(struct ray_info *ray)
{
  ray->I_inHI = 0.0e0;
#ifdef __HELIUM__
  ray->I_inHeI  = 0.0e0;
  ray->I_inHeII = 0.0e0;
#endif //__HELIUM__ 
}
