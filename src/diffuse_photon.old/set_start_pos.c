#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "diffuse_photon.h"


inline void set_position(int, int, int,
			 struct angle_info*, struct ray_info*,
			 struct run_param*);

inline void set_boundary_ray(struct ray_info*, long,
			     struct host_diffuse_param*, 
			     struct run_param*);


void set_ray_start_position(struct ray_face *ray, long ipix,
			    struct host_diffuse_param *hd_param, 
			    struct run_param *this_run)
{
  short base_id = hd_param->angle[ipix].base_id;

  switch(base_id) {
  case(0):
    if(this_run->rank_z == 0)
      set_boundary_ray(ray->xy, ipix, hd_param, this_run);
    break;
    
  case(1):
    if(this_run->rank_z == this_run->nnode_z-1)
      set_boundary_ray(ray->xy, ipix, hd_param, this_run);
    break;

  case(2):
    if(this_run->rank_x == 0)
      set_boundary_ray(ray->yz, ipix, hd_param, this_run);
    break;

  case(3):
    if(this_run->rank_x == this_run->nnode_x-1)
      set_boundary_ray(ray->yz, ipix, hd_param, this_run);
    break;

  case(4):
    if(this_run->rank_y == 0)
      set_boundary_ray(ray->zx, ipix, hd_param, this_run);
    break;
    
  case(5):
    if(this_run->rank_y == this_run->nnode_y-1)
      set_boundary_ray(ray->zx, ipix, hd_param, this_run);
    break;
  }
}


inline void set_boundary_ray(struct ray_info *ray, long ipix,
			     struct host_diffuse_param *hd_param, 
			     struct run_param *this_run)
{
  int  i,j,k;
  unsigned int  id=0, begin_address=0;

  switch(hd_param->angle[ipix].base_id){
  case(0):
  case(1):
    
    begin_address = NMESH_XY_LOCAL/RAY_GROUP_NUM;
    
    if(hd_param->angle[ipix].base_id == 0)  k = 0; 
    else                                    k = NMESH_Z_LOCAL - 1;
    
#ifdef __USE_ATOMIC__
#pragma omp parallel for schedule(auto) private(i,j,k,id)
      for(i=0; i<NMESH_X_LOCAL; i++){
	for(j=0; j<NMESH_Y_LOCAL; j++){
	  
	  id = j + NMESH_Y_LOCAL*i;
	  
	  set_position(i, j, k, &hd_param->angle[ipix],
		       &ray[id], this_run);
	}
      }
#else //!__USE_ATOMIC__
      
#pragma omp parallel for schedule(auto) private(i,j,k,id)
      for(i=0; i<NMESH_X_LOCAL; i=i+2){
	for(j=0; j<NMESH_Y_LOCAL; j=j+2){
	  
	id = (j>>1) + (NMESH_Y_LOCAL>>1)*(i>>1);
	
	set_position(i, j, k, &hd_param->angle[ipix],
		     &ray[id + 0*begin_address], this_run);
	set_position(i, j+1, k,  &hd_param->angle[ipix],
		     &ray[id + 1*begin_address], this_run);
	set_position(i+1, j, k, &hd_param->angle[ipix],
		     &ray[id + 2*begin_address], this_run);
	set_position(i+1, j+1, k, &hd_param->angle[ipix],
		     &ray[id + 3*begin_address], this_run);
	}
      }
#endif //__USE_ATOMIC__
      
      break;
      
  case(2):
  case(3):
    
    begin_address = NMESH_YZ_LOCAL/RAY_GROUP_NUM;
    
    if(hd_param->angle[ipix].base_id == 2)  i = 0; 
    else                                    i = NMESH_X_LOCAL - 1;
    
    
#ifdef __USE_ATOMIC__
#pragma omp parallel for schedule(auto) private(i,j,k,id)
    for(j=0; j<NMESH_Y_LOCAL; j++){
      for(k=0; k<NMESH_Z_LOCAL; k++){
	
	id = k + NMESH_Z_LOCAL*j;
	
	set_position(i, j, k, &hd_param->angle[ipix],
		     &ray[id], this_run);
      }
    }
#else //!__USE_ATOMIC__
    
#pragma omp parallel for schedule(auto) private(i,j,k,id)
    for(j=0; j<NMESH_Y_LOCAL; j=j+2){
      for(k=0; k<NMESH_Z_LOCAL; k=k+2){
    
	id = (k>>1) + (NMESH_Z_LOCAL>>1)*(j>>1);
	
	set_position(i, j, k, &hd_param->angle[ipix],
		     &ray[id + 0*begin_address], this_run);
	set_position(i, j, k+1, &hd_param->angle[ipix], 
		     &ray[id + 1*begin_address], this_run);
	set_position(i, j+1, k, &hd_param->angle[ipix], 
		     &ray[id + 2*begin_address], this_run);
	set_position(i, j+1, k+1, &hd_param->angle[ipix], 
		     &ray[id + 3*begin_address], this_run);
      }
    }

#endif //__USE_ATOMIC__

    break;
    
  case(4):
  case(5):    

    begin_address = NMESH_ZX_LOCAL/RAY_GROUP_NUM;
    
    if(hd_param->angle[ipix].base_id == 4)  j = 0; 
    else                                    j = NMESH_Y_LOCAL - 1;
    
#ifdef __USE_ATOMIC__
#pragma omp parallel for schedule(auto) private(i,j,k,id)
      for(k=0; k<NMESH_Z_LOCAL; k++){
	for(i=0; i<NMESH_X_LOCAL; i++){
	  
	  id = i + NMESH_X_LOCAL*k;
	  
	  set_position(i, j, k, &hd_param->angle[ipix],
		       &ray[id], this_run);
	}
      }
      
#else //!__USE_ATOMIC__
      
#pragma omp parallel for schedule(auto) private(i,j,k,id)
      for(k=0; k<NMESH_Z_LOCAL; k=k+2){
	for(i=0; i<NMESH_X_LOCAL; i=i+2){
	  
	  id = (i>>1) + (NMESH_X_LOCAL>>1)*(k>>1);
	  
	  set_position(i, j, k,     &hd_param->angle[ipix],
		       &ray[id + 0*begin_address], this_run);
	  set_position(i+1, j, k,   &hd_param->angle[ipix],
		       &ray[id + 1*begin_address], this_run);
	  set_position(i, j, k+1,   &hd_param->angle[ipix], 
		       &ray[id + 2*begin_address], this_run);
	  set_position(i+1, j, k+1, &hd_param->angle[ipix], 
		       &ray[id + 3*begin_address], this_run);
	}
      }
#endif //__USE_ATOMIC__
      
      break;
  }
}


inline void set_position(int ix_cur, int iy_cur, int iz_cur,
			 struct angle_info *ang, struct ray_info *ray,
			 struct run_param *this_run)
{
  double start_x, start_y, start_z;

  //start point is center of grid.  offset is random value.
  start_x = this_run->xmin_local + (double)(ix_cur+0.521212345)*this_run->delta_x; //offset mesh center+0.21212345
  start_y = this_run->ymin_local + (double)(iy_cur+0.508512345)*this_run->delta_y; //offset mesh center+0.08512345
  start_z = this_run->zmin_local + (double)(iz_cur+0.468287655)*this_run->delta_z; //offset mesh center-0.31712345

  switch(ang->base_id){
  case(0) : start_z = this_run->zmin;  break;
  case(1) : start_z = this_run->zmax;  break; 
  case(2) : start_x = this_run->xmin;  break;
  case(3) : start_x = this_run->xmax;  break; 
  case(4) : start_y = this_run->ymin;  break;
  case(5) : start_y = this_run->ymax;  break; 
  }
  
  ray->x = start_x;
  ray->y = start_y;
  ray->z = start_z;
  
  ray->I_inHI   = 0.0e0;
#ifdef __HELIUM__
  ray->I_inHeI  = 0.0e0;
  ray->I_inHeII = 0.0e0;
#endif
}


