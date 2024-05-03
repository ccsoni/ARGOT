#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <mpi.h>

#include "diffuse_photon.h"
#include "diffuse_photon_mpi.h"
#ifdef __USE_GPU__
#include "cuda_mem_space.h"
#endif

#include "diffuse_prototype.h"


#include <sys/time.h>
#include <sys/times.h>
extern float wallclock_timing(struct timeval, struct timeval);

#define WAVE_DEPTH ( (NNODE_X)+(NNODE_Y)+(NNODE_Z)-2 ) 


inline void set_target_source_rank(int*, int*, int, struct run_param*);
inline int calc_this_rank_depth(int, struct run_param*);
inline void alloc_ray_face(struct ray_face*, struct ray_face*, MPI_Info);
inline void free_ray_face(struct ray_face*, struct ray_face*);

void calc_diffuse_photon_radiation(struct fluid_mesh *mesh, 
				   struct run_param *this_run, 
#ifdef __USE_GPU__
				   struct cuda_mem_space *cuda_mem,
				   struct cuda_param *this_cuda,
#endif //__USE_GPU__
				   struct host_diffuse_param *hd_param
#ifdef __USE_GPU__
				   ,struct cuda_diffuse_param *cd_param
#endif //__USE_GPU__
				   )
{
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "no_locks", "true");


#ifndef __USE_GPU__
  /*** set host memory ***/
  static struct ray_face recv_ray, start_ray;
  alloc_ray_face(&recv_ray, &start_ray, info);


  /*** set device memory start***/
#else // __USE_GPU__
  
  static struct ray_face recv_ray[NMAX_CUDA_DEV], start_ray[NMAX_CUDA_DEV];
  for(int idev=0; idev<this_cuda->num_cuda_dev; idev++)
    alloc_ray_face(&recv_ray[idev], &start_ray[idev], info);

  static struct ray_face start_ray_dev[NMAX_CUDA_DEV];
  static cudaStream_t strm[NMAX_CUDA_DEV];     // gcc,icc's Macro : cudaStream_t -> int64_t
  
  setup_ray_face_dev(start_ray_dev, this_cuda, strm);

#endif //__USE_GPU__
  /*** set device memory end***/

  /*** set mpi start***/
  static struct dp_mpi_param this_dp_mpi;
  set_dp_mpi_type(&this_dp_mpi);


#ifndef __USE_GPU__
  MPI_Win win_mwf[3];     //[0]:xy [1]:yz [2]:zx 
  set_mpi_window(&recv_ray, win_mwf, info);
#else //__USE_GPU__
  MPI_Win *win_mwf[NMAX_CUDA_DEV];     
  for(int idev=0; idev<this_cuda->num_cuda_dev; idev++) {
    win_mwf[idev] = (MPI_Win*)malloc(sizeof(MPI_Win)*3); //[0]:xy [1]:yz [2]:zx 
    set_mpi_window(&recv_ray[idev], win_mwf[idev], info);
  }
#endif
  /*** set mpi end***/
  

#ifndef __USE_GPU__
  calc_rmesh_data(mesh, hd_param->rmesh, this_run);
#else //__USE_GPU__
  calc_rmesh_data(cuda_mem, cd_param, this_cuda, strm);
#endif //__USE_GPU__  

  int cid;

#ifdef __USE_GPU__
  int chunk = 8/this_cuda->num_cuda_dev;
  omp_set_nested(1);                       ///opemmp nest flag on
#pragma omp parallel for schedule(static,chunk) num_threads( this_cuda->num_cuda_dev ) ordered
#endif //__USE_GPU__
  for(cid=0; cid<8; cid++) {
   
#ifdef __USE_GPU__
    int device_id;
    //   device_id = omp_get_thread_num();
    if(this_cuda->num_cuda_dev==1) {
      device_id = 0;
    } else if(this_cuda->num_cuda_dev==2) {
      switch(cid) {
      case(0): case(1): case(2): case(3):
    	device_id = 0;
    	break;
      case(4): case(5): case(6): case(7):
    	device_id = 1;
    	break;
      }
    } else if(this_cuda->num_cuda_dev==4) {
      switch(cid) {
      case(0): case(1):
    	device_id = 0;
    	break;
      case(2): case(3):
    	device_id = 1;
    	break;
      case(4): case(5):
    	device_id = 2;
    	break;
      case(6): case(7):
    	device_id = 3;
    	break;
      }
    }
#endif //__USE_GPU__


    int this_depth = calc_this_rank_depth(cid, this_run);

    int target_rank[3];  // 0:xy, 1:yz, 2:zx
    int source_rank[3];  // 0:xy, 1:yz, 2:zx

    set_target_source_rank(target_rank, source_rank,
			   cid, this_run);
        
    long offset_ipix=0;
    int o_ip;
    for(o_ip=0; o_ip<cid; o_ip++) offset_ipix += hd_param->corner_id_num[o_ip]; 
    
    long corner_ipix, ipix;
    int cid_max = hd_param->corner_id_num[cid];
    int ipix_loop_size = cid_max + (WAVE_DEPTH-1);
    for(corner_ipix=0; corner_ipix<ipix_loop_size; corner_ipix++) {
      
      ipix = offset_ipix + corner_ipix;
   
      int wn;

      for(wn=0; wn<WAVE_DEPTH; wn++) {
	long angle=-1;	
	long true_ipix;

	if(this_depth == wn) {
	  angle = corner_ipix - wn;
	  true_ipix = ipix - wn;
	  
	  if(angle < 0 || angle>=cid_max) continue;
	  	  
#ifndef __USE_GPU__
	  set_ray_start_position(&start_ray, true_ipix, hd_param, this_run);
	  
	  zero_set_rmesh(hd_param->rmesh);

	  ray_tracing(true_ipix, &start_ray, mesh, hd_param, this_run);

	  calc_GH_tot(mesh, hd_param->rmesh, &hd_param->freq , this_run);

#else
	  set_ray_start_position(&start_ray[device_id], true_ipix, hd_param, this_run);
	  
	  send_ray_face(&start_ray[device_id], &start_ray_dev[device_id], 
			strm[device_id], device_id);  /// h2d
	  
	  zero_set_rmesh(cd_param[device_id].rmesh, strm[device_id], device_id);

	  ray_tracing(true_ipix, &start_ray_dev[device_id],
		      &cuda_mem[device_id], &cd_param[device_id], 
		      hd_param, strm[device_id], device_id);

	  send_ray_face(&start_ray_dev[device_id], &start_ray[device_id], 
			strm[device_id], device_id);  /// d2h
	  
	  calc_GH_tot(&cuda_mem[device_id], &cd_param[device_id], strm[device_id], device_id);
	  
#endif //__USE_GPU__
	  
	}// [this_depth == wn]

	
#ifndef __USE_GPU__  ////RMA operation
	mpi_win_fence( MPI_MODE_NOPRECEDE, win_mwf);
	
	if((this_depth==wn)&&((angle>=0)||(angle<cid_max))) 
	  mpi_put_to_target(&start_ray, target_rank, 
			    win_mwf, &this_dp_mpi);
	
	mpi_win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED, win_mwf);
#else
	  
	mpi_win_fence( MPI_MODE_NOPRECEDE, win_mwf[device_id]);
	
	if((this_depth==wn)&&((angle>=0)||(angle<cid_max)))
	  mpi_put_to_target(&start_ray[device_id], target_rank,
			    win_mwf[device_id], &this_dp_mpi);
	
	mpi_win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED, win_mwf[device_id]);

	/* } // omp ordered */
	
#endif
	
      } // wn loop
      
      for(wn=0; wn<WAVE_DEPTH; wn++) {
	long angle=-1;	

      	if(this_depth == wn+1) {
      	  angle = corner_ipix - wn;
	  
      	  if(angle < 0 || angle>=cid_max) continue;
	
#ifndef __USE_GPU__   
	  mpi_win_recv(&start_ray, &recv_ray, source_rank);   
#else ///__USE_GPU__
	  mpi_win_recv(&start_ray[device_id], &recv_ray[device_id], source_rank);
#endif

      	} //[this_depth == wn+1]
      }//wn loop
    }//ipix loop
  }//cid loop
 

#ifndef __USE_GPU__
  calc_GH_sum(mesh, hd_param->rmesh, this_run);
#else
  omp_set_nested(0);                       ///opemmp nest flag off:default  
  calc_GH_sum(cuda_mem, cd_param, this_cuda, strm);
#endif ///__USE_GPU__
 

  MPI_Info_free(&info);
  MPI_Type_free(&this_dp_mpi.ray_info_type);


#ifndef __USE_GPU__
  free_mpi_window(win_mwf);
  free_ray_face(&recv_ray, &start_ray);
#else //__USE_GPU__
  for(int idev=0; idev<this_cuda->num_cuda_dev; idev++){
    free_mpi_window(win_mwf[idev]);
    free(win_mwf[idev]);
    free_ray_face(&recv_ray[idev], &start_ray[idev]);
  }

  finalize_ray_face_dev(start_ray_dev, this_cuda, strm);
#endif

}



inline void set_target_source_rank(int *target_rank, int *source_rank,
				   int corner_id, struct run_param *this_run)
{
  /// [0]:xy, [1]:yz:, [2]:zx

  switch(corner_id) {
  case(0): 
    target_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z+1);
    target_rank[1] = mpi_rank(this_run->rank_x+1, this_run->rank_y,   this_run->rank_z);
    target_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y+1, this_run->rank_z);
    source_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z-1);
    source_rank[1] = mpi_rank(this_run->rank_x-1, this_run->rank_y,   this_run->rank_z);
    source_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y-1, this_run->rank_z);
    break;
   
  case(1): 
    target_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z-1);
    target_rank[1] = mpi_rank(this_run->rank_x+1, this_run->rank_y,   this_run->rank_z);
    target_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y+1, this_run->rank_z);
    source_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z+1);
    source_rank[1] = mpi_rank(this_run->rank_x-1, this_run->rank_y,   this_run->rank_z);
    source_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y-1, this_run->rank_z);
    break;

  case(2): 
    target_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z+1);
    target_rank[1] = mpi_rank(this_run->rank_x+1, this_run->rank_y,   this_run->rank_z);
    target_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y-1, this_run->rank_z);
    source_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z-1);
    source_rank[1] = mpi_rank(this_run->rank_x-1, this_run->rank_y,   this_run->rank_z);
    source_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y+1, this_run->rank_z);
    break;
    
  case(3): 
    target_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z-1);
    target_rank[1] = mpi_rank(this_run->rank_x+1, this_run->rank_y,   this_run->rank_z);
    target_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y-1, this_run->rank_z);
    source_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z+1);
    source_rank[1] = mpi_rank(this_run->rank_x-1, this_run->rank_y,   this_run->rank_z);
    source_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y+1, this_run->rank_z);
    break;

  case(4): 

    target_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z+1);
    target_rank[1] = mpi_rank(this_run->rank_x-1, this_run->rank_y,   this_run->rank_z);
    target_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y+1, this_run->rank_z);
    source_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z-1);
    source_rank[1] = mpi_rank(this_run->rank_x+1, this_run->rank_y,   this_run->rank_z);
    source_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y-1, this_run->rank_z);
    break;

  case(5): 

    target_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z-1);
    target_rank[1] = mpi_rank(this_run->rank_x-1, this_run->rank_y,   this_run->rank_z);
    target_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y+1, this_run->rank_z);
    source_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z+1);
    source_rank[1] = mpi_rank(this_run->rank_x+1, this_run->rank_y,   this_run->rank_z);
    source_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y-1, this_run->rank_z);
    break;

  case(6): 
    target_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z+1);
    target_rank[1] = mpi_rank(this_run->rank_x-1, this_run->rank_y,   this_run->rank_z);
    target_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y-1, this_run->rank_z);
    source_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z-1);
    source_rank[1] = mpi_rank(this_run->rank_x+1, this_run->rank_y,   this_run->rank_z);
    source_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y+1, this_run->rank_z);
    break;

  case(7): 
    target_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z-1);
    target_rank[1] = mpi_rank(this_run->rank_x-1, this_run->rank_y,   this_run->rank_z);
    target_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y-1, this_run->rank_z);
    source_rank[0] = mpi_rank(this_run->rank_x,   this_run->rank_y,   this_run->rank_z+1);
    source_rank[1] = mpi_rank(this_run->rank_x+1, this_run->rank_y,   this_run->rank_z);
    source_rank[2] = mpi_rank(this_run->rank_x,   this_run->rank_y+1, this_run->rank_z);
    break;
  }
}




inline int calc_this_rank_depth(int corner_id, struct run_param *this_run)
{
  switch(corner_id) {
  case(0):
    return (this_run->rank_x 
	    +this_run->rank_y 
	    +this_run->rank_z);
    break;

  case(1):
    return (this_run->rank_x
	    +this_run->rank_y
	    +(this_run->nnode_z-1)-this_run->rank_z);
    break;
    
  case(2):
    return (this_run->rank_x
	    +(this_run->nnode_y-1)-this_run->rank_y
	    +this_run->rank_z);
    break;
    
  case(3):
    return (this_run->rank_x
	    +(this_run->nnode_y-1)-this_run->rank_y
	    +(this_run->nnode_z-1)-this_run->rank_z);
    break;
    
  case(4):
    return ((this_run->nnode_x-1)-this_run->rank_x 
	    +this_run->rank_y
	    +this_run->rank_z);
    break;
    
  case(5):
    return ((this_run->nnode_x-1)-this_run->rank_x
	    +this_run->rank_y
	    +(this_run->nnode_z-1)-this_run->rank_z);
    break;
    
  case(6):
    return ((this_run->nnode_x-1)-this_run->rank_x
	    +(this_run->nnode_y-1)-this_run->rank_y
	    +this_run->rank_z);
    break;

  case(7):
    return ((this_run->nnode_x-1)-this_run->rank_x
	    +(this_run->nnode_y-1)-this_run->rank_y
	    +(this_run->nnode_z-1)-this_run->rank_z);
    break;
  }
  return -1;
}



inline void alloc_ray_face(struct ray_face *recv_ray, struct ray_face *start_ray, MPI_Info info)
{
  /* recv_ray->xy = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_XY_LOCAL); */
  /* assert( recv_ray->xy ); */
  /* recv_ray->yz = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_YZ_LOCAL); */
  /* assert( recv_ray->yz ); */
  /* recv_ray->zx = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_ZX_LOCAL); */
  /* assert( recv_ray->zx ); */

  MPI_Alloc_mem(sizeof(struct ray_info)*NMESH_XY_LOCAL , info, &recv_ray->xy );
  MPI_Alloc_mem(sizeof(struct ray_info)*NMESH_YZ_LOCAL , info, &recv_ray->yz );
  MPI_Alloc_mem(sizeof(struct ray_info)*NMESH_ZX_LOCAL , info, &recv_ray->zx );

  start_ray->xy = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_XY_LOCAL);
  assert( start_ray->xy );
  start_ray->yz = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_YZ_LOCAL);
  assert( start_ray->yz );
  start_ray->zx = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_ZX_LOCAL);
  assert( start_ray->zx );
}


inline void free_ray_face(struct ray_face *recv_ray, struct ray_face *start_ray)
{
  /* free( recv_ray->xy );  */
  /* free( recv_ray->yz ); */
  /* free( recv_ray->zx ); */

  MPI_Free_mem( recv_ray->xy );
  MPI_Free_mem( recv_ray->yz );
  MPI_Free_mem( recv_ray->zx );

  free( start_ray->xy );
  free( start_ray->yz );
  free( start_ray->zx );
}
