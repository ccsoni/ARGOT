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

extern float timing(struct tms, struct tms);
extern float wallclock_timing(struct timeval, struct timeval);

#define WAVE_DEPTH ( (NNODE_X)+(NNODE_Y)+(NNODE_Z)-2 ) 

#ifndef __USE_GPU__
void ray_tracing_loop_st(struct ray_face*, struct ray_face*, struct dp_mpi_param*, MPI_Win*,
			 struct fluid_mesh*, struct host_diffuse_param*, struct run_param*);
#else
void ray_tracing_loop_mt(struct ray_face*, struct ray_face*, struct ray_face*, struct dp_mpi_param*,
			 MPI_Win**, struct cuda_mem_space*, struct cuda_diffuse_param*,
			 struct host_diffuse_param*, struct run_param*, struct cuda_param*);
#endif

inline void set_target_source_rank(int*, int*, int, struct run_param*);
inline int  calc_this_rank_depth(int, struct run_param*);
inline void alloc_ray_face(struct ray_face*, struct ray_face*, MPI_Info);
inline void free_ray_face(struct ray_face*, struct ray_face*);

#ifdef __USE_GPU__
void calc_diffuse_photon_radiation(struct fluid_mesh *mesh, 
				   struct run_param *this_run, 
				   struct cuda_mem_space *cuda_mem,
				   struct cuda_param *this_cuda,
				   struct host_diffuse_param *hd_param,
				   struct cuda_diffuse_param *cd_param)
#else /*!__USE_GPU__*/
void calc_diffuse_photon_radiation(struct fluid_mesh *mesh, 
				   struct run_param *this_run, 
				   struct host_diffuse_param *hd_param)
#endif /*__USE_GPU__*/
{
#if 0
#ifdef __ARGOT_PROFILE__
  struct timeval dp_start_tv, dp_end_tv;
  struct tms dp_start_tms, dp_end_tms;
  times(&dp_start_tms);
  gettimeofday(&dp_start_tv, NULL);
#endif /*__ARGOT_PROFILE__*/
#endif

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

  setup_ray_face_dev(start_ray_dev, this_cuda);

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
  ray_tracing_loop_st(&recv_ray, &start_ray, &this_dp_mpi, win_mwf,
		      mesh, hd_param, this_run);
  calc_GH_sum(mesh, hd_param->rmesh);
#else //__USE_GPU__
  calc_rmesh_data(cuda_mem, cd_param, this_cuda);
  ray_tracing_loop_mt(recv_ray, start_ray, start_ray_dev, &this_dp_mpi, win_mwf,
		     cuda_mem, cd_param, hd_param, this_run, this_cuda);
  calc_GH_sum(cuda_mem, cd_param, this_cuda);
  
#endif //__USE_GPU__  


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

  finalize_ray_face_dev(start_ray_dev, this_cuda);
#endif

#if 0
#ifdef __ARGOT_PROFILE__
  times(&dp_end_tms);
  gettimeofday(&dp_end_tv, NULL);
  fprintf(this_run->proc_file,
	  "# calc_diffuse_photon_RT : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n"
	  ,timing(dp_start_tms, dp_end_tms)
	  ,wallclock_timing(dp_start_tv, dp_end_tv));
#endif /*__ARGOT_PROFILE__*/
#endif
}

#ifndef __USE_GPU__
/*single thread in MPI , CPU only*/
void ray_tracing_loop_st(struct ray_face *recv_ray, struct ray_face *start_ray,
			 struct dp_mpi_param *this_dp_mpi, MPI_Win *win_mwf,
			 struct fluid_mesh *mesh, struct host_diffuse_param *hd_param, 
			 struct run_param *this_run)
{
  int cid;
  
  for(cid=0; cid<8; cid++) {
   
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
	  
	  if(angle < 0 || angle >= cid_max) continue;
	  	  
	  zero_set_rmesh(hd_param->rmesh);

	  set_ray_start_position(start_ray, true_ipix, hd_param, this_run);

	  ray_tracing(true_ipix, start_ray, hd_param, this_run);

	  calc_GH_tot(hd_param->rmesh, hd_param->step_fact);
	  
	}// [this_depth == wn]

	mpi_win_fence( MPI_MODE_NOPRECEDE, win_mwf);
	
	if((this_depth==wn)&&((angle>=0)||(angle<cid_max))) 
	  mpi_put_to_target(start_ray, target_rank, 
			    win_mwf, this_dp_mpi);
	
	mpi_win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED , win_mwf);
	
      } // wn loop
      
      for(wn=0; wn<WAVE_DEPTH; wn++) {
	long angle=-1;	

      	if(this_depth == wn+1) {
      	  angle = corner_ipix - wn;
	  
      	  if(angle < 0 || angle>=cid_max) continue;
	
	  mpi_win_recv(start_ray, recv_ray, source_rank);   

      	} //[this_depth == wn+1]
      }//wn loop
    }//ipix loop
  }//cid loop

}

#else //__USE_GPU__

/*multi thread in MPI with GPU*/
void ray_tracing_loop_mt(struct ray_face *recv_ray, struct ray_face *start_ray, struct ray_face *start_ray_dev,
			 struct dp_mpi_param *this_dp_mpi, MPI_Win **win_mwf,
			 struct cuda_mem_space *cuda_mem, struct cuda_diffuse_param *cd_param,
			 struct host_diffuse_param *hd_param, 
			 struct run_param *this_run, struct cuda_param *this_cuda)
{
  int offset = 8/this_cuda->num_cuda_dev;
  int this_depth[NMAX_CUDA_DEV];
  
  int target_rank[NMAX_CUDA_DEV][3];  // 0:xy, 1:yz, 2:zx
  int source_rank[NMAX_CUDA_DEV][3];  // 0:xy, 1:yz, 2:zx

  int cid_max[NMAX_CUDA_DEV];
  int wn[NMAX_CUDA_DEV];
  long angle[NMAX_CUDA_DEV];

  omp_set_nested(1);                       ///opemmp nest flag on

  for(int omp_id=0; omp_id<offset; omp_id++) {

#pragma omp parallel num_threads( this_cuda->num_cuda_dev )
    { 

      int did = omp_get_thread_num();
      int cid = offset*did + omp_id;
      
      this_depth[did] = calc_this_rank_depth(cid, this_run);
      
      set_target_source_rank(target_rank[did], source_rank[did],
			     cid, this_run);

      long offset_ipix=0;
      int o_ip;
      for(o_ip=0; o_ip<cid; o_ip++) offset_ipix += hd_param->corner_id_num[o_ip]; 
      
      long corner_ipix, ipix;
      cid_max[did] = hd_param->corner_id_num[cid];
      int ipix_loop_size = cid_max[did] + (WAVE_DEPTH-1);
      for(corner_ipix=0; corner_ipix<ipix_loop_size; corner_ipix++) {
	
	ipix = offset_ipix + corner_ipix;

	for(wn[did]=0; wn[did]<WAVE_DEPTH; wn[did]++) {
	  angle[did] = -1;	
	  long true_ipix;

	  if(this_depth[did] == wn[did]) {
	    angle[did] = corner_ipix - wn[did];
	    true_ipix = ipix - wn[did];
	    
	    if(angle[did] < 0 || angle[did] >= cid_max[did]) continue;

	    zero_set_rmesh(cd_param[did].rmesh, this_cuda->strm[did], did);

	    set_ray_start_position(&start_ray[did], true_ipix, hd_param, this_run);

	    send_ray_face(&start_ray[did], &start_ray_dev[did], 
			  this_cuda->diffuse_strm[did], did);  /// h2d

            ray_tracing(true_ipix, &start_ray_dev[did],
			&cuda_mem[did], &cd_param[did], 
			hd_param, this_cuda->strm[did], did);

	    calc_GH_tot(&cuda_mem[did], &cd_param[did], this_cuda->strm[did], did);

	    send_ray_face(&start_ray_dev[did], &start_ray[did], 
			  this_cuda->diffuse_strm[did], did);  /// d2h

	  }// [this_depth == wn]

#pragma omp barrier
#pragma omp single
	  {
	    
	    for(int sid=0; sid<this_cuda->num_cuda_dev; sid++) {

	      mpi_win_fence( MPI_MODE_NOPRECEDE, win_mwf[sid]);
	      
	      if((this_depth[sid]==wn[sid])&&
		 ((angle[sid]>=0)||(angle[sid]<cid_max[sid])))
		mpi_put_to_target(&start_ray[sid], target_rank[sid],
				  win_mwf[sid], this_dp_mpi);

	      mpi_win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED, win_mwf[sid]);

	    } //single loop	    
	  } // omp single
	} // wn loop

	for(wn[did]=0; wn[did]<WAVE_DEPTH; wn[did]++) {
	  angle[did] = -1; 
	  
	  if(this_depth[did] == wn[did] + 1) {
	    angle[did] = corner_ipix - wn[did];
	    
	    if(angle[did] < 0 || angle[did] >= cid_max[did]) continue;
	    
	    mpi_win_recv(&start_ray[did], &recv_ray[did], source_rank[did]);

	  } //[this_depth == wn+1]
	}//wn loop
      }//ipix loop
    } // omp parallel
  }//omp_id loop

  omp_set_nested(0);                       ///opemmp nest flag off:default  

}
#endif //__USE_GPU__




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
  /* recv_ray->xy = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_MAX_FACE); */
  /* assert( recv_ray->xy ); */
  /* recv_ray->yz = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_MAX_FACE); */
  /* assert( recv_ray->yz ); */
  /* recv_ray->zx = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_MAX_FACE); */
  /* assert( recv_ray->zx ); */

  MPI_Alloc_mem(sizeof(struct ray_info)*NMESH_MAX_FACE , info, &recv_ray->xy );
  MPI_Alloc_mem(sizeof(struct ray_info)*NMESH_MAX_FACE , info, &recv_ray->yz );
  MPI_Alloc_mem(sizeof(struct ray_info)*NMESH_MAX_FACE , info, &recv_ray->zx );

  start_ray->xy = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_MAX_FACE);
  assert( start_ray->xy );
  start_ray->yz = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_MAX_FACE);
  assert( start_ray->yz );
  start_ray->zx = (struct ray_info*)malloc(sizeof(struct ray_info)*NMESH_MAX_FACE);
  assert( start_ray->zx );
}


inline void free_ray_face(struct ray_face *recv_ray, struct ray_face *start_ray)
{
  /* free( recv_ray->xy ); */
  /* free( recv_ray->yz ); */
  /* free( recv_ray->zx ); */

  MPI_Free_mem( recv_ray->xy );
  MPI_Free_mem( recv_ray->yz );
  MPI_Free_mem( recv_ray->zx );

  free( start_ray->xy );
  free( start_ray->yz );
  free( start_ray->zx );
}

