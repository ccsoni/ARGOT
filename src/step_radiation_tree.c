#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>

#include "constants.h"
#include "run_param.h"
#include "mpi_param.h"
#include "omp_param.h"
#include "radiation.h"
#include "tree_clist.h"

#include "prototype.h"

#ifdef __DIFFUSE_RADIATION__
#include "diffuse_photon.h"
#endif

uint64_t set_mesh_index_range(int imesh_start, int imesh_end, uint64_t *nray_to)
{
  int imesh;
  uint64_t nray_sum;

  nray_sum=0;
  for(imesh=imesh_start;imesh<imesh_end;imesh++) {
    nray_sum += nray_to[imesh];
  }

  assert(nray_sum <= MAX_NRAY_PER_TARGET);

  return nray_sum;
}


float calc_mem_size_for_radiation(struct radiation_src *src, 
				  struct run_param *this_run,
				  int nmesh_per_loop)
/* compute the total mem size for radiation transfer in units of GByte. */
{
  struct light_ray *ray;
  struct ray_segment *seg;

  struct clist_t *clist;
  uint64_t *key;
  int *index;
  int nclist;

  uint64_t *nray_to;

  struct omp_param this_omp;
  
  /* setup the parameters for the tree structure */
  this_run->theta_crit = ARGOT_THETA_CRIT;
  init_tree_param(60, this_run, &nclist);

  /* allocate the tree structure */
  clist = (struct clist_t *)malloc(sizeof(struct clist_t)*nclist);
  key = (uint64_t *)malloc(sizeof(uint64_t)*this_run->nsrc);
  index = (int *)malloc(sizeof(int)*this_run->nsrc);

  /* array of the number of light ray to a mesh point */
  nray_to = (uint64_t *)malloc(sizeof(uint64_t)*NMESH_LOCAL);

  construct_tree(clist, key, index, src, this_run);

  /* count the number of light rays targeted to each mesh */
#pragma omp parallel for schedule(auto)
  for(int imesh=0;imesh<NMESH_LOCAL;imesh++) {
    int ix, iy, iz;

    ix = imesh/(NMESH_Y_LOCAL*NMESH_Z_LOCAL);
    iy = (imesh-ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL)/NMESH_Z_LOCAL;
    iz = imesh - ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL - iy*NMESH_Z_LOCAL;

    nray_to[imesh] = count_ray_to(ix, iy, iz, clist, index, this_run);
    
    assert(nray_to[imesh] <= MAX_NRAY_PER_TARGET);
  }

  /* measure the maximum number of light rays for all loops of target meshes */
  uint64_t nray_sum_max;
  nray_sum_max=0;

#pragma omp parallel for schedule(auto) reduction(max:nray_sum_max)
  for(int imesh=0;imesh<NMESH_LOCAL;imesh+=nmesh_per_loop) {
    uint64_t nray_sum;
    int im_start, im_end;

    nray_sum=0;

    im_start = imesh;
    im_end = im_start + nmesh_per_loop;
    for(int im=im_start;im<im_end;im++) {
      nray_sum += nray_to[im];
    }

    nray_sum_max = MAX(nray_sum_max, nray_sum);
  }

  /* allocate an array of light_ray structures */
  ray = (struct light_ray *)malloc(sizeof(struct light_ray)*nray_sum_max);
  fprintf(this_run->proc_file, "# nray_sum_max = %llu\n", nray_sum_max);
  fprintf(this_run->proc_file, 
	  "# data size for the light rays : %14.6e [MByte] \n", 
	  (float)(sizeof(struct light_ray)*nray_sum_max)/(float)(1<<20));

  float mem_size_for_light_rays 
    = (float)sizeof(struct light_ray)*(float)nray_sum_max;
  mem_size_for_light_rays /= CUBE(1024.0);

  /* loop over the local target mesh */
  float max_mem_size_for_segments=0.0;
  float mem_size_for_segments;
  int loop_id=0;
  for(int imesh=0;imesh<NMESH_LOCAL;imesh+=nmesh_per_loop) {
    /* start and end indices */
    int im_start, im_end;
      
    im_start = imesh;
    im_end   = imesh+nmesh_per_loop;
	
    fprintf(this_run->proc_file,
	    "# ========================================\n");
    fprintf(this_run->proc_file,
	    "# loop ID : %d\n", loop_id++);
    fprintf(this_run->proc_file,
	    "# loop over target meshes from %d to %d of %d meshes\n",
	    im_start, im_end, NMESH_LOCAL);
    
    /* calculate the index range of the target meshes */
    this_run->nray = set_mesh_index_range(im_start, im_end, nray_to);

    /* setting up the light rays using tree structure */
    setup_light_ray_range(im_start, im_end, ray, 1, clist, src, 
			  index, this_run);

    /* counting the number of ray-segments of light rays */

    omp_init_nest_lock(&(this_omp.omp_lock));    /*** dummy ***/
    count_ray_segment(ray, this_run, &this_omp);
    omp_destroy_nest_lock(&(this_omp.omp_lock));  /*** dummy ***/

    
#ifdef __ARGOT_PROFILE__
    fprintf(this_run->proc_file,
	    "# number of ray segments :: %llu\n", this_run->nseg);
    fprintf(this_run->proc_file,
	    "# data size for ray segments :: %14.6e MBytes\n",
	    (float)(sizeof(struct ray_segment)*this_run->nseg)/(float)(1<<20));
#endif /* __ARGOT_PROFILE__ */
    mem_size_for_segments 
      = (float)sizeof(struct ray_segment)*(float)this_run->nseg;
    max_mem_size_for_segments = MAX(max_mem_size_for_segments,
				    mem_size_for_segments);

  }
	  
  max_mem_size_for_segments /= CUBE(1024.0);

  float  total_mem_size = mem_size_for_light_rays + max_mem_size_for_segments;

  free(clist);
  free(key);
  free(index);
  free(nray_to);
  free(ray);

  fprintf(this_run->proc_file, 
	  "# total memory size for radiation transfer : %14.6e [GByte]\n",
	  total_mem_size);

  return total_mem_size;
}

int get_optimal_nmesh_per_loop(struct radiation_src *src, 
			       struct run_param *this_run)
{
  int nmesh_per_loop, not_allowed, not_allowed_reduced;
  float mem_size;
  
  nmesh_per_loop = this_run->nmesh_per_loop;

  do {

    mem_size = calc_mem_size_for_radiation(src, this_run, nmesh_per_loop);
    not_allowed = (mem_size > ALLOWED_MEM_SIZE_IN_GB ? 1 : 0);
    MPI_Allreduce(&not_allowed, &not_allowed_reduced, 1, 
		  MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    if(not_allowed_reduced == 0) break;
    nmesh_per_loop /= 2;

  }while(not_allowed_reduced);

  int nmesh_per_loop_reduced;
  MPI_Allreduce(&nmesh_per_loop, &nmesh_per_loop_reduced, 1, 
		MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  return nmesh_per_loop_reduced;
}

void set_optimal_nmesh_per_loop(struct radiation_src *src, 
				struct run_param *this_run)
{
  this_run->nmesh_per_loop = get_optimal_nmesh_per_loop(src, this_run);
  fprintf(this_run->proc_file, "# optimal nmesh_per_loop = %d\n",
	  this_run->nmesh_per_loop);
  fflush(this_run->proc_file);
}


void calc_ARGOT_part(
#ifdef __USE_GPU_ARGOT__
		     struct cuda_mem_space *cuda_mem,
		     struct cuda_param *this_cuda,
#else
		     struct fluid_mesh *mesh,		     
#endif
		     struct light_ray *ray, struct radiation_src *src,
		     int *index, uint64_t *nray_to, struct clist_t *clist,
		     struct run_param *this_run, struct mpi_param *this_mpi, struct omp_param *this_omp)
{
  struct ray_segment *seg;
#ifdef __USE_GPU_ARGOT__
  struct light_ray_IO *ray_IO;
#endif
  
  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;
  
  /* loop over the local target meshes */
  int loop_id=0;
  for(int imesh=0;imesh<NMESH_LOCAL;imesh+=this_run->nmesh_per_loop) {
    
    start_timing(&start_tv, &start_tms);
    
    /* start and end indices */
    int im_start, im_end;
      
    im_start = imesh;
    im_end   = imesh+this_run->nmesh_per_loop;
    //      im_end   = imesh+NMESH_PER_LOOP;
	
    fprintf(this_run->proc_file,
	    "# ========================================\n");
    fprintf(this_run->proc_file,
	    "# loop ID : %d\n", loop_id++);
    fprintf(this_run->proc_file,
	    "# loop over taget meshes from %d to %d of %d meshes\n",
	    im_start, im_end, NMESH_LOCAL);
    
    /* calculate the index range of the target meshes */
    this_run->nray = set_mesh_index_range(im_start, im_end, nray_to);
    
    /* setting up the light rays using the tree structure */ 
    setup_light_ray_range(im_start, im_end, ray, 1, clist, src, 
			  index, this_run);
      
    /* setting up segments of light rays */
    count_ray_segment(ray, this_run, this_omp);

#ifdef __USE_GPU_ARGOT__
    allocate_pinned_segment(&seg, this_run->nseg);
    allocate_pinned_light_ray_IO(&ray_IO, this_run->nray);
#else
    seg = 
      (struct ray_segment*) malloc(sizeof(struct ray_segment)*this_run->nseg);
#endif
    
#ifdef __ARGOT_PROFILE__
    fprintf(this_run->proc_file,
	    "# number of ray segments :: %llu\n", this_run->nseg);
    fprintf(this_run->proc_file,
	    "# data size for ray segments :: %14.6e MBytes\n",
	    (float)(sizeof(struct ray_segment)*this_run->nseg)/(float)(1<<20));
#endif /* __ARGOT_PROFILE__ */
    
    end_timing(&start_tv, &end_tv, &start_tms, &end_tms,
	       "Initialization within the iteration", this_run);
    
    assign_ray_segment(ray, seg, this_run, this_mpi, this_omp);
    
    /* compute optical depths of the light ray segments */
#ifdef __USE_GPU_ARGOT__
    calc_optical_depth_cuda(seg, cuda_mem, this_cuda, this_run);
#else
    calc_optical_depth(seg, mesh, this_run);
#endif
    
    /* accumulate optical depths  */
    accum_optical_depth(ray, seg, this_run, this_omp);
      
    /* calculate photo-ionization rate */
#ifdef __USE_GPU_ARGOT__
    calc_photoion_rate_cuda(cuda_mem, ray, ray_IO, this_cuda, this_run);
#else
    calc_photoion_rate(mesh, ray, this_run);
#endif
    
#ifdef __USE_GPU_ARGOT__
    deallocate_pinned_segment(seg);
    deallocate_pinned_light_ray_IO(ray_IO);
#else
    free(seg);
#endif
  }
}


#ifdef __DIFFUSE_RADIATION__
void calc_ART_part(
		   struct fluid_mesh *mesh,		     
#ifdef __USE_GPU_ART__
		   struct cuda_mem_space *cuda_mem,
		   struct cuda_param *this_cuda,
		   struct cuda_diffuse_param *cd_param,
#endif
		   struct host_diffuse_param *hd_param,
		   struct run_param *this_run, struct omp_param *this_omp)
{
  struct timeval dp_start_tv, dp_end_tv;
  struct tms dp_start_tms, dp_end_tms;
  
  start_timing(&dp_start_tv, &dp_start_tms);

#ifdef __USE_GPU_ART__
  calc_diffuse_photon_radiation(mesh, this_run, cuda_mem, this_cuda, hd_param, cd_param, this_omp);
#else /* !__USE_GPU_ART__ */
  calc_diffuse_photon_radiation(mesh, this_run, hd_param, this_omp); 
#endif /* __USE_GPU_ART__ */
  
  end_timing(&dp_start_tv, &dp_end_tv, &dp_start_tms, &dp_end_tms,
	     "Diffuse RT", this_run);
}
#endif /* __DIFFUSE_RADIATION__ */ 


void calc_radiation_transfer(struct fluid_mesh *mesh,		     
#ifdef __USE_GPU__
			     struct cuda_mem_space *cuda_mem,
			     struct cuda_param *this_cuda,
#endif
			     struct light_ray *ray, struct radiation_src *src,
			     int *index, uint64_t *nray_to, struct clist_t *clist,
			     struct run_param *this_run, struct mpi_param *this_mpi
#ifdef __DIFFUSE_RADIATION__
#ifdef __USE_GPU__
			     ,struct cuda_diffuse_param *cd_param
#endif
			     ,struct host_diffuse_param *hd_param
#endif
			     )
{
  struct omp_param this_omp; 
  omp_init_nest_lock(&(this_omp.omp_lock));
  
#ifndef __DIFFUSE_RADIATION__
#ifdef __USE_GPU_ARGOT__
  calc_ARGOT_part(cuda_mem, this_cuda, ray, src, index, nray_to, clist, this_run, this_mpi, &this_omp);
#else
  calc_ARGOT_part(mesh, ray, src, index, nray_to, clist, this_run, this_mpi, &this_omp);
#endif 
#endif
  
#ifdef __DIFFUSE_RADIATION__  
#ifdef __USE_GPU__
#ifdef __USE_GPU_ARGOT__
  /* calc on GPU */
  calc_ARGOT_part(cuda_mem, this_cuda, ray, src, index, nray_to, clist, this_run, this_mpi, &this_omp);
  calc_ART_part(mesh, cuda_mem, this_cuda, cd_param, hd_param, this_run, &this_omp);
  
#else //!__USE_GPU_ARGOT__

  int total_threads_num = omp_get_max_threads();
  
  omp_set_nested(1);
  
  fprintf(this_run->proc_file, "# total threads num:%d , ARGOT threads num %d, ART threads num %d\n",
	  total_threads_num, total_threads_num-this_cuda->num_cuda_dev, this_cuda->num_cuda_dev);
  fflush(this_run->proc_file);
  
#pragma omp parallel sections
  { 
#pragma omp section
    {
      omp_set_num_threads(total_threads_num - this_cuda->num_cuda_dev);
      /* calc on CPU */
      calc_ARGOT_part(mesh, ray, src, index, nray_to, clist, this_run, this_mpi, &this_omp);
    }
#pragma omp section
    {
      omp_set_num_threads(this_cuda->num_cuda_dev);
      /* calc on GPU  */
      calc_ART_part(mesh, cuda_mem, this_cuda, cd_param, hd_param, this_run, &this_omp);
    }
  }
  omp_set_nested(0);
  
  /* ART.photoion += ARGOT.photoion on GPU */
  merge_photoion_data(mesh, cuda_mem, this_cuda, this_run);
#endif //__USE_GPU_ARGOT__    
#else //!__USE_GPU__
  calc_ARGOT_part(mesh, ray, src, index, nray_to, clist, this_run, this_mpi, &this_omp);
  calc_ART_part(mesh, hd_param, this_run, &this_omp);
#endif //__USE_GPU__
#endif //__DIFFUSE_RADIATION__

  omp_destroy_nest_lock(&(this_omp.omp_lock));
}


void calc_radiation_transfer_at_first(struct fluid_mesh *mesh,		     
#ifdef __USE_GPU_ARGOT__
				      struct cuda_mem_space *cuda_mem,
				      struct cuda_param *this_cuda,
#endif
				      struct light_ray *ray, struct radiation_src *src,
				      int *index, uint64_t *nray_to, struct clist_t *clist,
				      struct run_param *this_run, struct mpi_param *this_mpi)
{
  /*** dummy ***/
  struct omp_param this_omp; 
  omp_init_nest_lock(&(this_omp.omp_lock));

#ifdef __USE_GPU_ARGOT__
  calc_ARGOT_part(cuda_mem, this_cuda, ray, src, index, nray_to, clist, this_run, this_mpi, &this_omp);
#else
  calc_ARGOT_part(mesh, ray, src, index, nray_to, clist, this_run, this_mpi, &this_omp);
#endif 

  /*** dummy ***/
  omp_destroy_nest_lock(&(this_omp.omp_lock));
}


void calc_photoion_rate_at_first(struct fluid_mesh *mesh,
				 struct radiation_src *src,
#ifdef __USE_GPU__
				 struct cuda_mem_space *cuda_mem,
				 struct cuda_param *this_cuda,
#endif
				 struct run_param *this_run,
				 struct mpi_param *this_mpi
				 )
{
  struct light_ray *ray;
  struct clist_t *clist;
  uint64_t *key;
  int *index;
  int nclist;

  uint64_t *nray_to;

  struct timeval start_init_tv, end_init_tv;
  struct tms start_init_tms, end_init_tms;
  start_timing(&start_init_tv, &start_init_tms);

  /* setup the parameters for the tree structure */
  this_run->theta_crit = ARGOT_THETA_CRIT;
  init_tree_param(60, this_run, &nclist);

  /* allocate the tree structure */
  clist = (struct clist_t *)malloc(sizeof(struct clist_t)*nclist);
  key = (uint64_t *)malloc(sizeof(uint64_t)*this_run->nsrc);
  index = (int *)malloc(sizeof(int)*this_run->nsrc);

  /* array of the number of light ray to a mesh point */
  nray_to = (uint64_t *)malloc(sizeof(uint64_t)*NMESH_LOCAL);

  construct_tree(clist, key, index, src, this_run);
  
  /* count the number of light rays targeted to each mesh */
#pragma omp parallel for schedule(auto)
  for(int imesh=0;imesh<NMESH_LOCAL;imesh++) {
    int ix, iy, iz;

    ix = imesh/(NMESH_Y_LOCAL*NMESH_Z_LOCAL);
    iy = (imesh-ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL)/NMESH_Z_LOCAL;
    iz = imesh - ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL - iy*NMESH_Z_LOCAL;

    nray_to[imesh] = count_ray_to(ix, iy, iz, clist, index, this_run);
    
    assert(nray_to[imesh] <= MAX_NRAY_PER_TARGET);
  }

  /* measure the maximum number of light rays for all loops of target meshes */
  uint64_t nray_sum_max;
  nray_sum_max=0;

#pragma omp parallel for schedule(auto) reduction(max:nray_sum_max)
  for(int imesh=0;imesh<NMESH_LOCAL;imesh+=this_run->nmesh_per_loop) {

    uint64_t nray_sum;
    int im_start, im_end;

    nray_sum=0;

    im_start = imesh;
    im_end = im_start + this_run->nmesh_per_loop;

    for(int im=im_start;im<im_end;im++) {
      nray_sum += nray_to[im];
    }

    nray_sum_max = MAX(nray_sum_max, nray_sum);
  }

  free(key);
  
  /* allocate light_ray structures */
  ray = (struct light_ray *)malloc(sizeof(struct light_ray)*nray_sum_max);
  fprintf(this_run->proc_file, "# nray_sum_max = %llu\n", nray_sum_max);
  fprintf(this_run->proc_file, 
	  "# data size for the light rays : %llu [MByte] \n", 
	  (sizeof(struct light_ray)*nray_sum_max)>>20);

  end_timing(&start_init_tv, &end_init_tv, &start_init_tms, &end_init_tms,
	     "Initialization before the first estimate of photo-Gamma", this_run);

  
#ifdef __USE_GPU__
  zero_out_photoion_rate_cuda(cuda_mem, this_cuda, this_run);
#ifndef __USE_GPU_ARGOT__
  recv_mesh_data(mesh, cuda_mem, this_cuda, this_run);
#endif
#else
  zero_out_photoion_rate(mesh, this_run);
#endif

#ifdef __USE_GPU_ARGOT__
  calc_radiation_transfer_at_first(mesh, cuda_mem, this_cuda, ray, src, index, nray_to, clist, this_run, this_mpi);
#else
  calc_radiation_transfer_at_first(mesh, ray, src, index, nray_to, clist, this_run, this_mpi);
#endif
  

#ifdef __USE_GPU__
#ifndef __USE_GPU_ARGOT__
  send_mesh_data(mesh, cuda_mem, this_cuda, this_run);
#endif
  smooth_photoion_rate(mesh, this_run, cuda_mem, this_cuda, this_mpi);
#else
  smooth_photoion_rate(mesh, this_run, this_mpi);
#endif

  free(clist);
  free(index);
  free(nray_to);
  free(ray);
}

void step_radiation_tree(struct fluid_mesh *mesh, struct radiation_src *src, 
			 struct run_param *this_run, struct mpi_param *this_mpi,
#ifdef __USE_GPU__
			 struct cuda_mem_space *cuda_mem,
			 struct cuda_param *this_cuda,
#endif
#ifdef __DIFFUSE_RADIATION__
			 struct host_diffuse_param *hd_param,
#ifdef __USE_GPU__
			 struct cuda_diffuse_param *cd_param,
#endif /* __USE_GPU__ */
#endif /* __DIFFUSE_RADIATION__ */
			 float  dtime)
{

  struct light_ray *ray;
  struct clist_t *clist;
  uint64_t *key;
  int *index;
  int nclist;

  uint64_t *nray_to;

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

  struct timeval start_init_tv, end_init_tv;
  struct tms start_init_tms, end_init_tms;

#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);

  times(&start_init_tms);
  gettimeofday(&start_init_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

  float max_diff_chem, max_diff_uene;
  float elapsed_time, dtime_min;

  /* setup the parameters for the tree structure */
  this_run->theta_crit = ARGOT_THETA_CRIT;
  init_tree_param(60, this_run, &nclist);

  /* allocate the tree structure */
  clist = (struct clist_t *)malloc(sizeof(struct clist_t)*nclist);
  key = (uint64_t *)malloc(sizeof(uint64_t)*this_run->nsrc);
  index = (int *)malloc(sizeof(int)*this_run->nsrc);

  /* array of the number of light ray to a mesh point */
  nray_to = (uint64_t *)malloc(sizeof(uint64_t)*NMESH_LOCAL);

  construct_tree(clist, key, index, src, this_run);

  uint64_t max_nray_per_target;
  max_nray_per_target=0;
  /* count the number of light rays targeted to each mesh */
#pragma omp parallel for schedule(auto) reduction(max:max_nray_per_target)
  for(int imesh=0;imesh<NMESH_LOCAL;imesh++) {
    int ix, iy, iz;

    ix = imesh/(NMESH_Y_LOCAL*NMESH_Z_LOCAL);
    iy = (imesh-ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL)/NMESH_Z_LOCAL;
    iz = imesh - ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL - iy*NMESH_Z_LOCAL;

    nray_to[imesh] = count_ray_to(ix, iy, iz, clist, index, this_run);

    max_nray_per_target = MAX(max_nray_per_target, nray_to[imesh]);
    
    assert(nray_to[imesh] <= MAX_NRAY_PER_TARGET);
  }
  fprintf(this_run->proc_file, "# max_nray_per_target : %llu\n", 
	  max_nray_per_target);

  /* measure the maximum number of light rays for all loops of target meshes */
  uint64_t nray_sum_max;
  nray_sum_max=0;

#pragma omp parallel for schedule(auto) reduction(max:nray_sum_max)
  for(int imesh=0;imesh<NMESH_LOCAL;imesh+=this_run->nmesh_per_loop) {
    uint64_t nray_sum;
    int im_start, im_end;

    nray_sum=0;

    im_start = imesh;
    im_end = im_start + this_run->nmesh_per_loop;

    for(int im=im_start;im<im_end;im++) {
      nray_sum += nray_to[im];
    }

    nray_sum_max = MAX(nray_sum_max, nray_sum);
  }

  free(key);
  
  /* allocate light_ray structures */
  ray = (struct light_ray *)malloc(sizeof(struct light_ray)*nray_sum_max);
  fprintf(this_run->proc_file, "# nray_sum_max = %llu\n", nray_sum_max);
  fprintf(this_run->proc_file, 
	  "# data size for the light rays : %llu [MByte] \n", 
	  (sizeof(struct light_ray)*nray_sum_max)>>20);

  end_timing(&start_init_tv, &end_init_tv, &start_init_tms, &end_init_tms,
	     "Initialization before the iteration", this_run);
  
  struct timeval iter_start_tv, iter_end_tv;
  struct tms iter_start_tms, iter_end_tms;

  /* iteration to achieve the radiation equilibrium */
  int iter = 0;
  do {

    start_timing(&iter_start_tv, &iter_start_tms);
    
#ifdef __USE_GPU__
    zero_out_photoion_rate_cuda(cuda_mem, this_cuda, this_run);
#ifndef __USE_GPU_ARGOT__
    recv_mesh_data(mesh, cuda_mem, this_cuda, this_run);
#endif
#else
    zero_out_photoion_rate(mesh, this_run);
#endif

    
#ifdef __DIFFUSE_RADIATION__
#ifdef __USE_GPU__
    calc_radiation_transfer(mesh, cuda_mem, this_cuda, ray, src, index, nray_to, clist, this_run, this_mpi, cd_param, hd_param);
#else
    calc_radiation_transfer(mesh, ray, src, index, nray_to, clist, this_run, this_mpi, hd_param);
#endif
#else
#ifdef __USE_GPU__
    calc_radiation_transfer(mesh, cuda_mem, this_cuda, ray, src, index, nray_to, clist, this_run, this_mpi);
#else
    calc_radiation_transfer(mesh, ray, src, index, nray_to, clist, this_run, this_mpi);
#endif
#endif
    
  
#ifdef __USE_GPU__
    smooth_photoion_rate(mesh, this_run, cuda_mem, this_cuda, this_mpi);
#else
    smooth_photoion_rate(mesh, this_run, this_mpi);
#endif

#ifdef __USE_GPU__
    step_chemistry(cuda_mem, this_cuda, this_run, dtime, &max_diff_chem, &max_diff_uene);
#else      
    step_chemistry(mesh, this_run, dtime, &max_diff_chem, &max_diff_uene);
#endif


    fprintf(this_run->proc_file, "# iteration (iter=%d)\n",iter);
    end_timing(&iter_start_tv, &iter_end_tv, &iter_start_tms, &iter_end_tms,
	       "iteration", this_run);
    fprintf(this_run->proc_file, "\n");

    iter++;

    // }while((max_diff_chem>1.0e-4 || max_diff_uene > 1.0e-4) && (iter<100));
  }while((max_diff_chem>1.0e-3 || max_diff_uene>1.0e-3) && (iter<5));

  
#ifdef __USE_GPU__
  update_chemistry_gpu(cuda_mem, this_cuda, this_run);
#else
  update_chemistry(mesh, this_run);
#endif

  free(ray);
  free(index);
  free(nray_to);
  free(clist);
  
  end_timing(&start_tv, &end_tv, &start_tms, &end_tms,
	     "step_radiation_tree", this_run);
  fprintf(this_run->proc_file,"\n");
}


#if 0

void calc_photoion_rate_at_first(struct fluid_mesh *mesh,
				 struct radiation_src *src,
#ifdef __USE_GPU__
				 struct cuda_mem_space *cuda_mem,
				 struct cuda_param *this_cuda,
#endif
				 struct run_param *this_run,
				 struct mpi_param *this_mpi
				 )
{
  struct light_ray *ray;
  struct ray_segment *seg;
#ifdef __USE_GPU_ARGOT__
  struct light_ray_IO *ray_IO;
#endif

  struct clist_t *clist;
  uint64_t *key;
  int *index;
  int nclist;

  uint64_t *nray_to;

  struct timeval start_init_tv, end_init_tv;
  struct tms start_init_tms, end_init_tms;
  start_timing(&start_init_tv, &start_init_tms);

  /* setup the parameters for the tree structure */
  this_run->theta_crit = ARGOT_THETA_CRIT;
  init_tree_param(60, this_run, &nclist);

  /* allocate the tree structure */
  clist = (struct clist_t *)malloc(sizeof(struct clist_t)*nclist);
  key = (uint64_t *)malloc(sizeof(uint64_t)*this_run->nsrc);
  index = (int *)malloc(sizeof(int)*this_run->nsrc);

  /* array of the number of light ray to a mesh point */
  nray_to = (uint64_t *)malloc(sizeof(uint64_t)*NMESH_LOCAL);

  construct_tree(clist, key, index, src, this_run);

  /* count the number of light rays targeted to each mesh */
#pragma omp parallel for schedule(auto)
  for(int imesh=0;imesh<NMESH_LOCAL;imesh++) {
    int ix, iy, iz;

    ix = imesh/(NMESH_Y_LOCAL*NMESH_Z_LOCAL);
    iy = (imesh-ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL)/NMESH_Z_LOCAL;
    iz = imesh - ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL - iy*NMESH_Z_LOCAL;

    nray_to[imesh] = count_ray_to(ix, iy, iz, clist, index, this_run);
    
    assert(nray_to[imesh] <= MAX_NRAY_PER_TARGET);
  }

  /* measure the maximum number of light rays for all loops of target meshes */
  uint64_t nray_sum_max;
  nray_sum_max=0;

#pragma omp parallel for schedule(auto) reduction(max:nray_sum_max)
  for(int imesh=0;imesh<NMESH_LOCAL;imesh+=this_run->nmesh_per_loop) {

    uint64_t nray_sum;
    int im_start, im_end;

    nray_sum=0;

    im_start = imesh;
    im_end = im_start + this_run->nmesh_per_loop;

    for(int im=im_start;im<im_end;im++) {
      nray_sum += nray_to[im];
    }

    nray_sum_max = MAX(nray_sum_max, nray_sum);
  }

  /* allocate light_ray structures */
  ray = (struct light_ray *)malloc(sizeof(struct light_ray)*nray_sum_max);
  fprintf(this_run->proc_file, "# nray_sum_max = %llu\n", nray_sum_max);
  fprintf(this_run->proc_file, 
	  "# data size for the light rays : %llu [MByte] \n", 
	  (sizeof(struct light_ray)*nray_sum_max)>>20);

  end_timing(&start_init_tv, &end_init_tv, &start_init_tms, &end_init_tms,
	     "Initialization before the first estimate of photo-Gamma", this_run);
  
#ifdef __USE_GPU__
    zero_out_photoion_rate_cuda(cuda_mem, this_cuda, this_run);
#ifndef __USE_GPU_ARGOT__
    recv_mesh_data(mesh, cuda_mem, this_cuda, this_run);
#endif
#else
    zero_out_photoion_rate(mesh, this_run);
#endif
    
  /* loop over the local target meshes */
  int loop_id = 0;

  for(int imesh=0;imesh<NMESH_LOCAL;imesh+=this_run->nmesh_per_loop) {

    start_timing(&start_init_tv, &start_init_tms);

    /* start and end indices */
    int im_start, im_end;
      
    im_start = imesh;
    im_end   = imesh+this_run->nmesh_per_loop;
	
    fprintf(this_run->proc_file,
	    "# ========================================\n");
    fprintf(this_run->proc_file,
	    "# loop ID : %d\n", loop_id++);
    fprintf(this_run->proc_file,
	    "# loop over taget meshes from %d to %d of %d meshes\n",
	    im_start, im_end, NMESH_LOCAL);
    
    /* calculate the index range of the target meshes */
    this_run->nray = set_mesh_index_range(im_start, im_end, nray_to);
    
    /* setting up the light rays using the tree structure */ 
    setup_light_ray_range(im_start, im_end, ray, 1, clist, src, 
			  index, this_run);
      
    /* setting up segments of light rays */
    count_ray_segment(ray, this_run);

#ifdef __ARGOT_PROFILE__
    fprintf(this_run->proc_file,
	    "# number of ray segments :: %llu\n", this_run->nseg);
    fprintf(this_run->proc_file,
	    "# data size for ray segments :: %14.6e MBytes\n",
	    (float)(sizeof(struct ray_segment)*this_run->nseg)/(float)(1<<20));
#endif /* __ARGOT_PROFILE__ */


    end_timing(&start_init_tv, &end_init_tv, &start_init_tms, &end_init_tms,
	       "Initialization in the first estimate of photo-Gamma", this_run);

#ifdef __USE_GPU_ARGOT__
    allocate_pinned_segment(&seg, this_run->nseg);
    allocate_pinned_light_ray_IO(&ray_IO, this_run->nray);
#else
    seg = 
      (struct ray_segment*) malloc(sizeof(struct ray_segment)*this_run->nseg);
#endif

    assign_ray_segment(ray, seg, this_run, this_mpi);

    /* compute optical depths of the light ray segments */
#ifdef __USE_GPU_ARGOT__
    calc_optical_depth_cuda(seg, cuda_mem, this_cuda, this_run);
#else
    calc_optical_depth(seg, mesh, this_run);
#endif

    /* accumulate optical depths  */
    accum_optical_depth(ray, seg, this_run);
      
    /* calculate photo-ionization rate */
#ifdef __USE_GPU_ARGOT__
    calc_photoion_rate_cuda(cuda_mem, ray, ray_IO, this_cuda, this_run);
#else
    calc_photoion_rate(mesh, ray, this_run);
#endif

#ifdef __USE_GPU_ARGOT__
    deallocate_pinned_segment(seg);
    deallocate_pinned_light_ray_IO(ray_IO);
#else
    free(seg);
#endif
  }

#ifdef __USE_GPU__
#ifndef __USE_GPU_ARGOT__
  send_mesh_data(mesh, cuda_mem, this_cuda, this_run);
#endif
  smooth_photoion_rate(mesh, this_run, cuda_mem, this_cuda, this_mpi);
#else
  smooth_photoion_rate(mesh, this_run, this_mpi);
#endif

  free(clist);
  free(key);
  free(index);
  free(nray_to);
  free(ray);
}

void step_radiation_tree(struct fluid_mesh *mesh, struct radiation_src *src, 
			 struct run_param *this_run, struct mpi_param *this_mpi,
#ifdef __USE_GPU__
			 struct cuda_mem_space *cuda_mem,
			 struct cuda_param *this_cuda,
#endif
#ifdef __DIFFUSE_RADIATION__
			 struct host_diffuse_param *hd_param,
#ifdef __USE_GPU__
			 struct cuda_diffuse_param *cd_param,
#endif /* __USE_GPU__ */
#endif /* __DIFFUSE_RADIATION__ */
			 float  dtime)
{

  struct light_ray *ray;
  struct ray_segment *seg;
#ifdef __USE_GPU_ARGOT__
  struct light_ray_IO *ray_IO;
#endif

  struct clist_t *clist;
  uint64_t *key;
  int *index;
  int nclist;

  uint64_t *nray_to;

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

  struct timeval start_init_tv, end_init_tv;
  struct tms start_init_tms, end_init_tms;

#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);

  times(&start_init_tms);
  gettimeofday(&start_init_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

  float max_diff_chem, max_diff_uene;
  float elapsed_time, dtime_min;

  /* setup the parameters for the tree structure */
  this_run->theta_crit = ARGOT_THETA_CRIT;
  init_tree_param(60, this_run, &nclist);

  /* allocate the tree structure */
  clist = (struct clist_t *)malloc(sizeof(struct clist_t)*nclist);
  key = (uint64_t *)malloc(sizeof(uint64_t)*this_run->nsrc);
  index = (int *)malloc(sizeof(int)*this_run->nsrc);

  /* array of the number of light ray to a mesh point */
  nray_to = (uint64_t *)malloc(sizeof(uint64_t)*NMESH_LOCAL);

  construct_tree(clist, key, index, src, this_run);

  uint64_t max_nray_per_target;
  max_nray_per_target=0;
  /* count the number of light rays targeted to each mesh */
#pragma omp parallel for schedule(auto) reduction(max:max_nray_per_target)
  for(int imesh=0;imesh<NMESH_LOCAL;imesh++) {
    int ix, iy, iz;

    ix = imesh/(NMESH_Y_LOCAL*NMESH_Z_LOCAL);
    iy = (imesh-ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL)/NMESH_Z_LOCAL;
    iz = imesh - ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL - iy*NMESH_Z_LOCAL;

    nray_to[imesh] = count_ray_to(ix, iy, iz, clist, index, this_run);

    max_nray_per_target = MAX(max_nray_per_target, nray_to[imesh]);
    
    assert(nray_to[imesh] <= MAX_NRAY_PER_TARGET);
  }
  fprintf(this_run->proc_file, "# max_nray_per_target : %llu\n", 
	  max_nray_per_target);

  /* measure the maximum number of light rays for all loops of target meshes */
  uint64_t nray_sum_max;
  nray_sum_max=0;

#pragma omp parallel for schedule(auto) reduction(max:nray_sum_max)
  for(int imesh=0;imesh<NMESH_LOCAL;imesh+=this_run->nmesh_per_loop) {
    uint64_t nray_sum;
    int im_start, im_end;

    nray_sum=0;

    im_start = imesh;
    im_end = im_start + this_run->nmesh_per_loop;

    for(int im=im_start;im<im_end;im++) {
      nray_sum += nray_to[im];
    }

    nray_sum_max = MAX(nray_sum_max, nray_sum);
  }

  /* allocate light_ray structures */
  ray = (struct light_ray *)malloc(sizeof(struct light_ray)*nray_sum_max);
  fprintf(this_run->proc_file, "# nray_sum_max = %llu\n", nray_sum_max);
  fprintf(this_run->proc_file, 
	  "# data size for the light rays : %llu [MByte] \n", 
	  (sizeof(struct light_ray)*nray_sum_max)>>20);

  end_timing(&start_init_tv, &end_init_tv, &start_init_tms, &end_init_tms,
	     "Initialization before the iteration", this_run);
  
  int iter;

  struct timeval iter_start_tv, iter_end_tv;
  struct tms iter_start_tms, iter_end_tms;

  /* iteration to achieve the radiation equilibrium */
  iter = 0;
  do {

    start_timing(&iter_start_tv, &iter_start_tms);

#ifdef __USE_GPU__
    zero_out_photoion_rate_cuda(cuda_mem, this_cuda, this_run);
#ifndef __USE_GPU_ARGOT__
    recv_mesh_data(mesh, cuda_mem, this_cuda, this_run);
#endif
#else
    zero_out_photoion_rate(mesh, this_run);
#endif
    
    /* loop over the local target meshes */
    int loop_id=0;
    for(int imesh=0;imesh<NMESH_LOCAL;imesh+=this_run->nmesh_per_loop) {

      start_timing(&start_init_tv, &start_init_tms);
      
      /* start and end indices */
      int im_start, im_end;
      
      im_start = imesh;
      im_end   = imesh+this_run->nmesh_per_loop;
      //      im_end   = imesh+NMESH_PER_LOOP;
	
      fprintf(this_run->proc_file,
	      "# ========================================\n");
      fprintf(this_run->proc_file,
	      "# loop ID : %d\n", loop_id++);
      fprintf(this_run->proc_file,
	      "# loop over taget meshes from %d to %d of %d meshes\n",
	      im_start, im_end, NMESH_LOCAL);

      /* calculate the index range of the target meshes */
      this_run->nray = set_mesh_index_range(im_start, im_end, nray_to);

      /* setting up the light rays using the tree structure */ 
      setup_light_ray_range(im_start, im_end, ray, 1, clist, src, 
			    index, this_run);
      
      /* setting up segments of light rays */
      count_ray_segment(ray, this_run);

#ifdef __USE_GPU_ARGOT__
      allocate_pinned_segment(&seg, this_run->nseg);
      allocate_pinned_light_ray_IO(&ray_IO, this_run->nray);
#else
      seg = 
	(struct ray_segment*) malloc(sizeof(struct ray_segment)*this_run->nseg);
#endif

#ifdef __ARGOT_PROFILE__
      fprintf(this_run->proc_file,
	      "# number of ray segments :: %llu\n", this_run->nseg);
      fprintf(this_run->proc_file,
	      "# data size for ray segments :: %14.6e MBytes\n",
	      (float)(sizeof(struct ray_segment)*this_run->nseg)/(float)(1<<20));
#endif /* __ARGOT_PROFILE__ */

      end_timing(&start_init_tv, &end_init_tv, &start_init_tms, &end_init_tms,
		 "Initialization within the iteration", this_run);
      
      assign_ray_segment(ray, seg, this_run, this_mpi);

      /* compute optical depths of the light ray segments */
#ifdef __USE_GPU_ARGOT__
      calc_optical_depth_cuda(seg, cuda_mem, this_cuda, this_run);
#else
      calc_optical_depth(seg, mesh, this_run);
#endif

      /* accumulate optical depths  */
      accum_optical_depth(ray, seg, this_run);
      
      /* calculate photo-ionization rate */
#ifdef __USE_GPU_ARGOT__
      calc_photoion_rate_cuda(cuda_mem, ray, ray_IO, this_cuda, this_run);
#else
      calc_photoion_rate(mesh, ray, this_run);
#endif

#ifdef __USE_GPU_ARGOT__
      deallocate_pinned_segment(seg);
      deallocate_pinned_light_ray_IO(ray_IO);
#else
      free(seg);
#endif
    }


#ifdef __DIFFUSE_RADIATION__
    struct timeval dp_start_tv, dp_end_tv;
    struct tms dp_start_tms, dp_end_tms;

    start_timing(&dp_start_tv, &dp_start_tms);

#ifdef __USE_GPU_ART__
    calc_diffuse_photon_radiation(mesh, this_run, cuda_mem, this_cuda, hd_param, cd_param);

#ifndef __USE_GPU_ARGOT__
    /* ART.photoion += ARGOT.photoion on GPU */
    merge_photoion_data(mesh, cuda_mem, this_cuda, this_run);
#endif
#else /* !__USE_GPU_ART__ */
    calc_diffuse_photon_radiation(mesh, this_run, hd_param); 
#endif /* __USE_GPU_ART__ */

    end_timing(&dp_start_tv, &dp_end_tv, &dp_start_tms, &dp_end_tms,
	       "Diffuse RT", this_run);
#endif /* __DIFFUSE_RADIATION__ */
    
    
#ifdef __USE_GPU__
    smooth_photoion_rate(mesh, this_run, cuda_mem, this_cuda, this_mpi);
#else
    smooth_photoion_rate(mesh, this_run, this_mpi);
#endif

#ifdef __USE_GPU__
    step_chemistry(cuda_mem, this_cuda, this_run, dtime, &max_diff_chem, &max_diff_uene);
#else      
    step_chemistry(mesh, this_run, dtime, &max_diff_chem, &max_diff_uene);
#endif


    fprintf(this_run->proc_file, "# iteration (iter=%d)\n",iter);
    end_timing(&iter_start_tv, &iter_end_tv, &iter_start_tms, &iter_end_tms,
	       "iteration", this_run);
    fprintf(this_run->proc_file, "\n");

    iter++;

    // }while((max_diff_chem>1.0e-4 || max_diff_uene > 1.0e-4) && (iter<100));
  }while((max_diff_chem>1.0e-3 || max_diff_uene>1.0e-3) && (iter<8));

  
#ifdef __USE_GPU__
  update_chemistry_gpu(cuda_mem, this_cuda, this_run);
#else
  update_chemistry(mesh, this_run);
#endif

  free(ray);
  free(clist);
  free(key);
  free(index);
  free(nray_to);

  end_timing(&start_tv, &end_tv, &start_tms, &end_tms,
	     "step_radiation_tree", this_run);
  fprintf(this_run->proc_file,"\n");
}


#endif
