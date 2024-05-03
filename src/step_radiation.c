/***
not supported.
 ***/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>

#include "constants.h"
#include "run_param.h"
#include "mpi_param.h"
#include "radiation.h"

#include "prototype.h"

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])


#ifdef __DIFFUSE_RADIATION__
#include "diffuse_photon.h"
#endif

uint64_t set_mesh_index_range(int imesh_start, int imesh_end, uint64_t *nray_to)
{
  int imesh;
  uint64_t nray_sum;

#if 0
  nray_sum=0;

  imesh = imesh_start;
  while(nray_sum+nray_to[imesh] < MAX_NRAY_PER_TARGET && imesh < NMESH_LOCAL) {
    nray_sum += nray_to[imesh];
    imesh++;
  }

  *imesh_end = imesh;

  assert(nray_sum <= MAX_NRAY_PER_TARGET);
#else
  nray_sum=0;
  for(imesh=imesh_start;imesh<imesh_end;imesh++) {
    nray_sum += nray_to[imesh];
  }

  assert(nray_sum <= MAX_NRAY_PER_TARGET);
#endif

  return nray_sum;
}


void set_optimal_nmesh_per_loop(struct radiation_src *src, 
				struct run_param *this_run)
{
  fprintf(this_run->proc_file, "# optimal nmesh_per_loop = %d\n",
	  this_run->nmesh_per_loop);
  fflush(this_run->proc_file);
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
  struct ray_segment *seg;
#ifdef __USE_GPU__
  struct light_ray_IO *ray_IO;
#endif

  int iter;

  struct timeval iter_start_tv, iter_end_tv;
  struct tms iter_start_tms, iter_end_tms;
  struct timeval start_init_tv, end_init_tv;
  struct tms start_init_tms, end_init_tms;

#ifdef __ARGOT_PROFILE__
  times(&iter_start_tms);
  gettimeofday(&iter_start_tv, NULL);

  times(&start_init_tms);
  gettimeofday(&start_init_tv, NULL);
#endif

#ifdef __USE_GPU__
  zero_out_photoion_rate(cuda_mem, this_cuda, this_run);
#else
  zero_out_photoion_rate(mesh, this_run);
#endif
    
  /* loop over the local target meshes */
  int loop_id = 0;

  for(int imesh=0;imesh<NMESH_LOCAL;imesh+=this_run->nmesh_per_loop) {
    
#ifdef __ARGOT_PROFILE__
    times(&start_init_tms);
    gettimeofday(&start_init_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

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

    /* setting up the light rays */ 
    setup_light_ray_long(&ray, src, this_run);
     
    /* setting up segments of light rays */
    count_ray_segment(ray, this_run);

#ifdef __ARGOT_PROFILE__
    fprintf(this_run->proc_file,
	    "# number of ray segments :: %llu\n", this_run->nseg);
    fprintf(this_run->proc_file,
	    "# data size for ray segments :: %14.6e MBytes\n",
	    (float)(sizeof(struct ray_segment)*this_run->nseg)/(float)(1<<20));
#endif /* __ARGOT_PROFILE__ */

#ifdef __ARGOT_PROFILE__
    times(&end_init_tms);
    gettimeofday(&end_init_tv, NULL);
    
    fprintf(this_run->proc_file,
	    "# Initialization in the first estimate of photo-Gamma : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	    timing(start_init_tms, end_init_tms), 
	    wallclock_timing(start_init_tv, end_init_tv));
    fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */

#ifdef __USE_GPU__
    allocate_pinned_segment(&seg, this_run->nseg);
    allocate_pinned_light_ray_IO(&ray_IO, this_run->nray);
#else
    seg = 
      (struct ray_segment*) malloc(sizeof(struct ray_segment)*this_run->nseg);
#endif

    assign_ray_segment(ray, seg, this_run, this_mpi);

    /* compute optical depths of the light ray segments */
#ifdef __USE_GPU__
    calc_optical_depth(seg, cuda_mem, this_cuda, this_run);
#else
    calc_optical_depth(seg, mesh, this_run);
#endif

    /* accumulate optical depths  */
    accum_optical_depth(ray, seg, this_run);
      
    /* calculate photo-ionization rate */
#ifdef __USE_GPU__
    calc_photoion_rate(cuda_mem, ray, ray_IO, this_cuda, this_run);
#else
    calc_photoion_rate(mesh, ray, this_run);
#endif

#ifdef __USE_GPU__
    deallocate_pinned_segment(seg);
    deallocate_pinned_light_ray_IO(ray_IO);
#else
    free(seg);
#endif
   
    free(ray);
  }


#ifdef __USE_GPU__
  smooth_photoion_rate(mesh, this_run, cuda_mem, this_cuda, this_mpi);
#else
  smooth_photoion_rate(mesh, this_run, this_mpi);
#endif

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
#ifdef __USE_GPU__
  struct light_ray_IO *ray_IO;
#endif

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

#ifdef __ARGOT_PROFILE__
  times(&end_init_tms);
  gettimeofday(&end_init_tv, NULL);
  
  fprintf(this_run->proc_file,
	  "# Initialization before the iteration : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	  timing(start_init_tms, end_init_tms), 
	  wallclock_timing(start_init_tv, end_init_tv));
  fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */

  int iter;

  struct timeval iter_start_tv, iter_end_tv;
  struct tms iter_start_tms, iter_end_tms;

  /* iteration to achieve the radiation equilibrium */
  iter = 0;
  do {

#ifdef __ARGOT_PROFILE__
    times(&iter_start_tms);
    gettimeofday(&iter_start_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

#ifdef __USE_GPU__
    zero_out_photoion_rate(cuda_mem, this_cuda, this_run);
#else
    zero_out_photoion_rate(mesh, this_run);
#endif
    
    /* loop over the local target meshes */
    int loop_id=0;

    for(int imesh=0;imesh<NMESH_LOCAL;imesh+=this_run->nmesh_per_loop) {

#ifdef __ARGOT_PROFILE__
      times(&start_init_tms);
      gettimeofday(&start_init_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

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

      /* setting up the light rays */ 
      setup_light_ray_long(&ray, src, this_run);
      
      /* setting up segments of light rays */
      count_ray_segment(ray, this_run);

#ifdef __USE_GPU__
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

#ifdef __ARGOT_PROFILE__
      times(&end_init_tms);
      gettimeofday(&end_init_tv, NULL);
      
      fprintf(this_run->proc_file,
	      "# Initialization within the iteration : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
              timing(start_init_tms, end_init_tms), 
	      wallclock_timing(start_init_tv, end_init_tv));
      fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */

      assign_ray_segment(ray, seg, this_run, this_mpi);
      
      /* compute optical depths of the light ray segments */
#ifdef __USE_GPU__
      calc_optical_depth(seg, cuda_mem, this_cuda, this_run);
#else
      calc_optical_depth(seg, mesh, this_run);
#endif

      /* accumulate optical depths  */
      accum_optical_depth(ray, seg, this_run);
      
      /* calculate photo-ionization rate */
#ifdef __USE_GPU__
      calc_photoion_rate(cuda_mem, ray, ray_IO, this_cuda, this_run);
#else
      calc_photoion_rate(mesh, ray, this_run);
#endif

#ifdef __USE_GPU__
      deallocate_pinned_segment(seg);
      deallocate_pinned_light_ray_IO(ray_IO);
#else
      free(seg);
#endif
      free(ray);
      
    }

    // diffuse radiation 
#ifdef __DIFFUSE_RADIATION__
    struct timeval dp_start_tv, dp_end_tv;
    struct tms dp_start_tms, dp_end_tms;

    times(&dp_start_tms);
    gettimeofday(&dp_start_tv, NULL);

#ifdef __USE_GPU__
    calc_diffuse_photon_radiation(mesh, this_run, cuda_mem, this_cuda, hd_param, cd_param);
#else
    calc_diffuse_photon_radiation(mesh, this_run, hd_param); 
#endif /* __USE_GPU__ */
    
    times(&dp_end_tms);
    gettimeofday(&dp_end_tv, NULL);
     
    fprintf(this_run->proc_file,
            "# Diffuse RT : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n"
            ,timing(dp_start_tms, dp_end_tms)
            ,wallclock_timing(dp_start_tv, dp_end_tv));

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

#ifdef __ARGOT_PROFILE__
    times(&iter_end_tms);
    gettimeofday(&iter_end_tv, NULL);
    fprintf(this_run->proc_file,
	    "# iteration (iter=%d) : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	    iter, timing(iter_start_tms, iter_end_tms), 
	    wallclock_timing(iter_start_tv, iter_end_tv));

    fprintf(this_run->proc_file, "\n");
    fflush(this_run->proc_file);
#endif

    iter++;

    // }while((max_diff_chem>1.0e-3 || max_diff_uene>1.0e-3) && (iter<20));
  }while((max_diff_chem>1.0e-2 || max_diff_uene>1.0e-2) && (iter<20));
    //}while((max_diff_chem>6.0e-2 || max_diff_uene>6.0e-2) && (iter<10)); /* for narrow ionization transition reagion */

#ifdef __USE_GPU__
  update_chemistry_gpu(cuda_mem, this_cuda, this_run);
#else
  update_chemistry(mesh, this_run);
#endif

#ifdef __ARGOT_PROFILE__
  times(&end_tms);
  gettimeofday(&end_tv, NULL);

  fprintf(this_run->proc_file,
	  "# step_radiation_tree : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	  timing(start_tms, end_tms), wallclock_timing(start_tv, end_tv));
  fprintf(this_run->proc_file,"\n");
  fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */


}





/* old version */
#if 0

void allocate_pinned_segment(struct ray_segment**, uint64_t);

void step_radiation(struct fluid_mesh *mesh, struct radiation_src *src, 
		    struct run_param *this_run, struct mpi_param *this_mpi,
#ifdef __USE_GPU__
		    struct cuda_mem_space *cuda_mem,
		    struct cuda_param *this_cuda,
#endif
		    float  dtime)
{
  struct light_ray *ray;
  struct ray_segment *seg;

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

#if 0
  setup_light_ray_long(&ray, src, this_run);
#else
  this_run->theta_crit = 0.7;
  setup_light_ray_tree(&ray, src, this_run);
#endif

  count_ray_segment(ray, this_run);

  fprintf(this_run->proc_file,"# number of ray segments :: %llu\n", this_run->nseg);

#ifdef __USE_GPU__
  allocate_pinned_segment(&seg, this_run->nseg);
#else
  seg = (struct ray_segment*) malloc(sizeof(struct ray_segment)*this_run->nseg);
#endif

  assign_ray_segment(ray, seg, this_run, this_mpi);

  float maxdiff;
  float elapsed_time, dtime_min;

  elapsed_time = 0.0;

  while( elapsed_time < dtime ) {
    int iter;
    struct timeval start_iter_tv,  end_iter_tv;
    struct tms     start_iter_tms, end_iter_tms;

    iter = 0;
    do {

#ifdef __ARGOT_PROFILE__
      times(&start_iter_tms);
      gettimeofday(&start_iter_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

#ifdef __USE_GPU__
      calc_optical_depth(seg, cuda_mem, this_cuda, this_run);
#else
      calc_optical_depth(seg, mesh, this_run);
#endif

      accum_optical_depth(ray, seg, this_run);

      zero_out_photoion_rate(mesh, this_run);
      calc_photoion_rate(mesh, ray, this_run);

#ifdef __USE_GPU__
      smooth_photoion_rate(mesh, this_run, cuda_mem, this_cuda, this_mpi);
#else
      smooth_photoion_rate(mesh, this_run, this_mpi);
#endif

      if(iter == 0) {
#ifdef __USE_GPU__
	dtime_min = DTFACT_RAD*calc_timestep(cuda_mem, this_cuda, this_run);
#else
	dtime_min = DTFACT_RAD*calc_timestep(mesh, this_run);
#endif
	fprintf(this_run->proc_file,"# dt = %14.6e\n",dtime_min);
	fflush(this_run->proc_file);
	if(elapsed_time + dtime_min > dtime) {
	  dtime_min = dtime - elapsed_time;
	}
      }

#ifdef __USE_GPU__
      step_chemistry(cuda_mem, this_cuda, this_run, dtime_min, &maxdiff);
#else      
      step_chemistry(mesh, this_run, dtime_min, &maxdiff);
#endif
     
      iter++;
      //      if(this_run->mpi_rank == 0) printf("%14.6e\n", maxdiff);

#ifdef __ARGOT_PROFILE__
      times(&end_iter_tms);
      gettimeofday(&end_iter_tv, NULL);
      fprintf(this_run->proc_file,
	      "# iteration : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	      timing(start_iter_tms, end_iter_tms), wallclock_timing(start_iter_tv, end_iter_tv));
      fprintf(this_run->proc_file,"\n");
#endif /* __ARGOT_PROFILE__ */

    }while(maxdiff>1.0e-3);

#ifdef __USE_GPU__
    update_chemistry_gpu(cuda_mem, this_cuda, this_run);
#else
    update_chemistry(mesh, this_run);
#endif

    elapsed_time += dtime_min;
    if(this_run->mpi_rank == 0) fprintf(this_run->proc_file,
					"elapsed_time = %14.6e\n",elapsed_time);

  }

#ifdef __USE_GPU__
  deallocate_pinned_segment(seg);
#else
  free(seg);
#endif

  free(ray);

#ifdef __ARGOT_PROFILE__
  times(&end_tms);
  gettimeofday(&end_tv, NULL);

  fprintf(this_run->proc_file,
	  "# step_radiation : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	  timing(start_tms, end_tms), wallclock_timing(start_tv, end_tv));
  fprintf(this_run->proc_file,"\n");
  fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */

}

#if 0
  uint64_t iray;
  for(iray=0;iray<this_run->nray;iray++) {
    float dist;
    float x_tgt, y_tgt, z_tgt;
    
    x_tgt = this_run->xmin + ((float)(ray[iray].ix_target)+0.5)*this_run->delta_x;
    y_tgt = this_run->ymin + ((float)(ray[iray].iy_target)+0.5)*this_run->delta_y;
    z_tgt = this_run->zmin + ((float)(ray[iray].iz_target)+0.5)*this_run->delta_z;

    dist = SQR(ray[iray].src.xpos-x_tgt)
      +    SQR(ray[iray].src.ypos-y_tgt) 
      +    SQR(ray[iray].src.zpos-z_tgt);
    dist = sqrt(dist);
    
    fprintf(this_run->proc_file, "%14.6e %14.6e %14.6e aaa\n", 
	    dist, ray[iray].optical_depth, 
	    exp(-ray[iray].optical_depth)/(4.0*PI*SQR(dist*this_run->lunit)));
  }
#endif

#if 0
  int ix,iy,iz;
  float x, y, z;
  for(ix=0;ix<NMESH_X_LOCAL;ix++) {
    x = this_run->xmin_local + ((float)ix+0.5)*this_run->delta_x;
    for(iy=0;iy<NMESH_Y_LOCAL;iy++) {
      y = this_run->ymin_local + ((float)iy+0.5)*this_run->delta_y;
      for(iz=0;iz<NMESH_Z_LOCAL;iz++) {
	z = this_run->zmin_local + ((float)iz+0.5)*this_run->delta_z;

	float dist2;

	dist2 = SQR(x-src[0].xpos)+SQR(y-src[0].ypos)+SQR(z-src[0].zpos);

	fprintf(this_run->proc_file,"%14.6e %14.6e %14.6e\n", sqrt(dist2), MESH(ix,iy,iz).prev_chem.GammaHI, MESH(ix,iy,iz).chem.GammaHI);


      }
    }
  }
  fflush(this_run->proc_file);
#endif


#endif /* old version */
