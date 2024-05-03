#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include "constants.h"
#include "run_param.h"
#include "mpi_param.h"
#include "radiation.h"

#include "prototype.h"

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

int main(int argc, char **argv) 
{
  static struct run_param this_run;
  static struct mpi_param this_mpi;

  struct fluid_mesh *mesh;
  struct light_ray *ray;
  static struct radiation_src src[NSOURCE_MAX];

  struct ray_segment *seg;

#ifdef __USE_GPU__
  struct cuda_param this_cuda;
  struct cuda_mem_space cuda_mem[NMAX_CUDA_DEV];
#endif

  int ix, iy, iz;

  mesh = (struct fluid_mesh *) malloc(sizeof(struct fluid_mesh)*NMESH_LOCAL);  

  MPI_Init(&argc, &argv);

  init_mpi(&this_run, &this_mpi);

  input_data(mesh, src, &this_run, argv[1]);

  init_run(&this_run);

#ifdef __USE_GPU__
  init_gpu(mesh, cuda_mem, &this_cuda, &this_run);
  send_mesh_data(mesh, cuda_mem, &this_cuda, &this_run);
#endif

  setup_light_ray_long(&ray, src, &this_run);

  count_ray_segment(ray, &this_run);

  fprintf(this_run.proc_file,"# number of ray segments :: %llu\n", this_run.nseg);

#ifdef __USE_GPU__
  allocate_pinned_segment(&seg, this_run.nseg);
#else
  seg = (struct ray_segment*) malloc(sizeof(struct ray_segment)*this_run.nseg);
#endif

  //  fprintf(this_run.proc_file,"%llu %14.6e\n",this_run.nsrc, src[0].L);

#if 0
  if(this_run.mpi_rank == 0) {
    printf("# of light rays : %d\n", this_run.nray);
    int iray;
    for(iray=0;iray<this_run.nray;iray++) {
      printf("# of segments : %d\n", ray[iray].num_segment);
    }
  }
#endif

  assign_ray_segment(ray, seg, &this_run, &this_mpi);

#ifdef __USE_GPU__
  calc_optical_depth(seg, cuda_mem, &this_cuda, &this_run);
#else
  calc_optical_depth(seg, mesh, &this_run);
#endif

#if 0
  uint64_t iseg;
  for(iseg=0;iseg<this_run.nseg;iseg++) {
    float dist;
    struct ray_segment *s;
    s = seg+iseg;
    dist = SQR(s->xpos_start-s->xpos_end)
      +    SQR(s->ypos_start-s->ypos_end)
      +    SQR(s->zpos_start-s->zpos_end);
    dist = sqrt(dist);
    fprintf(this_run.proc_file,"%llu %14.6e %14.6e\n", iseg, dist, s->optical_depth);
  }
#endif

  accum_optical_depth(ray, seg, &this_run);

#if 0
  uint64_t iray;
  for(iray=0;iray<this_run.nray;iray++) {
    float dist;
    float x_tgt, y_tgt, z_tgt;
    
    x_tgt = this_run.xmin + ((float)(ray[iray].ix_target)+0.5)*this_run.delta_x;
    y_tgt = this_run.ymin + ((float)(ray[iray].iy_target)+0.5)*this_run.delta_y;
    z_tgt = this_run.zmin + ((float)(ray[iray].iz_target)+0.5)*this_run.delta_z;

    dist = SQR(ray[iray].src.xpos-x_tgt)
      +    SQR(ray[iray].src.ypos-y_tgt) 
      +    SQR(ray[iray].src.zpos-z_tgt);
    dist = sqrt(dist);
    
    fprintf(this_run.proc_file, "%14.6e %14.6e\n", dist, ray[iray].optical_depth);
  }
#endif

  calc_photoion_rate(mesh, ray, &this_run);

#ifdef __USE_GPU__
  smooth_photoion_rate(mesh, &this_run, cuda_mem, &this_cuda, &this_mpi);
#else
  smooth_photoion_rate(mesh, &this_run, &this_mpi);
#endif

#if 0
   float x, y, z;
   for(ix=0;ix<NMESH_X_LOCAL;ix++) {
     x = this_run.xmin_local + ((float)ix+0.5)*this_run.delta_x;
     for(iy=0;iy<NMESH_Y_LOCAL;iy++) {
       y = this_run.ymin_local + ((float)iy+0.5)*this_run.delta_y;
       for(iz=0;iz<NMESH_Z_LOCAL;iz++) {
	 z = this_run.zmin_local + ((float)iz+0.5)*this_run.delta_z;

	 float dist2;

	 dist2 = SQR(x-src[0].xpos)+SQR(y-src[0].ypos)+SQR(z-src[0].zpos);

	 fprintf(this_run.proc_file,"%14.6e %14.6e %14.6e\n", sqrt(dist2), MESH(ix,iy,iz).prev_chem.GammaHI, MESH(ix,iy,iz).chem.GammaHI);


       }
     }
   }
#endif

  float dtime_min;

#ifdef __USE_GPU__
  dtime_min = DTFACT_RAD*calc_timestep(cuda_mem, &this_cuda, &this_run);
#else
  dtime_min = DTFACT_RAD*calc_timestep(mesh, &this_run);
#endif

  fprintf(this_run.proc_file, "dtime = %14.6e\n", dtime_min);

  float maxdiff;
#ifdef __USE_GPU__
  step_chemistry(cuda_mem, &this_cuda, &this_run, dtime_min, &maxdiff);
#else
  step_chemistry(mesh, &this_run, dtime_min, &maxdiff);
#endif

#ifdef __USE_GPU__
    update_chemistry_gpu(cuda_mem, &this_cuda, &this_run);
#else
    update_chemistry(mesh, &this_run);
#endif

  MPI_Finalize();
  exit(EXIT_SUCCESS);
  
}
