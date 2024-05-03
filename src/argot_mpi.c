#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include "constants.h"
#include "run_param.h"
#include "mpi_param.h"
#include "radiation.h"
#include "source.h"

#include "prototype.h"

#ifdef __USE_GPU__
#include "cuda_mem_space.h"
#endif

#ifdef __DIFFUSE_RADIATION__
#include "diffuse_photon.h"
#endif

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

int main(int argc, char **argv) 
{
  static struct run_param this_run;
  static struct mpi_param this_mpi;

  static struct radiation_src src[NSOURCE_MAX];

  struct fluid_mesh *mesh;

  mesh = (struct fluid_mesh *) malloc(sizeof(struct fluid_mesh)*NMESH_LOCAL);

#ifdef __USE_GPU__
  struct cuda_param this_cuda;
  struct cuda_mem_space cuda_mem[NMAX_CUDA_DEV];
#endif

#ifdef __DIFFUSE_RADIATION__
  static struct host_diffuse_param hd_param;
#ifdef __USE_GPU__
  struct cuda_diffuse_param cd_param[NMAX_CUDA_DEV];
#endif /* __USE_GPU__ */
#endif /* __DIFFUSE_RADIATION__ */

  int ix, iy, iz;

#if defined(__USE_GPU__) && defined(__DIFFUSE_RADIATION__) && (NMAX_CUDA_DEV != 1)
  int required = MPI_THREAD_MULTIPLE;   // multi threads call MPI operation 
  //int required = MPI_THREAD_SERIALIZED;     // single thread call MPI operation
  //int required = MPI_THREAD_FUNNELED;   // master thread call MPI operation
  int provided;
  MPI_Init_thread(&argc, &argv, required, &provided);
#else
  MPI_Init(&argc, &argv);
#endif

  init_mpi(&this_run, &this_mpi);

  input_data(mesh, src, &this_run, argv[1]);

  input_params(&this_run, argv[2]);

  init_run(&this_run);

#ifdef __USE_GPU__
  init_gpu(mesh, cuda_mem, &this_cuda, &this_run);
  send_mesh_data(mesh, cuda_mem, &this_cuda, &this_run);
#endif

#ifdef __DIFFUSE_RADIATION__
  init_hd_param( &hd_param );
  setup_diffuse_photon( &hd_param, &this_run );
#ifdef __USE_GPU__
  init_diffuse_gpu(cd_param, &this_cuda, &this_run);
  send_diffuse_data(&hd_param, cd_param, &this_cuda);
#endif /* __USE_GPU__  */
#endif /* __DIFFUSE_RADIATION__ */

  set_optimal_nmesh_per_loop(src, &this_run);

  this_run.step = 0;

  float dtime=1.0e-10;

#ifdef __USE_GPU__
  calc_photoion_rate_at_first(mesh, src, cuda_mem, &this_cuda, 
			      &this_run, &this_mpi);
  dtime = DTFACT_RAD*calc_timestep_chem(cuda_mem, &this_cuda, &this_run);
#else
  calc_photoion_rate_at_first(mesh, src, &this_run, &this_mpi);
  dtime = DTFACT_RAD*calc_timestep_chem(mesh, &this_run);
#endif
  fprintf(this_run.proc_file,"# dtime = %14.6e\n", dtime);

  output_diagnostics(mesh, &this_run, dtime);

  while(this_run.tnow < this_run.tend) {

#ifdef __USE_GPU__
#ifdef __DIFFUSE_RADIATION__
    step_radiation_tree(mesh, src, &this_run, &this_mpi, 
			cuda_mem, &this_cuda, &hd_param, cd_param, dtime);
#else /* !__DIFFUSE_RADIATION__ */
    step_radiation_tree(mesh, src, &this_run, &this_mpi, 
			cuda_mem, &this_cuda, dtime);
#endif /* __DIFFUSE_RADIATION__ */
#else /* !__USE_GPU__ */
#ifdef __DIFFUSE_RADIATION__
    step_radiation_tree(mesh, src, &this_run, &this_mpi, &hd_param, dtime);
#else /* !__DIFFUSE_RADIATION__ */
    step_radiation_tree(mesh, src, &this_run, &this_mpi, dtime);
#endif /* __DIFFUSE_RADIATION__ */
#endif /* __USE_GPU__ */

    this_run.tnow += dtime;
    this_run.step++;

#ifdef __COSMOLOGICAL__
    update_expansion(this_run.tnow, &this_run);
#ifdef __USE_GPU__
    send_run_param_data(&this_run, cuda_mem, &this_cuda);
#endif
#endif

#ifdef __USE_GPU__
    recv_mesh_data(mesh, cuda_mem, &this_cuda, &this_run);
#endif
    output_data_in_run(mesh, src, &this_run, this_run.model_name);
    output_diagnostics(mesh, &this_run, dtime);

#ifdef __USE_GPU__
    dtime = DTFACT_RAD*calc_timestep_chem(cuda_mem, &this_cuda, &this_run);
#else
    dtime = DTFACT_RAD*calc_timestep_chem(mesh, &this_run);
#endif
    
    /* adjust output timing */
    float next_output_timing = this_run.output_timing[this_run.output_indx]*1.001;
    if(this_run.tnow+dtime > next_output_timing) {
      dtime = next_output_timing-this_run.tnow;
    }

  }

#ifdef __DIFFUSE_RADIATION__
  free_hd_param(&hd_param);
#ifdef __USE_GPU__
  free_diffuse_gpu(cd_param, &this_cuda);
#endif /* __USE_GPU__ */  
#endif /* __DIFFUSE_RADIATION__ */

  MPI_Finalize();
  exit(EXIT_SUCCESS);
  
}


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

	fprintf(this_run.proc_file,"%14.6e %14.6e %14.6e %14.6e %14.6e\n", 
		sqrt(dist2), MESH(ix,iy,iz).chem.fHI, x, y, z);

      }
    }
  }
#endif

