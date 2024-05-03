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
#include "fluid.h"

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

  struct fluid_mesh *mesh;
  struct radiation_src *src;

  mesh = (struct fluid_mesh *) malloc(sizeof(struct fluid_mesh)*NMESH_LOCAL);
  src  = (struct radiation_src *)malloc(sizeof(struct radiation_src)*NSOURCE_MAX);

#ifdef __GRAVITY__
  static struct fftw_mpi_param this_fftw_mpi;
  float *green_func, *dens;
#endif

#ifdef __USE_GPU__
  struct cuda_param this_cuda;
  struct cuda_mem_space cuda_mem[NMAX_CUDA_DEV];
#endif

#ifdef __RADIATION_TRANSFER__
#ifdef __DIFFUSE_RADIATION__
  static struct host_diffuse_param hd_param;
#ifdef __USE_GPU__
  struct cuda_diffuse_param cd_param[NMAX_CUDA_DEV];
#endif /* __USE_GPU__ */
#endif /* __DIFFUSE_RADIATION__ */
#endif /* __RADIATION_TRANSFER__ */

#if defined(__USE_GPU__) && defined(__DIFFUSE_RADIATION__) && (NMAX_CUDA_DEV != 1)
  int required = MPI_THREAD_MULTIPLE;   // multi threads call MPI operation 
  // int required = MPI_THREAD_SERIALIZED;     // single thread call MPI operation
  // int required = MPI_THREAD_FUNNELED;   // master thread call MPI operation
  int provided;
  MPI_Init_thread(&argc, &argv, required, &provided);
#else
  MPI_Init(&argc, &argv);
#endif

  init_mpi(&this_run, &this_mpi);

#ifdef __GRAVITY__
  init_fftw_mpi(&this_run, &this_fftw_mpi, &green_func);
#endif

  input_data(mesh, src, &this_run, argv[1]);

  input_params(&this_run, argv[2]);

  init_run(&this_run);

#ifdef __RADIATION_TRANSFER__

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
#endif /* __USE_GPU__ */
#endif /* __DIFFUSE_RADIATION__ */

#endif /* __RADIATION_TRANSFER__ */

#ifdef __GRAVITY__
  init_green(green_func, &this_run, &this_fftw_mpi);
  dens = (float *) malloc(sizeof(float)*this_fftw_mpi.local_size);
#endif

#ifdef __RADIATION_TRANSFER__
  set_optimal_nmesh_per_loop(src, &this_run);
#endif

  this_run.step = 0;

  float dtime;
  dtime = 1.0e-10;
  
#ifdef __GRAVITY__
  /* first compute the gravitational potential field */
  zero_out_mesh_density(dens, &this_fftw_mpi);
  calc_mesh_density(dens, mesh, &this_run, &this_fftw_mpi);
  calc_mesh_grav_pot(dens, mesh, &this_run, &this_fftw_mpi, green_func);
#endif

  /* calculate delta_t for the first step */
#ifdef __RADIATION_TRANSFER__
#ifdef __USE_GPU__
  calc_photoion_rate_at_first(mesh, src, cuda_mem, &this_cuda, 
			      &this_run, &this_mpi);
  dtime = calc_timestep(mesh, cuda_mem, &this_cuda, &this_run);
#else
  calc_photoion_rate_at_first(mesh, src, &this_run, &this_mpi);
  dtime = calc_timestep(mesh, &this_run);
#endif
#else /* !__RADIATION_TRANSFER__ */
  dtime = calc_timestep_fluid(mesh, &this_run);
#endif /* __RADIATION_TRANFER__ */
  fprintf(this_run.proc_file,"# dtime = %14.6e\n", dtime);  

  output_diagnostics(mesh, &this_run, dtime);

  //  init_pad_region(&(this_run.pad), &this_run);

  while(this_run.tnow < this_run.tend) {

    //    update_pad_region(mesh, &this_run, &this_mpi);

#ifdef __GRAVITY__
    /* gravitational potential at t=t^{n} */
    zero_out_mesh_density(dens, &this_fftw_mpi);
    calc_mesh_density(dens, mesh, &this_run, &this_fftw_mpi);
    calc_mesh_grav_pot(dens, mesh, &this_run, &this_fftw_mpi, green_func);

    fluid_integrate(mesh, &this_run, &this_mpi, &this_fftw_mpi, green_func, dens, dtime);
#else /* !__GRAVITY__ */
    fluid_integrate(mesh, &this_run, &this_mpi, dtime);
#endif /* __GRAVITY__ */

#ifdef __RADIATION_TRANSFER__
#ifdef __USE_GPU__
    send_mesh_data(mesh, cuda_mem, &this_cuda, &this_run);
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
#else /* __DIFFUSE_RADIATION__ */
    step_radiation_tree(mesh, src, &this_run, &this_mpi, dtime);
#endif /* __DIFFUSE_RADIATION__ */
#endif /* __USE_GPU__ */
#endif /* __RADIATION_TRANSFER__ */

    this_run.tnow += dtime;
    this_run.step++;

#ifdef __RADIATION_TRANSFER__
#ifdef __USE_GPU__
    recv_mesh_data(mesh, cuda_mem, &this_cuda, &this_run);
#endif
#endif /* __RADIATION_TRANSFER__ */

    output_data_in_run(mesh, src, &this_run, this_run.model_name);
    output_diagnostics(mesh, &this_run, dtime);

#ifdef __RADIATION_TRANSFER__
#ifdef __USE_GPU__
    dtime = calc_timestep(mesh, cuda_mem, &this_cuda, &this_run);
#else
    dtime = calc_timestep(mesh, &this_run);
#endif
#else  /* !__RADIATION_TRANSFER__ */
    dtime = calc_timestep_fluid(mesh, &this_run);
    fprintf(this_run.proc_file, "# dtime = %14.6e\n", dtime);
#endif /* __RADIATION_TRANSFER__ */

    /* adjust output timing */
    float next_output_timing = this_run.output_timing[this_run.output_indx]*1.001;
    if(this_run.tnow+dtime > next_output_timing) {
      dtime = next_output_timing-this_run.tnow;
    }

  }

#ifdef __RADIATION_TRANSFER__
#ifdef __DIFFUSE_RADIATION__
  free_hd_param(&hd_param);
#ifdef __USE_GPU__
  free_diffuse_gpu(cd_param, &this_cuda);
#endif /* __USE_GPU__ */
#endif /* __DIFFUSE_RADIATION__ */
#endif /* __RADIATION_TRANSFER__ */

  MPI_Finalize();
  exit(EXIT_SUCCESS);

}
