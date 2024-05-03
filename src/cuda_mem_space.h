#ifndef __ARGOT_CUDA_MEM_SPACE__
#define __ARGOT_CUDA_MEM_SPACE__

#include "run_param.h"
#include "fluid.h"
#include "radiation.h"
#include "source.h"

// size of cudaStream_t is 64-bit.
#ifndef __CUDACC__
#define cudaStream_t int64_t 
#endif

struct cuda_param {
  int num_cuda_dev;
  int nmesh_per_dev;
  int cuda_nblock;

  int max_thread_dimx;
  int max_block_dimx;

  cudaStream_t strm[NMAX_CUDA_DEV];
  cudaStream_t diffuse_strm[NMAX_CUDA_DEV];
};

struct cuda_mem_space {
  struct fluid_mesh *mesh_dev;
  struct run_param  *this_run_dev;
  struct freq_param *freq_dev;
  struct ray_segment *segment_dev;
  struct photoion_rate *rate_dev;
  struct prim_chem *chem_updated;
  struct light_ray_IO *ray_IO_dev;

  float *optical_depth_dev;
  float *dtime_dev;
  float *diff_chem_dev;
  float *diff_uene_dev;
};

#endif /* __ARGOT_CUDA_MEM_SPACE__ */
