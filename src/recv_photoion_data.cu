#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "cuda_mem_space.h"
#include "run_param.h"
#include "radiation.h"

__global__ void recv_photoion_data_kernel(struct fluid_mesh*, 
					  struct photoion_rate*,
					  uint64_t);

extern "C"
void recv_photoion_data(struct photoion_rate *gamma,
			struct cuda_mem_space *cuda_mem,
			struct cuda_param *this_cuda,
			struct run_param *this_run)
{
  static uint64_t offset[NMAX_CUDA_DEV];
  cudaError_t err;

  dim3 nblk(this_cuda->cuda_nblock, 1, 1);
  dim3 nthd(NMESH_PER_BLOCK, 1, 1);

  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    offset[idev] = idev*this_cuda->cuda_nblock*NMESH_PER_BLOCK;
    cudaSetDevice(idev);

    recv_photoion_data_kernel<<<nblk, nthd, 0, this_cuda->strm[idev]>>>
      (cuda_mem[idev].mesh_dev, cuda_mem[idev].rate_dev, offset[idev]);

    err = cudaMemcpyAsync(gamma+offset[idev], 
			  cuda_mem[idev].rate_dev+offset[idev],
			  sizeof(struct photoion_rate)*this_cuda->nmesh_per_dev,
			  cudaMemcpyDeviceToHost, this_cuda->strm[idev]);
    assert(err == cudaSuccess);
  }

  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaStreamSynchronize(this_cuda->strm[idev]);
  }
}

__global__ void recv_photoion_data_kernel(struct fluid_mesh *mesh,
					  struct photoion_rate *gamma,
					  uint64_t offset)
{
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x + offset;
  
  gamma[tid].GammaHI = mesh[tid].prev_chem.GammaHI;
  gamma[tid].HeatHI  = mesh[tid].prev_chem.HeatHI;
#ifdef __HELIUM__
  gamma[tid].GammaHeI  = mesh[tid].prev_chem.GammaHeI;
  gamma[tid].GammaHeII = mesh[tid].prev_chem.GammaHeII;
  gamma[tid].HeatHeI  = mesh[tid].prev_chem.HeatHeI;
  gamma[tid].HeatHeII = mesh[tid].prev_chem.HeatHeII;
#endif
#ifdef __HYDROGEN_MOL__
  gamma[tid].GammaHM      = mesh[tid].prev_chem.GammaHM;
  gamma[tid].GammaH2I_I   = mesh[tid].prev_chem.GammaH2I_I;
  gamma[tid].GammaH2I_II  = mesh[tid].prev_chem.GammaH2I_II;
  gamma[tid].GammaH2II_I  = mesh[tid].prev_chem.GammaH2II_I;
  gamma[tid].GammaH2II_II = mesh[tid].prev_chem.GammaH2II_II;

  gamma[tid].HeatHM       = mesh[tid].prev_chem.HeatHM;
  gamma[tid].HeatH2I_I    = mesh[tid].prev_chem.HeatH2I_I;
  gamma[tid].HeatH2I_II   = mesh[tid].prev_chem.HeatH2I_II;
  gamma[tid].HeatH2II_I   = mesh[tid].prev_chem.HeatH2II_I;
  gamma[tid].HeatH2II_II  = mesh[tid].prev_chem.HeatH2II_II;
#endif
}
