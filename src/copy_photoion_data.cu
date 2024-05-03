#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "cuda_mem_space.h"
#include "run_param.h"
#include "radiation.h"

__global__ void copy_photoion_data_kernel(struct fluid_mesh*, const struct photoion_rate* __restrict__);

#define RATE(ix,iy,iz) (rate[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])
#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

extern "C"
void copy_photoion_data(struct fluid_mesh *mesh, struct cuda_mem_space *cuda_mem,
			struct cuda_param *this_cuda, struct run_param *this_run)
{
  cudaError_t err;

  int idev;

  //  dim3 nblk(this_cuda->cuda_nblock, 1, 1);
  dim3 nblk(NMESH_LOCAL/NMESH_PER_BLOCK, 1, 1);
  dim3 nthd(NMESH_PER_BLOCK, 1, 1);

  static struct photoion_rate rate[NMESH_LOCAL];

  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	RATE(ix,iy,iz).GammaHI = MESH(ix,iy,iz).prev_chem.GammaHI;
#ifdef __HELIUM__	
	RATE(ix,iy,iz).GammaHeI  = MESH(ix,iy,iz).prev_chem.GammaHeI;
	RATE(ix,iy,iz).GammaHeII = MESH(ix,iy,iz).prev_chem.GammaHeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
	RATE(ix,iy,iz).GammaHM      = MESH(ix,iy,iz).prev_chem.GammaHM;
	RATE(ix,iy,iz).GammaH2I_I   = MESH(ix,iy,iz).prev_chem.GammaH2I_I;
	RATE(ix,iy,iz).GammaH2I_II  = MESH(ix,iy,iz).prev_chem.GammaH2I_II;
	RATE(ix,iy,iz).GammaH2II_I  = MESH(ix,iy,iz).prev_chem.GammaH2II_I;
	RATE(ix,iy,iz).GammaH2II_II = MESH(ix,iy,iz).prev_chem.GammaH2II_II;
#endif /* __HYDROGEN_MOL__ */
      }
    }
  }

  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    err = cudaMemcpyAsync(cuda_mem[idev].rate_dev, rate,
			  sizeof(struct photoion_rate)*NMESH_LOCAL,
			  cudaMemcpyHostToDevice, this_cuda->strm[idev]);
    assert(err == cudaSuccess);

    copy_photoion_data_kernel<<<nblk, nthd, 0, this_cuda->strm[idev]>>>(cuda_mem[idev].mesh_dev,
									cuda_mem[idev].rate_dev);
  }

  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaStreamSynchronize(this_cuda->strm[idev]);
  }

}

__global__ void copy_photoion_data_kernel(struct fluid_mesh *mesh, 
					  const struct photoion_rate* __restrict__ rate)
{
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

  mesh[tid].prev_chem.GammaHI = rate[tid].GammaHI;
#ifdef __HELIUM__
  mesh[tid].prev_chem.GammaHeI  = rate[tid].GammaHeI;
  mesh[tid].prev_chem.GammaHeII = rate[tid].GammaHeII;
#endif
#ifdef __HYDROGEN_MOL__
  mesh[tid].prev_chem.GammaHM     = rate[tid].GammaHM;
  mesh[tid].prev_chem.GammaH2I_I  = rate[tid].GammaH2I_I;
  mesh[tid].prev_chem.GammaH2I_II = rate[tid].GammaH2I_II;
  mesh[tid].prev_chem.GammaH2II_I = rate[tid].GammaH2II_I;
  mesh[tid].prev_chem.GammaH2II_II = rate[tid].GammaH2II_II;
#endif 

}
