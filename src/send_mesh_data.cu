#include <stdio.h>
#include <assert.h>

#include "fluid.h"
#include "cuda_mem_space.h"
#include "run_param.h"

extern "C"
void send_mesh_data(struct fluid_mesh *mesh, struct cuda_mem_space *cuda_mem, 
                    struct cuda_param *this_cuda, struct run_param *this_run)
{
  cudaError_t err;

  int idev;

  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    err = cudaMemcpyAsync(cuda_mem[idev].mesh_dev, mesh, 
			  sizeof(struct fluid_mesh)*NMESH_LOCAL, 
			  cudaMemcpyHostToDevice, this_cuda->strm[idev]);
    assert(err == cudaSuccess);
  }

  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaStreamSynchronize(this_cuda->strm[idev]);
  }

}
