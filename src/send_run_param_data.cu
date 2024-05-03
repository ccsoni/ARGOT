#include <stdio.h>
#include <assert.h>

#include "cuda_mem_space.h"
#include "run_param.h"

extern "C"
void send_run_param_data(struct run_param *this_run, 
			 struct cuda_mem_space *cuda_mem,
			 struct cuda_param *this_cuda)
{
  cudaError_t err;

  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    err = cudaMemcpy(cuda_mem[idev].this_run_dev, this_run,
                     sizeof(struct run_param), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
  }

}
