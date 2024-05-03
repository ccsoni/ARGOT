#include <stdio.h>
#include <stdlib.h>

#include "diffuse_photon.h"
#include "cuda_mem_space.h"

extern "C" 
void free_diffuse_gpu(struct cuda_diffuse_param *cd_param, struct cuda_param *this_cuda)
{
  int did;
  for(did=0; did < this_cuda->num_cuda_dev; did++){
    CUDA_SAFE( cudaSetDevice(did) );
    
    CUDA_SAFE( cudaFree(cd_param[did].step_fact) );
    CUDA_SAFE( cudaFree(cd_param[did].angle) );
    CUDA_SAFE( cudaFree(cd_param[did].rmesh) );
 
  }

#if 0
  if(this_cuda->num_cuda_dev==2) { 
    CUDA_SAFE( cudaSetDevice(0) );
    CUDA_SAFE( cudaDeviceDisablePeerAccess(1) );
    CUDA_SAFE( cudaSetDevice(1) );
    CUDA_SAFE( cudaDeviceDisablePeerAccess(0) );    
  }

  if(this_cuda->num_cuda_dev==4) { 
    CUDA_SAFE( cudaSetDevice(0) );
    CUDA_SAFE( cudaDeviceDisablePeerAccess(1) );     
    CUDA_SAFE( cudaSetDevice(1) );
    CUDA_SAFE( cudaDeviceDisablePeerAccess(0) );    

    CUDA_SAFE( cudaSetDevice(2) );
    CUDA_SAFE( cudaDeviceDisablePeerAccess(3) );     
    CUDA_SAFE( cudaSetDevice(3) );
    CUDA_SAFE( cudaDeviceDisablePeerAccess(2) ); 
  }
#endif
}

