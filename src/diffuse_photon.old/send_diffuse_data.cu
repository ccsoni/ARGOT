#include <stdio.h>
#include <stdlib.h>

#include "diffuse_photon.h"
#include "cuda_mem_space.h"

extern "C" 
void send_diffuse_data(struct host_diffuse_param *hd_param,
		       struct cuda_diffuse_param *cd_param, struct cuda_param *this_cuda)
{
  static cudaStream_t strm[NMAX_CUDA_DEV];
  int idev;

  /* Creating CUDA streams */
  for(idev=0; idev<this_cuda->num_cuda_dev; idev++) {
    CUDA_SAFE( cudaSetDevice(idev) );
    CUDA_SAFE( cudaStreamCreate(&(strm[idev])) );
  }

  
  for(idev=0; idev < this_cuda->num_cuda_dev; idev++){
    CUDA_SAFE( cudaSetDevice(idev) );
    
    CUDA_SAFE( cudaMemcpyAsync( cd_param[idev].step_fact, hd_param->step_fact, 
				sizeof(struct step_func_factor), 
				cudaMemcpyDefault, strm[idev]) );

    CUDA_SAFE( cudaMemcpyAsync( cd_param[idev].angle, hd_param->angle, 
				sizeof(struct angle_info)*N_ANG, 
				cudaMemcpyDefault, strm[idev]) );
  }
  

  /* Destroy CUDA streams */
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    CUDA_SAFE( cudaSetDevice(idev) );
    CUDA_SAFE( cudaStreamDestroy(strm[idev]) );
  }
  
 
}

