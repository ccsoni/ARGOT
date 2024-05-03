#include <stdio.h>
#include <stdlib.h>

#include "diffuse_photon.h"
#include "cuda_mem_space.h"

extern "C" 
void init_diffuse_gpu(struct cuda_diffuse_param *cd_param, 
		      struct cuda_param *this_cuda, struct run_param *this_run)
{
  int idev;
  long alloc_dev_mem;
  long temp_dev_mem;
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    alloc_dev_mem = 0;
    
    CUDA_SAFE( cudaSetDevice(idev) );
    
    CUDA_SAFE( cudaMalloc((void **) &(cd_param[idev].step_fact),
			  sizeof(struct step_func_factor)) );
    alloc_dev_mem += sizeof(struct step_func_factor);
    
    
    CUDA_SAFE( cudaMalloc((void **) &(cd_param[idev].angle),
			  sizeof(struct angle_info)*N_ANG) );
    alloc_dev_mem += sizeof(struct angle_info)*N_ANG;

    
    CUDA_SAFE( cudaMalloc((void **) &(cd_param[idev].rmesh),
			  sizeof(struct radiation_mesh)*NMESH_LOCAL) );
    alloc_dev_mem += sizeof(struct radiation_mesh)*NMESH_LOCAL;
    
    temp_dev_mem = 0;
    temp_dev_mem += sizeof(struct ray_info)*NMESH_MAX_FACE*3;  // 3:xy,yz,zx
    
    fprintf(this_run->proc_file, 
            "# Allocated global memory for diffuse photon on the device %d     :: %llu [MByte]\n", 
            idev, ((alloc_dev_mem+temp_dev_mem)>>20));

    fflush(this_run->proc_file);
  }


  //peeraccess 0to1 , 1to0 , 2to3 , 3to2
  //Second argument is always 0 in CUDA4.0 .
#if 1
  if(this_cuda->num_cuda_dev==2) { 
    CUDA_SAFE( cudaSetDevice(0) );
    CUDA_SAFE( cudaDeviceEnablePeerAccess(1, 0) ); //0->1    
    CUDA_SAFE( cudaSetDevice(1) );
    CUDA_SAFE( cudaDeviceEnablePeerAccess(0, 0) ); //1->0   
  }

  if(this_cuda->num_cuda_dev==4) { 
    CUDA_SAFE( cudaSetDevice(0) );
    CUDA_SAFE( cudaDeviceEnablePeerAccess(1, 0) ); //0->1    
    CUDA_SAFE( cudaSetDevice(1) );
    CUDA_SAFE( cudaDeviceEnablePeerAccess(0, 0) ); //1->0   

    CUDA_SAFE( cudaSetDevice(2) );
    CUDA_SAFE( cudaDeviceEnablePeerAccess(3, 0) ); //2->3    
    CUDA_SAFE( cudaSetDevice(3) );
    CUDA_SAFE( cudaDeviceEnablePeerAccess(2, 0) ); //3->2
  }
#endif

}

