#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <unistd.h>

#include "run_param.h"
#include "fluid.h"
#include "chemistry.h"
#include "source.h"
#include "cuda_mem_space.h"

extern "C"
void init_gpu(struct fluid_mesh *mesh, struct cuda_mem_space *cuda_mem,
	      struct cuda_param *this_cuda, struct run_param *this_run)
{
  cudaError_t err;

  int device_count;
  static struct cudaDeviceProp prop[NMAX_CUDA_DEV];

  this_cuda->num_cuda_dev=4;

  err = cudaGetDeviceCount(&device_count); assert(err == cudaSuccess);
  if(this_cuda->num_cuda_dev > device_count || device_count > NMAX_CUDA_DEV) {
    fprintf(stderr,"Inconsistent # of GPUs\n");
    fprintf(stderr,"this_cuda->num_cuda_dev=%d, device_count=%d, NMAX_CUDA_DEV=%d\n",
	   this_cuda->num_cuda_dev, device_count, NMAX_CUDA_DEV);
    char hostname[256];
    gethostname(hostname,sizeof(hostname));
    fprintf(stderr, "hostname: %s\n", hostname);

    fflush(stderr);
    
    exit(EXIT_FAILURE);
  }

  fprintf(this_run->proc_file,
	  "# of devices : %d \n", device_count);
  fprintf(this_run->proc_file, 
	  "# of devices in use : %d \n", this_cuda->num_cuda_dev);
  fprintf(this_run->proc_file, 
	  "# NMAX_CUDA_DEV : %d \n", NMAX_CUDA_DEV);
  fprintf(this_run->proc_file, 
	  "# NMESH_PER_BLOCK : %d \n", NMESH_PER_BLOCK);

  this_cuda->cuda_nblock = NMESH_LOCAL/NMESH_PER_BLOCK/(this_cuda->num_cuda_dev);
  fprintf(this_run->proc_file, 
	  "# NBLOCK : %d \n", this_cuda->cuda_nblock);
  this_cuda->nmesh_per_dev = NMESH_LOCAL/(this_cuda->num_cuda_dev);
  fprintf(this_run->proc_file, 
	  "# NMESH_PER_DEV : %d \n", this_cuda->nmesh_per_dev);
  fflush(this_run->proc_file);

  uint64_t alloc_dev_mem;
  
  int idev;
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    err = cudaGetDeviceProperties(&prop[idev],idev);

    fprintf(this_run->proc_file, 
	    "# Max Number of Threads Per Block    :: %d\n",
	    prop[idev].maxThreadsPerBlock);
    fprintf(this_run->proc_file,
	    "# Max Diemsnion of Threads Per Block :: %d x %d x %d\n",
	    prop[idev].maxThreadsDim[0], 
	    prop[idev].maxThreadsDim[1],
	    prop[idev].maxThreadsDim[2]);
    fprintf(this_run->proc_file,
	    "# Max Diemsnion of Blocks Per Grid   :: %d x %d x %d\n",
	    prop[idev].maxGridSize[0], 
	    prop[idev].maxGridSize[1],
	    prop[idev].maxGridSize[2]);

    this_cuda->max_thread_dimx = prop[idev].maxThreadsDim[0];
    this_cuda->max_block_dimx = prop[idev].maxGridSize[0];

    alloc_dev_mem = 0;

    cudaSetDevice(idev);
    err = cudaMalloc((void **) &(cuda_mem[idev].mesh_dev),
		     sizeof(struct fluid_mesh)*NMESH_LOCAL);
    assert(err == cudaSuccess);
    alloc_dev_mem += sizeof(struct fluid_mesh)*NMESH_LOCAL;
    
    err = cudaMalloc((void **) &(cuda_mem[idev].this_run_dev), 
		     sizeof(struct run_param));
    assert(err == cudaSuccess);
    alloc_dev_mem += sizeof(struct run_param);

    err = cudaMalloc((void **) &(cuda_mem[idev].rate_dev),
		     sizeof(struct photoion_rate)*NMESH_LOCAL);
    assert(err == cudaSuccess);
    alloc_dev_mem += sizeof(struct photoion_rate)*NMESH_LOCAL;

    err = cudaMalloc((void **) &(cuda_mem[idev].chem_updated), 
                     sizeof(struct prim_chem)*NMESH_LOCAL);
    assert(err == cudaSuccess);
    alloc_dev_mem += sizeof(struct prim_chem)*NMESH_LOCAL;

    err = cudaMalloc((void **) &(cuda_mem[idev].dtime_dev),
                     sizeof(float)*this_cuda->cuda_nblock);
    assert(err == cudaSuccess);
    alloc_dev_mem += sizeof(float)*this_cuda->cuda_nblock;

    err = cudaMalloc((void **) &(cuda_mem[idev].diff_chem_dev),
                     sizeof(float)*this_cuda->cuda_nblock);
    assert(err == cudaSuccess);
    alloc_dev_mem += sizeof(float)*this_cuda->cuda_nblock;

    err = cudaMalloc((void **) &(cuda_mem[idev].diff_uene_dev),
                     sizeof(float)*this_cuda->cuda_nblock);
    assert(err == cudaSuccess);
    alloc_dev_mem += sizeof(float)*this_cuda->cuda_nblock;

    fprintf(this_run->proc_file, 
            "# Allocated device memory on the device %d :: %llu [MByte]\n", 
            idev, (alloc_dev_mem>>20));
    fflush(this_run->proc_file);
  }

  /* Cache Config */
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    assert(err == cudaSuccess);
  }

  /* Send the run_param structure */
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    err = cudaMemcpy(cuda_mem[idev].this_run_dev, this_run,
		     sizeof(struct run_param), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
  }

  /* Creating CUDA streams */
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    err = cudaStreamCreate(&(this_cuda->strm[idev]));
    assert(err == cudaSuccess);
    err = cudaStreamCreate(&(this_cuda->diffuse_strm[idev]));
    assert(err == cudaSuccess);
  }
}
