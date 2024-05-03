#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>

#include "fluid.h"
#include "cuda_mem_space.h"
#include "run_param.h"


extern "C" int start_timing(struct timeval*, struct tms*);
extern "C" int end_timing(struct timeval*, struct timeval*, struct tms*, struct tms*, const char*, struct run_param*);

__global__ void merge_photoion_data_kernel(struct fluid_mesh*, struct photoion_rate*, uint64_t);

extern "C"
void merge_photoion_data(struct fluid_mesh *mesh, struct cuda_mem_space *cuda_mem, 
			 struct cuda_param *this_cuda, struct run_param *this_run)
{
  cudaError_t err;

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

  start_timing(&start_tv, &start_tms);
  
  struct photoion_rate *photoion;
  photoion = (struct photoion_rate*) malloc (sizeof(struct photoion_rate)*NMESH_LOCAL);

#pragma omp parallel for schedule(auto)
  for(uint64_t imesh=0;imesh<NMESH_LOCAL;imesh++) {
    photoion[imesh].GammaHI = mesh[imesh].prev_chem.GammaHI;
    photoion[imesh].HeatHI  = mesh[imesh].prev_chem.HeatHI;
#ifdef __HELIUM__
    photoion[imesh].GammaHeI  = mesh[imesh].prev_chem.GammaHeI;
    photoion[imesh].GammaHeII = mesh[imesh].prev_chem.GammaHeII;
    photoion[imesh].HeatHeI   = mesh[imesh].prev_chem.HeatHeI;
    photoion[imesh].HeatHeII  = mesh[imesh].prev_chem.HeatHeII;
#endif
#ifdef __HYDROGEN_MOL__
    photoion[imesh].GammaHM      = mesh[imesh].prev_chem.GammaHM;
    photoion[imesh].GammaH2I_I   = mesh[imesh].prev_chem.GammaH2I_I;
    photoion[imesh].GammaH2I_II  = mesh[imesh].prev_chem.GammaH2I_II;
    photoion[imesh].GammaH2II_I  = mesh[imesh].prev_chem.GammaH2II_I;
    photoion[imesh].GammaH2II_II = mesh[imesh].prev_chem.GammaH2II_II;

    photoion[imesh].HeatHM      = mesh[imesh].prev_chem.HeatHM;
    photoion[imesh].HeatH2I_I   = mesh[imesh].prev_chem.HeatH2I_I;
    photoion[imesh].HeatH2I_II  = mesh[imesh].prev_chem.HeatH2I_II;
    photoion[imesh].HeatH2II_I  = mesh[imesh].prev_chem.HeatH2II_I;
    photoion[imesh].HeatH2II_II = mesh[imesh].prev_chem.HeatH2II_II;
#endif
  }



  dim3 nblk(this_cuda->cuda_nblock, 1, 1);
  dim3 nthd(NMESH_PER_BLOCK, 1, 1);
  
  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    uint64_t offset = idev*this_cuda->cuda_nblock*NMESH_PER_BLOCK;

    cudaSetDevice(idev);
    err = cudaMemcpyAsync(cuda_mem[idev].rate_dev, photoion, 
			  sizeof(struct photoion_rate)*NMESH_LOCAL,
			  cudaMemcpyHostToDevice, this_cuda->strm[idev]);
    assert(err == cudaSuccess);

    merge_photoion_data_kernel<<<nblk, nthd, 0, this_cuda->strm[idev]>>>
      (cuda_mem[idev].mesh_dev, cuda_mem[idev].rate_dev, offset); 
  }


  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaStreamSynchronize(this_cuda->strm[idev]);
  }

  free(photoion);

  end_timing(&start_tv, &end_tv, &start_tms, &end_tms,
	     "merge_photoion_data", this_run);
}


__global__ void merge_photoion_data_kernel(struct fluid_mesh *mesh, 
					   struct photoion_rate *photoion, 
					   uint64_t offset)
{
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x + offset;

  /* ART.photoion += ARGOT.photoion on GPU */
  mesh[tid].prev_chem.GammaHI += photoion[tid].GammaHI;
  mesh[tid].prev_chem.HeatHI  += photoion[tid].HeatHI;
#ifdef __HELIUM__
  mesh[tid].prev_chem.GammaHeI  += photoion[tid].GammaHeI;
  mesh[tid].prev_chem.GammaHeII += photoion[tid].GammaHeII;
  mesh[tid].prev_chem.HeatHeI   += photoion[tid].HeatHeI;
  mesh[tid].prev_chem.HeatHeII  += photoion[tid].HeatHeII;
#endif
#ifdef __HYDROGEN_MOL__
  mesh[tid].prev_chem.GammaHM      += photoion[tid].GammaHM;
  mesh[tid].prev_chem.GammaH2I_I   += photoion[tid].GammaH2I_I;
  mesh[tid].prev_chem.GammaH2I_II  += photoion[tid].GammaH2I_II;
  mesh[tid].prev_chem.GammaH2II_I  += photoion[tid].GammaH2II_I;
  mesh[tid].prev_chem.GammaH2II_II += photoion[tid].GammaH2II_II;

  mesh[tid].prev_chem.HeatHM       += photoion[tid].HeatHM;
  mesh[tid].prev_chem.HeatH2I_I    += photoion[tid].HeatH2I_I;
  mesh[tid].prev_chem.HeatH2I_II   += photoion[tid].HeatH2I_II;
  mesh[tid].prev_chem.HeatH2II_I   += photoion[tid].HeatH2II_I;
  mesh[tid].prev_chem.HeatH2II_II  += photoion[tid].HeatH2II_II;
#endif 
}
