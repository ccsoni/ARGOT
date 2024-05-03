#include <stdio.h>
#include <stdlib.h>

#include "diffuse_photon.h"
#include "cuda_mem_space.h"

#ifndef TINY
#define TINY (1.0e-31)
#endif

__global__ void calc_GH_tot_kernel(struct radiation_mesh*, const struct step_func_factor* __restrict__);
__global__ void cuda_sum_GH(struct radiation_mesh*, const struct radiation_mesh* __restrict__);
__global__ void cuda_calc_GH(const struct radiation_mesh* __restrict__, struct fluid_mesh*);

extern "C" void calc_GH(struct cuda_mem_space* , struct cuda_diffuse_param*, struct cuda_param*, cudaStream_t*);

extern "C"
void calc_GH_tot(struct cuda_mem_space *cuda_mem,
		 struct cuda_diffuse_param *cd_param,
		 cudaStream_t strm, int device_id)
{
  dim3 block(NMESH_LOCAL/NMESH_PER_BLOCK_DMESH, 1, 1);
  dim3 thread(NMESH_PER_BLOCK_DMESH, 1, 1);
  
  CUDA_SAFE( cudaSetDevice(device_id) );
  calc_GH_tot_kernel<<<block, thread, 0, strm>>>
    ( cd_param->rmesh, cd_param->step_fact );
}


__forceinline__ __device__ double calc_I_seg_GammaHI(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;
    
  I_seg_in_bar = rmesh->I_nu1/rmesh->absorption_nu1;

#ifdef __HELIUM__
  I_seg_in_bar += rmesh->I_nu2*RINT_J_HI_nu2/rmesh->absorption_nu2;
  I_seg_in_bar += rmesh->I_nu3*RINT_J_HI_nu3/rmesh->absorption_nu3;
#ifdef __HELIUM_BB__
  I_seg_in_bar += rmesh->I_nu4*RINT_J_HI_nu4/rmesh->absorption_nu4;
  I_seg_in_bar += rmesh->I_nu5*RINT_J_HI_nu5/rmesh->absorption_nu5;
  I_seg_in_bar += rmesh->I_nu6*RINT_J_HI_nu6/rmesh->absorption_nu6;
#endif //__HELIUM_BB__
#endif //__HELIUM__

  return I_seg_in_bar;
}

__forceinline__ __device__ double calc_I_seg_HeatHI(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;
    
  I_seg_in_bar = rmesh->I_nu1/rmesh->absorption_nu1;

#ifdef __HELIUM__
  I_seg_in_bar += rmesh->I_nu2*RINT_H_HI_nu2/rmesh->absorption_nu2;
  I_seg_in_bar += rmesh->I_nu3*RINT_H_HI_nu3/rmesh->absorption_nu3;
#ifdef __HELIUM_BB__
  I_seg_in_bar += rmesh->I_nu4*RINT_H_HI_nu4/rmesh->absorption_nu4;
  I_seg_in_bar += rmesh->I_nu5*RINT_H_HI_nu5/rmesh->absorption_nu5;
  I_seg_in_bar += rmesh->I_nu6*RINT_H_HI_nu6/rmesh->absorption_nu6;
#endif //__HELIUM_BB__
#endif //__HELIUM__

  return I_seg_in_bar;
}

__forceinline__ __device__ double calc_source_GammaHI(struct radiation_mesh *rmesh)
{
  double source_func;

  source_func = rmesh->source_func_nu1;

#ifdef __HELIUM__
  source_func += rmesh->source_func_nu2*RINT_J_HI_nu2;
  source_func += rmesh->source_func_nu3*RINT_J_HI_nu3;
#ifdef __HELIUM_BB__
  source_func += rmesh->source_func_nu4*RINT_J_HI_nu4;
  source_func += rmesh->source_func_nu5*RINT_J_HI_nu5;
  source_func += rmesh->source_func_nu6*RINT_J_HI_nu6;
#endif //__HELIUM_BB__
#endif //__HELIUM__

  return source_func;
}

__forceinline__ __device__ double calc_source_HeatHI(struct radiation_mesh *rmesh)
{
  double source_func;

  source_func = rmesh->source_func_nu1;

#ifdef __HELIUM__
  source_func += rmesh->source_func_nu2*RINT_H_HI_nu2;
  source_func += rmesh->source_func_nu3*RINT_H_HI_nu3;
#ifdef __HELIUM_BB__
  source_func += rmesh->source_func_nu4*RINT_H_HI_nu4;
  source_func += rmesh->source_func_nu5*RINT_H_HI_nu5;
  source_func += rmesh->source_func_nu6*RINT_H_HI_nu6;
#endif //__HELIUM_BB__
#endif //__HELIUM__

  return source_func;
}

#ifdef __HELIUM__
__forceinline__ __device__ double calc_I_seg_GammaHeI(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;
    
  I_seg_in_bar  = rmesh->I_nu2/rmesh->absorption_nu2;
  I_seg_in_bar += rmesh->I_nu3*RINT_J_HeI_nu3/rmesh->absorption_nu3;
#ifdef __HELIUM_BB__
  I_seg_in_bar += rmesh->I_nu6*RINT_J_HeI_nu6/rmesh->absorption_nu6;
#endif //__HELIUM_BB__

  return I_seg_in_bar;
}

__forceinline__ __device__ double calc_I_seg_HeatHeI(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;
    
  I_seg_in_bar  = rmesh->I_nu2/rmesh->absorption_nu2;
  I_seg_in_bar += rmesh->I_nu3*RINT_H_HeI_nu3/rmesh->absorption_nu3;
#ifdef __HELIUM_BB__
  I_seg_in_bar += rmesh->I_nu6*RINT_H_HeI_nu6/rmesh->absorption_nu6;
#endif //__HELIUM_BB__

  return I_seg_in_bar;
}

__forceinline__ __device__ double calc_source_GammaHeI(struct radiation_mesh *rmesh)
{
  double source_func;

  source_func  = rmesh->source_func_nu2;
  source_func += rmesh->source_func_nu3*RINT_J_HeI_nu3;
#ifdef __HELIUM_BB__
  source_func += rmesh->source_func_nu6*RINT_J_HeI_nu6;
#endif //__HELIUM_BB__

  return source_func;
}

__forceinline__ __device__ double calc_source_HeatHeI(struct radiation_mesh *rmesh)
{
  double source_func;

  source_func  = rmesh->source_func_nu2;
  source_func += rmesh->source_func_nu3*RINT_H_HeI_nu3;
#ifdef __HELIUM_BB__
  source_func += rmesh->source_func_nu6*RINT_H_HeI_nu6;
#endif //__HELIUM_BB__

  return source_func;
}


__forceinline__ __device__ double calc_I_seg_GammaHeII(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;   
  I_seg_in_bar = rmesh->I_nu3/rmesh->absorption_nu3;
  return I_seg_in_bar;
}

__forceinline__ __device__ double calc_I_seg_HeatHeII(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;
  I_seg_in_bar = rmesh->I_nu3/rmesh->absorption_nu3;
  return I_seg_in_bar;
}

__forceinline__ __device__ double calc_source_GammaHeII(struct radiation_mesh *rmesh)
{
  double source_func;
  source_func = rmesh->source_func_nu3;
  return source_func;
}

__forceinline__ __device__ double calc_source_HeatHeII(struct radiation_mesh *rmesh)
{
  double source_func;
  source_func = rmesh->source_func_nu3;
  return source_func;
}
#endif //__HELIUM__



__global__ void calc_GH_tot_kernel(struct radiation_mesh *rmesh, const struct step_func_factor* __restrict__ step_fact)
{
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  
  double length_tot;
  double Iseg_Gamma, Iseg_Heat;
  double source_Gamma, source_Heat;
  
  length_tot  = rmesh[ix].length;

  /* HI */
  Iseg_Gamma  = calc_I_seg_GammaHI(&rmesh[ix]);
  Iseg_Heat   = calc_I_seg_HeatHI(&rmesh[ix]);

  Iseg_Gamma /= length_tot;
  Iseg_Heat  /= length_tot;
    
  source_Gamma = calc_source_GammaHI(&rmesh[ix]);
  source_Heat  = calc_source_HeatHI(&rmesh[ix]);
    
  rmesh[ix].GHI_tot += (Iseg_Gamma + source_Gamma)*step_fact->HI[0];
  rmesh[ix].HHI_tot += (Iseg_Heat  + source_Heat) *step_fact->HI[1];

#ifdef __HELIUM__
  /* HeI */
  Iseg_Gamma = calc_I_seg_GammaHeI(&rmesh[ix]);
  Iseg_Heat  = calc_I_seg_HeatHeI(&rmesh[ix]);

  Iseg_Gamma /= length_tot;
  Iseg_Heat  /= length_tot;
    
  source_Gamma = calc_source_GammaHeI(&rmesh[ix]);
  source_Heat  = calc_source_HeatHeI(&rmesh[ix]);
    
  rmesh[ix].GHeI_tot += (Iseg_Gamma + source_Gamma)*step_fact->HeI[0];
  rmesh[ix].HHeI_tot += (Iseg_Heat  + source_Heat) *step_fact->HeI[1];

  /* HeII */
  Iseg_Gamma = calc_I_seg_GammaHeII(&rmesh[ix]);
  Iseg_Heat  = calc_I_seg_HeatHeII(&rmesh[ix]);

  Iseg_Gamma /= length_tot;
  Iseg_Heat /= length_tot;
    
  source_Gamma = calc_source_GammaHeII(&rmesh[ix]);
  source_Heat  = calc_source_HeatHeII(&rmesh[ix]);
    
  rmesh[ix].GHeII_tot += (Iseg_Gamma + source_Gamma)*step_fact->HeII[0];
  rmesh[ix].HHeII_tot += (Iseg_Heat  + source_Heat) *step_fact->HeII[1];
#endif //__HELIUM__
}


extern "C"
void calc_GH_sum(struct cuda_mem_space *cuda_mem,
		 struct cuda_diffuse_param *cd_param,
		 struct cuda_param *this_cuda)
{
  int idev;
  dim3 block, thread;
  int this_func_per_block = 1024;

  if(this_cuda->num_cuda_dev > 1) {
    block   = dim3(NMESH_LOCAL/this_func_per_block, 1, 1);
    thread  = dim3(this_func_per_block,1,1);

    for(idev=0; idev<this_cuda->num_cuda_dev; idev++){
      CUDA_SAFE( cudaSetDevice(idev) );
      CUDA_SAFE( cudaDeviceSynchronize() );
    }
      
    if(this_cuda->num_cuda_dev == 2) {

      CUDA_SAFE( cudaSetDevice(0) );
      cuda_sum_GH<<< block, thread, 0, this_cuda->strm[0] >>>
	(cd_param[0].rmesh, cd_param[1].rmesh);
      
      for(idev=0; idev<this_cuda->num_cuda_dev; idev++){
	CUDA_SAFE( cudaSetDevice(idev) );
	CUDA_SAFE( cudaDeviceSynchronize() );
      }
      
    }else if(this_cuda->num_cuda_dev == 4) {
      
      for(int cloop=0; cloop<2; cloop++) {
	
#pragma omp parallel sections num_threads(2)
	{
#pragma omp section
	  {
	    CUDA_SAFE( cudaSetDevice(0) );
	    cuda_sum_GH<<< block, thread, 0, this_cuda->strm[0] >>>
	      (cd_param[0].rmesh, cd_param[1].rmesh);
	  }      
#pragma omp section
	  {
	    CUDA_SAFE( cudaSetDevice(2) );
	    cuda_sum_GH<<< block, thread, 0, this_cuda->strm[2] >>>
	      (cd_param[2].rmesh, cd_param[3].rmesh);
	  }  
	} //omp sections
	
	for(idev=0; idev<this_cuda->num_cuda_dev; idev++){
	  CUDA_SAFE( cudaSetDevice(idev) );
	  CUDA_SAFE( cudaDeviceSynchronize() );
	}
	
	if(cloop==0) {

#pragma omp parallel sections num_threads(2)
	  {
#pragma omp section
	    {
	      CUDA_SAFE( cudaSetDevice(0) );
	      CUDA_SAFE( cudaMemcpyAsync( cd_param[3].rmesh, cd_param[0].rmesh,
					  sizeof(struct radiation_mesh)*NMESH_LOCAL,
					  cudaMemcpyDefault, this_cuda->strm[0]) );
	    }
#pragma omp section
	    {
	      CUDA_SAFE( cudaSetDevice(2) );
	      CUDA_SAFE( cudaMemcpyAsync( cd_param[1].rmesh, cd_param[2].rmesh,
					  sizeof(struct radiation_mesh)*NMESH_LOCAL,
					  cudaMemcpyDefault, this_cuda->strm[2]) );
	    }
	  } //omp sections
	  
	} else if(cloop==1) {

#pragma omp parallel sections num_threads(2)
	  {
#pragma omp section
	    {
	      CUDA_SAFE( cudaSetDevice(0) );
	      CUDA_SAFE( cudaMemcpyAsync( cd_param[1].rmesh, cd_param[0].rmesh,
					  sizeof(struct radiation_mesh)*NMESH_LOCAL,
					  cudaMemcpyDefault, this_cuda->strm[0]) );
	    }
#pragma omp section
	    {
	      CUDA_SAFE( cudaSetDevice(2) );
	      CUDA_SAFE( cudaMemcpyAsync( cd_param[3].rmesh, cd_param[2].rmesh,
					  sizeof(struct radiation_mesh)*NMESH_LOCAL,
					  cudaMemcpyDefault, this_cuda->strm[2]) );
	    }
	  } //omp sections
	} 
	

	for(idev=0; idev<this_cuda->num_cuda_dev; idev++){
	  CUDA_SAFE( cudaSetDevice(idev) );
	  CUDA_SAFE( cudaDeviceSynchronize() );
	}
	
      } //cloop end
    } // if(cuda dev == 4)
  }  // if(cuda dev > 1)
  
  
  calc_GH(cuda_mem, cd_param, this_cuda, this_cuda->strm);
}


__global__ void cuda_sum_GH(struct radiation_mesh *rmesh1,
			    const struct radiation_mesh* __restrict__ rmesh2)
{
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;

  rmesh1[ix].GHI_tot += rmesh2[ix].GHI_tot;
  rmesh1[ix].HHI_tot += rmesh2[ix].HHI_tot;
#ifdef __HELIUM__
  rmesh1[ix].GHeI_tot  += rmesh2[ix].GHeI_tot;
  rmesh1[ix].GHeII_tot += rmesh2[ix].GHeII_tot;
  rmesh1[ix].HHeI_tot  += rmesh2[ix].HHeI_tot;
  rmesh1[ix].HHeII_tot += rmesh2[ix].HHeII_tot;
#endif //__HELIUM__

}

extern "C"
void calc_GH(struct cuda_mem_space *cuda_mem,
	     struct cuda_diffuse_param *cd_param,
	     struct cuda_param *this_cuda,
	     cudaStream_t *strm)
{
  dim3 block(NMESH_LOCAL/NMESH_PER_BLOCK_DMESH, 1, 1);
  dim3 thread(NMESH_PER_BLOCK_DMESH, 1, 1);  
  int idev;
 
  for(idev=0; idev<this_cuda->num_cuda_dev; idev++){
    CUDA_SAFE( cudaSetDevice(idev) );
    CUDA_SAFE( cudaDeviceSynchronize() );
  }
  

#pragma omp parallel for num_threads(this_cuda->num_cuda_dev)
  for(idev=0; idev<this_cuda->num_cuda_dev; idev++){
    CUDA_SAFE( cudaSetDevice(idev) );
    cuda_calc_GH <<< block, thread, 0, strm[idev] >>>
      (cd_param[idev].rmesh, cuda_mem[idev].mesh_dev);
  }
  
  
  for(idev=0; idev<this_cuda->num_cuda_dev; idev++){
    CUDA_SAFE( cudaSetDevice(idev) );
    CUDA_SAFE( cudaDeviceSynchronize() );
  }
  
}
  
__global__ void cuda_calc_GH(const struct radiation_mesh* __restrict__ rmesh, struct fluid_mesh *mesh)
{
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  float r_n_ang = 1.0e0/N_ANG;
 
  ////Gamma_p += Gamma_d ,Heat_p += Heat_d 
  mesh[ix].prev_chem.GammaHI   += rmesh[ix].GHI_tot * r_n_ang;
  mesh[ix].prev_chem.HeatHI    += rmesh[ix].HHI_tot * r_n_ang; 
#ifdef __HELIUM__
  mesh[ix].prev_chem.GammaHeI  += rmesh[ix].GHeI_tot  * r_n_ang;
  mesh[ix].prev_chem.HeatHeI   += rmesh[ix].HHeI_tot  * r_n_ang;
  mesh[ix].prev_chem.GammaHeII += rmesh[ix].GHeII_tot * r_n_ang;
  mesh[ix].prev_chem.HeatHeII  += rmesh[ix].HHeII_tot * r_n_ang;
#endif //__HELIUM__
 }

