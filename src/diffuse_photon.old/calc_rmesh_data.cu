#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "diffuse_photon.h"
#include "cuda_mem_space.h"


#include "diffuse_chemistry.cu"

#ifndef TINY
#define TINY (1.0e-31)
#endif

extern "C" void merge_cuda_mem(struct cuda_mem_space*, struct cuda_param*, cudaStream_t*);

__global__ void zero_set_rmesh_kernel(struct radiation_mesh*);
__global__ void calc_rmesh_data_kernel(const struct fluid_mesh* __restrict__, struct radiation_mesh*, 
				       const struct run_param* __restrict__);


extern "C" 
void zero_set_rmesh(struct radiation_mesh *rmesh,
		    cudaStream_t strm, int device_id,struct ray_face *ray)
{
  int this_func_per_block = NMESH_PER_BLOCK_DMESH;

  dim3 block(NMESH_LOCAL/this_func_per_block,1,1);
  dim3 thread(this_func_per_block,1,1);
  
  CUDA_SAFE( cudaSetDevice(device_id) );
  zero_set_rmesh_kernel<<<block, thread, 0, strm>>>
    ( rmesh );

}


__global__ void zero_set_rmesh_kernel(struct radiation_mesh *rmesh)
{
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;

  rmesh[ix].length = 0.0e0;
  
  rmesh[ix].IHI    = 0.0e0;
#ifdef __HELIUM__
  rmesh[ix].IHeI   = 0.0e0;
  rmesh[ix].IHeII   = 0.0e0;
#endif //__HELIUM__
}


extern "C" 
void calc_rmesh_data(struct cuda_mem_space *cuda_mem,
		     struct cuda_diffuse_param *cd_param,
		     struct cuda_param *this_cuda)
{
  dim3 block(NMESH_LOCAL/NMESH_PER_BLOCK_DMESH,1,1);
  dim3 thread(NMESH_PER_BLOCK_DMESH,1,1);
  int idev;
  
  if(this_cuda->num_cuda_dev > 1) {
    merge_cuda_mem(cuda_mem, this_cuda, this_cuda->strm);
  }
  
  for(idev=0; idev<this_cuda->num_cuda_dev; idev++) {
    
    CUDA_SAFE( cudaSetDevice(idev) );
    calc_rmesh_data_kernel<<<block, thread, 0, this_cuda->strm[idev]>>>
      ( cuda_mem[idev].mesh_dev, cd_param[idev].rmesh,
	cuda_mem[idev].this_run_dev );
    
    CUDA_SAFE( cudaStreamSynchronize(this_cuda->strm[idev]) );
  }
  
}


__global__ void calc_rmesh_data_kernel(const struct fluid_mesh* __restrict__ mesh,
				       struct radiation_mesh *rmesh,
				       const struct run_param* __restrict__ this_run)
{
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;

  float csecHI   = csectHI_dev(HI_LYMAN_LIMIT+1.0e-7);      //13.6eV
#ifdef __HELIUM__
  float csecHeI  = csectHeI_dev(HeI_LYMAN_LIMIT+1.0e-7);    //24.6eV
  float csecHeII = csectHeII_dev(HeII_LYMAN_LIMIT+1.0e-7);  //54.4eV
#endif //__HELIUM__

  float wmol, temper, absorption;
  double emission; 
  double nH,nHI,nHII,ne;
    
  wmol   = WMOL(mesh[ix].prev_chem);
  temper = mesh[ix].prev_uene * this_run->uenetok * wmol;
  if(temper < 1.0) temper = 1.0;
  
  nH   = mesh[ix].dens * this_run->denstonh;
#ifdef __COSMOLOGICAL__
  nH  /= CUBE(this_run->anow); 
#endif

  nHI  = mesh[ix].prev_chem.fHI  * nH;
  nHII = mesh[ix].prev_chem.fHII * nH;
  ne   = nHII;
#ifdef __HELIUM__
  double nHeI,nHeII,nHeIII;
  nHeI  = mesh[ix].prev_chem.fHeI  * HELIUM_FACT*nH;
  nHeII = mesh[ix].prev_chem.fHeII * HELIUM_FACT*nH;
  nHeIII= mesh[ix].prev_chem.fHeIII* HELIUM_FACT*nH;
  
  ne += nHeII + 2.0e0*nHeIII;
#endif  

  emission = (k02_A(temper)-k02_B(temper))*ne*nHII * nuLeV*HI_LYMAN_LIMIT*eV_to_erg/(4.0*PI);
  absorption = csecHI * nHI +TINY;
  rmesh[ix].absorptionHI  = absorption;
  rmesh[ix].source_funcHI = emission / absorption;
  
#ifdef __HELIUM__
  emission = (k04_A(temper)-k04_B(temper))*ne*nHeII * nuLeV*HeI_LYMAN_LIMIT*eV_to_erg/(4.0*PI);
  absorption = csecHeI * nHeI +TINY;
  rmesh[ix].absorptionHeI  = absorption;
  rmesh[ix].source_funcHeI = emission / absorption;
  
  emission = (k06_A(temper)-k06_B(temper))*ne*nHeIII * nuLeV*HeII_LYMAN_LIMIT*eV_to_erg/(4.0*PI);
  absorption = csecHeII * nHeII +TINY;
  rmesh[ix].absorptionHeII  = absorption;
  rmesh[ix].source_funcHeII = emission / absorption;

#endif //__HELIUM__

  ///zero_set
  rmesh[ix].GHI_tot   = 0.0e0;  rmesh[ix].HHI_tot   = 0.0e0;
#ifdef __HELIUM__
  rmesh[ix].GHeI_tot  = 0.0e0;  rmesh[ix].HHeI_tot  = 0.0e0;
  rmesh[ix].GHeII_tot = 0.0e0;  rmesh[ix].HHeII_tot = 0.0e0;
#endif //__HELIUM__

}



extern "C"
void merge_cuda_mem(struct cuda_mem_space *cuda_mem, 
		struct cuda_param *this_cuda,
		cudaStream_t *strm)
{
  size_t mesh_s = sizeof(struct fluid_mesh)*NMESH_LOCAL/this_cuda->num_cuda_dev;
  
  int idev;
  for(idev=0; idev<this_cuda->num_cuda_dev; idev++){
    CUDA_SAFE( cudaSetDevice(idev) );
    CUDA_SAFE( cudaDeviceSynchronize() );
  }
  
  for(idev=0; idev<this_cuda->num_cuda_dev; idev++){
    
    long begin_address;
    int tid;   
    
    CUDA_SAFE( cudaSetDevice(idev) );
    
    begin_address = idev * (NMESH_LOCAL / this_cuda->num_cuda_dev);
    
    for(tid=0;tid<this_cuda->num_cuda_dev;tid++){
      if(idev==tid) continue;
      
      CUDA_SAFE( cudaMemcpyAsync(&cuda_mem[tid].mesh_dev[begin_address], &cuda_mem[idev].mesh_dev[begin_address],
				 mesh_s, cudaMemcpyDefault, strm[idev]) );
      
    }
    
  }
  
  for(idev=0; idev<this_cuda->num_cuda_dev; idev++){
    CUDA_SAFE( cudaSetDevice(idev) );
    CUDA_SAFE( cudaDeviceSynchronize() );
  }
  
}

