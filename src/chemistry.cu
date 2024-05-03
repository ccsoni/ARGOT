#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <float.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include "run_param.h"
#include "fluid.h"
#include "cuda_mem_space.h"

#include "reaction_rate.cu"
#include "step_reaction.cu"
#include "step_heatcool.cu"

#define __USE_THRUST__

__shared__ float dtime_shared[NMESH_PER_BLOCK+1];
__shared__ float diff_shared[NMESH_PER_BLOCK+1];

__global__ void calc_dtime_gpu_kernel(struct fluid_mesh*, struct run_param*, 
				      float*, uint64_t);
__global__ void advance_reaction_kernel(struct fluid_mesh*, struct prim_chem*, 
					struct run_param*, float, float*, uint64_t);
__global__ void advance_heatcool_kernel(struct fluid_mesh*, const struct prim_chem* __restrict__, 
					struct run_param*, float, float*, uint64_t);
__global__ void copy_chemistry_kernel(struct fluid_mesh*, const struct prim_chem* __restrict__);
__global__ void update_chemistry_kernel(struct fluid_mesh*, uint64_t);
__global__ void advance_reaction_and_heatcool_kernel(struct fluid_mesh*, struct prim_chem*,
						     struct run_param*, float, float*, float*, uint64_t);


extern "C"
void update_chemistry_gpu(struct cuda_mem_space *cuda_mem, 
			  struct cuda_param *this_cuda,
			  struct run_param *this_run)
{

  static uint64_t offset[NMAX_CUDA_DEV];

  cudaError_t err;

  dim3 nblk(this_cuda->cuda_nblock, 1, 1);
  dim3 nthd(NMESH_PER_BLOCK, 1, 1);

  int idev;

  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    offset[idev] = idev*this_cuda->cuda_nblock*NMESH_PER_BLOCK;
    cudaSetDevice(idev);
    update_chemistry_kernel<<<nblk, nthd, 0, this_cuda->strm[idev]>>>
      (cuda_mem[idev].mesh_dev, offset[idev]);
  }

}

extern "C"
double calc_dtime_gpu(struct cuda_mem_space *cuda_mem, 
		      struct cuda_param *this_cuda,
		      struct run_param *this_run)
{
  static uint64_t offset[NMAX_CUDA_DEV];

  cudaError_t err;

  dim3 nblk(this_cuda->cuda_nblock, 1, 1);
  dim3 nthd(NMESH_PER_BLOCK, 1, 1);

  int idev;

  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    offset[idev] = idev*this_cuda->cuda_nblock*NMESH_PER_BLOCK;
    cudaSetDevice(idev);
    calc_dtime_gpu_kernel <<<nblk, nthd, 0, this_cuda->strm[idev]>>> 
      (cuda_mem[idev].mesh_dev, cuda_mem[idev].this_run_dev, 
       cuda_mem[idev].dtime_dev, offset[idev]);
  }

  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaStreamSynchronize(this_cuda->strm[idev]);
  }
  
  static float dtime[NMAX_CUDA_DEV];
  float dtmin = FLT_MAX;
#ifdef __USE_THRUST__
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    thrust::device_ptr<float> dtime_dev_ptr(cuda_mem[idev].dtime_dev);
    dtime[idev] = thrust::reduce(dtime_dev_ptr, 
				 dtime_dev_ptr+this_cuda->cuda_nblock,
				 FLT_MAX,
				 thrust::minimum<float>());
    fprintf(this_run->proc_file, 
	    "# device id = %d /  dtmin : %14.6e\n", idev, dtime[idev]/this_run->tunit);
    dtmin = MIN(dtmin, dtime[idev]);
  }
#else
  float *dt_cpu;
  dt_cpu = (float *) malloc(sizeof(float)*this_cuda->cuda_nblock);
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaMemcpyAsync(dt_cpu, cuda_mem[idev].dtime_dev,
		    sizeof(float)*this_cuda->cuda_nblock,
		    cudaMemcpyDeviceToHost,this_cuda->strm[idev]);
  }
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaStreamSynchronize(this_cuda->strm[idev]);
    dtime[idev] = FLT_MAX;
    for(int iblck=0;iblck<this_cuda->cuda_nblock;iblck++) {
      dtime[idev] = MIN(dtime[idev], dt_cpu[iblck]);
    }
    fprintf(this_run->proc_file, 
	    "# device id = %d /  dtmin : %14.6e\n", idev, dtime[idev]/this_run->tunit);
    dtmin = MIN(dtmin, dtime[idev]);
  }
  free(dt_cpu);
#endif

  return dtmin;
}

extern "C"
void step_chemistry_gpu(struct cuda_mem_space *cuda_mem, 
			struct cuda_param *this_cuda,
			struct run_param *this_run,
			float *diff_chem,
			float *diff_uene,
			float dtime)
{
  static uint64_t offset[NMAX_CUDA_DEV];

  static bool first_call=true;
  static struct prim_chem *chem_all;

  cudaError_t err;

  dim3 nblk(this_cuda->cuda_nblock, 1, 1);
  dim3 nblk_all(NMESH_LOCAL/NMESH_PER_BLOCK, 1, 1);
  dim3 nthd(NMESH_PER_BLOCK, 1, 1);

  int idev;

  if(first_call && this_cuda->num_cuda_dev > 1) {
    err = cudaMallocHost((void **) &chem_all, 
                         sizeof(struct prim_chem)*NMESH_LOCAL);
    assert(err == cudaSuccess);
    first_call = false;
  }

  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    offset[idev] = idev*this_cuda->cuda_nblock*NMESH_PER_BLOCK;
    cudaSetDevice(idev);

#ifdef __HEATCOOL__
    advance_reaction_and_heatcool_kernel<<<nblk, nthd, 0, this_cuda->strm[idev]>>>
      (cuda_mem[idev].mesh_dev, cuda_mem[idev].chem_updated,
       cuda_mem[idev].this_run_dev, dtime, 
       cuda_mem[idev].diff_chem_dev, cuda_mem[idev].diff_uene_dev, 
       offset[idev]);
#else /* !__HEATCOOL__ */
    advance_reaction_kernel<<<nblk, nthd, 0, this_cuda->strm[idev]>>>
      (cuda_mem[idev].mesh_dev, cuda_mem[idev].chem_updated,
       cuda_mem[idev].this_run_dev, dtime, 
       cuda_mem[idev].diff_chem_dev, offset[idev]);
#endif /* __HEATCOOL__ */

    if(this_cuda->num_cuda_dev > 1) {
      cudaMemcpyAsync(chem_all+offset[idev], 
		      cuda_mem[idev].chem_updated+offset[idev],
		      sizeof(struct prim_chem)*this_cuda->nmesh_per_dev,
		      cudaMemcpyDeviceToHost, this_cuda->strm[idev]);
    }
  }

  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaStreamSynchronize(this_cuda->strm[idev]);
  }

  static float diff[NMAX_CUDA_DEV];
  *diff_chem = FLT_MIN;
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    thrust::device_ptr<float> diff_chem_dev_ptr(cuda_mem[idev].diff_chem_dev);
    diff[idev] = thrust::reduce(diff_chem_dev_ptr, 
				diff_chem_dev_ptr+this_cuda->cuda_nblock,
				0.0f,
				thrust::maximum<float>());
    *diff_chem = MAX(*diff_chem, diff[idev]);
  }

  *diff_uene = FLT_MIN;
  for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    thrust::device_ptr<float> diff_uene_dev_ptr(cuda_mem[idev].diff_uene_dev);
    diff[idev] = thrust::reduce(diff_uene_dev_ptr,
				diff_uene_dev_ptr+this_cuda->cuda_nblock,
				0.0f,
				thrust::maximum<float>());
    *diff_uene = MAX(*diff_uene, diff[idev]);
  }

  if(this_cuda->num_cuda_dev > 1) {
    for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
      cudaSetDevice(idev);
      cudaMemcpyAsync(cuda_mem[idev].chem_updated, chem_all,
		      sizeof(struct prim_chem)*NMESH_LOCAL,cudaMemcpyHostToDevice,
		      this_cuda->strm[idev]);
      copy_chemistry_kernel <<<nblk_all, nthd, 0, this_cuda->strm[idev]>>> 
	(cuda_mem[idev].mesh_dev, cuda_mem[idev].chem_updated);
    }
    
    for(idev=0;idev<this_cuda->num_cuda_dev;idev++) {
      cudaSetDevice(idev);
      cudaStreamSynchronize(this_cuda->strm[idev]);
    }
  }

  //  return diffmax;
}

__global__ void update_chemistry_kernel(struct fluid_mesh *mesh, uint64_t offset) 
{
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x + offset;

  mesh[tid].prev_chem = mesh[tid].chem;
  mesh[tid].prev_uene = mesh[tid].uene;
  mesh[tid].eneg = mesh[tid].uene*mesh[tid].dens
    + 0.5*(SQR(mesh[tid].momx)+
	   SQR(mesh[tid].momy)+
	   SQR(mesh[tid].momz))/mesh[tid].dens;
}


__global__ void copy_chemistry_kernel(struct fluid_mesh *mesh, 
				      const struct prim_chem* __restrict__ chem)
{
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  
  mesh[tid].chem = chem[tid];
}


__global__ void calc_dtime_gpu_kernel(struct fluid_mesh *mesh, 
				      struct run_param *this_run,
				      float *dtime, uint64_t offset)
{
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x + offset;
  unsigned int local_tid = threadIdx.x;
  
  struct fluid_mesh *target_mesh; /* pointer to target cells */

  target_mesh = &mesh[tid];

  dtime_shared[local_tid] = 
    calc_dtime_dev(target_mesh, &(target_mesh->prev_chem), this_run);

  __syncthreads();

  for(unsigned int s=blockDim.x/2; s>0;s>>=1) {
    if (local_tid < s) {
      dtime_shared[local_tid] = MIN(dtime_shared[local_tid+s],
                                    dtime_shared[local_tid]);
    }
    __syncthreads();
  }

  dtime[blockIdx.x] = dtime_shared[0];
}

__device__ float diff_chem(struct prim_chem *chem1, struct prim_chem *chem2)
{
  float diff_H, diff_elec;

#if 1
  diff_H = fabs(chem1->GammaHI-chem2->GammaHI)/(chem1->GammaHI+1.0e-30);
  diff_elec = fabs(chem1->felec - chem2->felec)/(chem1->felec+1.0e-30);
#else 
  if(chem1->fHI < 0.5) {
    diff_H = fabs(chem1->fHI - chem2->fHI)/(chem1->fHI+1.0e-30);
  }else{
    diff_H = fabs(chem1->fHII - chem2->fHII)/(chem1->fHII+1.0e-30);
  }
#endif

  return MAX(diff_H,diff_elec);
}

__device__ float diff_uene(float uene1, float uene2)
{
  float diff;
  diff = fabs(uene1-uene2)/(MAX(uene1,uene2)+1.0e-30);

  return diff;
}

__global__ void advance_heatcool_kernel(struct fluid_mesh *mesh,
					const struct prim_chem* __restrict__ chem_updated,
					struct run_param *this_run,
					float dtime, 
					float *diff, 
					uint64_t offset)
{
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x + offset;
  unsigned int local_tid = threadIdx.x;

  struct fluid_mesh *target_mesh; /* pointer to target cells */
  struct prim_chem prev_chem;
  float prev_uene;

  int nrec, niter;

  target_mesh = &mesh[tid];

  prev_uene = target_mesh->prev_uene;
  prev_chem = chem_updated[tid];

  target_mesh->duene = 0.0; 
  nrec = 0;

  advance_heatcool_dev(target_mesh, &prev_uene, &prev_chem, 
		       this_run, dtime, &nrec, &niter);

  diff_shared[local_tid] = diff_uene(prev_uene, target_mesh->uene);

  __syncthreads();

  for(unsigned int s=blockDim.x/2; s>0;s>>=1) {
    if (local_tid < s) {
      diff_shared[local_tid] = MAX(diff_shared[local_tid+s],
				   diff_shared[local_tid]);
    }
    __syncthreads();
  }
  
  diff[blockIdx.x] = diff_shared[0];

  target_mesh->uene = prev_uene;
  
}

__global__ void advance_reaction_kernel(struct fluid_mesh *mesh,
					struct prim_chem *chem_updated,
					struct run_param *this_run,
					float dtime, 
					float *diff, 
					uint64_t offset)
{
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x + offset;
  unsigned int local_tid = threadIdx.x;

  struct fluid_mesh *target_mesh; /* pointer to target cells */
  struct prim_chem prev_chem;

  target_mesh = &mesh[tid];

  prev_chem = target_mesh->prev_chem;

  advance_reaction_dev(target_mesh, &prev_chem, this_run, dtime);

  diff_shared[local_tid] = diff_chem(&prev_chem, &(target_mesh->chem));

  __syncthreads();

  for(unsigned int s=blockDim.x/2; s>0;s>>=1) {
    if (local_tid < s) {
      diff_shared[local_tid] = MAX(diff_shared[local_tid+s],
				   diff_shared[local_tid]);
    }
    __syncthreads();
  }
  
  diff[blockIdx.x] = diff_shared[0];

  target_mesh->chem = prev_chem;
  chem_updated[tid] = prev_chem;

}

__global__ void advance_reaction_and_heatcool_kernel(struct fluid_mesh *mesh,
                                                     struct prim_chem *chem_updated,
                                                     struct run_param *this_run,
                                                     float dtime, 
                                                     float *diffc, float *diffu, 
                                                     uint64_t offset)
{
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x + offset;
  unsigned int local_tid = threadIdx.x;

  struct fluid_mesh *target_mesh; /* pointer to target cells */
  struct prim_chem prev_chem;
  float prev_uene;

  target_mesh = &mesh[tid];

  prev_uene = target_mesh->prev_uene;
  prev_chem = target_mesh->prev_chem;

  target_mesh->duene = 0.0; 
  advance_reaction_and_heatcool_dev(target_mesh, &prev_uene, &prev_chem, this_run, dtime);

  diff_shared[local_tid] = diff_chem(&prev_chem, &(target_mesh->chem));
  __syncthreads();

  for(unsigned int s=blockDim.x/2; s>0;s>>=1) {
    if (local_tid < s) {
      diff_shared[local_tid] = MAX(diff_shared[local_tid+s],
                                   diff_shared[local_tid]);
    }
    __syncthreads();
  }
  diffc[blockIdx.x] = diff_shared[0];
  

  diff_shared[local_tid] = diff_uene(prev_uene, target_mesh->uene);
  __syncthreads();

  for(unsigned int s=blockDim.x/2; s>0;s>>=1) {
    if (local_tid < s) {
      diff_shared[local_tid] = MAX(diff_shared[local_tid+s],
                                   diff_shared[local_tid]);

    }
    __syncthreads();
  }
  diffu[blockIdx.x] = diff_shared[0];

  target_mesh->uene = prev_uene;
  target_mesh->chem = prev_chem;
  chem_updated[tid] = prev_chem;
}
