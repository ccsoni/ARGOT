#include <stdio.h>
#include <stdlib.h>

#include "diffuse_photon.h"
#include "cuda_mem_space.h"

extern "C" 
void setup_ray_face_dev(struct ray_face *start_ray, struct cuda_param *this_cuda)
{
  int idev;

  for(idev=0; idev<this_cuda->num_cuda_dev; idev++) {
   
    CUDA_SAFE( cudaSetDevice(idev) );
    
    CUDA_SAFE( cudaMalloc((void **) &(start_ray[idev].xy),
			  sizeof(struct ray_info)*NMESH_MAX_FACE) );
    CUDA_SAFE( cudaMalloc((void **) &(start_ray[idev].yz),
			  sizeof(struct ray_info)*NMESH_MAX_FACE) );
    CUDA_SAFE( cudaMalloc((void **) &(start_ray[idev].zx),
			  sizeof(struct ray_info)*NMESH_MAX_FACE) );
  }
  
}

extern "C" 
void finalize_ray_face_dev(struct ray_face *start_ray, struct cuda_param *this_cuda)
{
  int idev;

  for(idev=0; idev<this_cuda->num_cuda_dev; idev++) {
   
    CUDA_SAFE( cudaSetDevice(idev) );
    
    CUDA_SAFE( cudaFree(start_ray[idev].xy) );
    CUDA_SAFE( cudaFree(start_ray[idev].yz) );
    CUDA_SAFE( cudaFree(start_ray[idev].zx) );
  }

}

extern "C" 
void send_ray_face(struct ray_face *ray1, struct ray_face *ray2,
		   cudaStream_t cp_strm, int device_id)
{
  CUDA_SAFE( cudaSetDevice(device_id) );

  CUDA_SAFE( cudaMemcpyAsync(ray2->xy, ray1->xy, 
			     sizeof(struct ray_info)*NMESH_MAX_FACE,
			     cudaMemcpyDefault, cp_strm) );
  
  CUDA_SAFE( cudaMemcpyAsync(ray2->yz, ray1->yz, 
			     sizeof(struct ray_info)*NMESH_MAX_FACE,
			     cudaMemcpyDefault, cp_strm) );
  
  CUDA_SAFE( cudaMemcpyAsync(ray2->zx, ray1->zx, 
			     sizeof(struct ray_info)*NMESH_MAX_FACE,
			     cudaMemcpyDefault, cp_strm) );

  CUDA_SAFE( cudaStreamSynchronize(cp_strm) );
}

extern "C"
void cuda_set_device(int device_id)
{
  CUDA_SAFE( cudaSetDevice(device_id) );
}
