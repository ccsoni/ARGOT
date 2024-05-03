#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>

#include "run_param.h"
#include "constants.h"
#include "fluid.h"
#include "radiation.h"

extern "C"
void allocate_pinned_segment(struct ray_segment **seg, uint64_t nseg)
{
  cudaHostAlloc(seg, sizeof(struct ray_segment)*nseg, cudaHostAllocDefault);
}

extern "C"
void deallocate_pinned_segment(struct ray_segment *seg)
{
  cudaFreeHost(seg);
}

extern "C" 
void allocate_pinned_light_ray_IO(struct light_ray_IO **ray_IO, uint64_t nray)
{
  cudaHostAlloc(ray_IO, sizeof(struct light_ray_IO)*nray, cudaHostAllocDefault);
}

extern "C"
void deallocate_pinned_light_ray_IO(struct light_ray_IO *ray_IO)
{
  cudaFreeHost(ray_IO);
}

