#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

#include "constants.h"
#include "run_param.h"
#include "mpi_param.h"
#include "radiation.h"
#include "tree_clist.h"

#include "prototype.h"

uint64_t set_mesh_index_range(int imesh_start, int imesh_end, uint64_t *nray_to)
{
  int imesh;
  uint64_t nray_sum;

#if 0
  nray_sum=0;

  imesh = imesh_start;
  while(nray_sum+nray_to[imesh] < MAX_NRAY_PER_TARGET && imesh < NMESH_LOCAL) {
    nray_sum += nray_to[imesh];
    imesh++;
  }

  *imesh_end = imesh;

  assert(nray_sum <= MAX_NRAY_PER_TARGET);
#else
  nray_sum=0;
  for(imesh=imesh_start;imesh<imesh_end;imesh++) {
    nray_sum += nray_to[imesh];
  }

  assert(nray_sum <= MAX_NRAY_PER_TARGET);
#endif

  return nray_sum;
}

int main(int argc, char **argv) 
{
  
  static struct run_param this_run;

  static struct radiation_src src[NSOURCE_MAX];
  static struct light_ray ray[NMESH_LOCAL];
  //  struct light_ray *ray;

  this_run.proc_file = stdout;
  this_run.xmax = this_run.ymax = this_run.zmax = 1.0;
  this_run.xmin = this_run.ymin = this_run.zmin = 0.0;

  this_run.rank_x = this_run.rank_y = this_run.rank_z = 1;
  this_run.xmin_local = this_run.ymin_local = this_run.zmin_local = 0.5;
  this_run.xmax_local = this_run.ymax_local = this_run.zmax_local = 1.0;
  this_run.delta_x = this_run.delta_y = this_run.delta_z = 1.0/NMESH_X_TOTAL;

  if(argc != 3) {
    fprintf(stderr, "Usage :: %s prefix theta\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  input_src(src, &this_run, argv[1]);
  src[0].xpos = 0.5; src[0].ypos = src[0].zpos = 1.0;
  printf("number of sources :: %d\n", this_run.nsrc);

  this_run.theta_crit = atof(argv[2]);

  struct clist_t *clist;
  uint64_t *key;
  int *index;
  int nclist;

  init_tree_param(60, &this_run, &nclist);

  clist = (struct clist_t *) malloc(sizeof(struct clist_t)*nclist);
  key = (uint64_t *)malloc(sizeof(uint64_t)*this_run.nsrc);
  index = (int *)malloc(sizeof(int)*this_run.nsrc);
  construct_tree(clist, key, index, src, &this_run);
  //  setup_light_ray_tree(&ray, src, &this_run);

  int imesh;
  int im_start, im_end;
  im_start = 3325;
  im_end = 3328;

  uint64_t *nray_to;
  nray_to = (uint64_t *)malloc(sizeof(uint64_t)*NMESH_LOCAL);  

  /* count the number of light rays targeted to each mesh */
  for(int imesh=0;imesh<NMESH_LOCAL;imesh++) {
    int ix, iy, iz;

    ix = imesh/(NMESH_Y_LOCAL*NMESH_Z_LOCAL);
    iy = (imesh-ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL)/NMESH_Z_LOCAL;
    iz = imesh - ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL - iy*NMESH_Z_LOCAL;

    nray_to[imesh] = count_ray_to(ix, iy, iz, clist, index, &this_run);
    
    assert(nray_to[imesh] <= MAX_NRAY_PER_TARGET);
  }

  uint64_t set_mesh_index_range(int, int, uint64_t*);
  this_run.nray = set_mesh_index_range(im_start, im_end, nray_to);
  
  setup_light_ray_range(im_start, im_end, ray, 1, clist, src, index, &this_run);

#if 1
  static double total_lum[NMESH_LOCAL];
  static uint64_t nray_per_mesh[NMESH_LOCAL];
  for(int imesh=0;imesh<NMESH_LOCAL;imesh++)  {
    total_lum[imesh] = 0.0;
    nray_per_mesh[imesh] = 0;
  }
  
  for(int iray=0;iray<this_run.nray;iray++) {
    int ix = ray[iray].ix_target - NMESH_X_LOCAL*this_run.rank_x;
    int iy = ray[iray].iy_target - NMESH_Y_LOCAL*this_run.rank_y;
    int iz = ray[iray].iz_target - NMESH_Z_LOCAL*this_run.rank_z;
    int imesh = iz + NMESH_Z_LOCAL*(iy + NMESH_Y_LOCAL*ix);
    printf("%d %d : (%14.6e %14.6e %14.6e) :  %14.6e ===> ",
    	   iray, imesh, ray[iray].src.xpos,ray[iray].src.ypos,ray[iray].src.zpos,ray[iray].src.photon_rate[0]);
    printf("(%14.6e %14.6e %14.6e)\n",
    	   this_run.xmin+this_run.delta_x*(0.5+ray[iray].ix_target),
	   this_run.ymin+this_run.delta_y*(0.5+ray[iray].iy_target),
	   this_run.zmin+this_run.delta_z*(0.5+ray[iray].iz_target));
    total_lum[imesh] += ray[iray].src.photon_rate[0];
    nray_per_mesh[imesh] += 1;
  }

  //  for(int imesh=0;imesh<NMESH_LOCAL;imesh++) printf("%d %14.6e %d\n",imesh, total_lum[imesh],nray_per_mesh[imesh]);
#endif	 

  //  free(ray);
}
