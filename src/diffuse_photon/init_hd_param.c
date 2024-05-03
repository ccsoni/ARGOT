#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "diffuse_photon.h"

void init_hd_param(struct host_diffuse_param *hd_param)
{
  hd_param->step_fact = (struct step_func_factor*)malloc(sizeof(struct step_func_factor));
  assert( hd_param->step_fact );
  hd_param->angle = (struct angle_info*)malloc(sizeof(struct angle_info)*N_ANG);
  assert( hd_param->angle );
#ifndef __USE_GPU__
  hd_param->rmesh = (struct radiation_mesh*)malloc(sizeof(struct radiation_mesh)*NMESH_LOCAL);
  assert( hd_param->rmesh );
#endif

}


void free_hd_param(struct host_diffuse_param *hd_param)
{
  free( hd_param->step_fact );
  free( hd_param->angle );
#ifndef __USE_GPU__
  free( hd_param->rmesh );
#endif
}
