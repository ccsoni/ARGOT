#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>

#include "run_param.h"
#include "fluid.h"
#include "radiation.h"

#include "prototype.h"

void setup_light_ray_long(struct light_ray **ray, struct radiation_src *src, 
			  struct run_param *this_run)
{
  int ix, iy, iz;
  uint64_t isrc;

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

  *ray = (struct light_ray *) malloc(sizeof(struct light_ray)*NMESH_LOCAL*this_run->nsrc);

  this_run->nray = 0;
  for(isrc=0;isrc<this_run->nsrc;isrc++) {
    for(ix=0;ix<NMESH_X_LOCAL;ix++) {
      int ix_global = this_run->rank_x*NMESH_X_LOCAL+ix;
      for(iy=0;iy<NMESH_Y_LOCAL;iy++) {
	int iy_global = this_run->rank_y*NMESH_Y_LOCAL+iy;
	for(iz=0;iz<NMESH_Z_LOCAL;iz++) {
	  int iz_global = this_run->rank_z*NMESH_Z_LOCAL+iz;
	  (*ray)[this_run->nray].src = src[isrc];
	  (*ray)[this_run->nray].ix_target = ix_global;
	  (*ray)[this_run->nray].iy_target = iy_global;
	  (*ray)[this_run->nray].iz_target = iz_global;
	  this_run->nray++;

	  //	  assert(this_run->nray < NRAY_MAX);
	}
      }
    }
  }

  calc_ray_segment(*ray,this_run);

#ifdef __ARGOT_PROFILE__
    times(&end_tms);
    gettimeofday(&end_tv, NULL);

    fprintf(this_run->proc_file,
	    "# setup_light_ray : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	    timing(start_tms, end_tms), wallclock_timing(start_tv, end_tv));
    fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */





}

