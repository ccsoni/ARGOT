#include <math.h>
#include "run_param.h"

void update_now(struct run_param *this_run)
{
  float zred,ascale,om,ov;

  om = this_run->cosm.omega0;
  ov = this_run->cosm.lambda0;

  zred = timetoz(this_run->tnow, this_run->cosm);

  ascale = 1.e0/(1.e0+zred);

  this_run->znow = zred;
  this_run->anow = ascale;
  this_run->hnow = sqrt(1.e0+om*(1.e0/ascale-1.e0)
			+ov*(ascale*ascale-1.e0))/ascale;

  return;
}

