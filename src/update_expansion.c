#include <stdio.h>
#include <math.h>

#include "cosmology.h"
#include "run_param.h"

void update_expansion(float tnow, struct run_param *this_run)
{
  float om,ov;

  om = this_run->cosm.omega_m;
  ov = this_run->cosm.omega_v;

  this_run->anow = timetoa(tnow, this_run->cosm);
  this_run->znow = 1.0/this_run->anow-1.0;

  this_run->hnow = sqrt(1.e0+om*(1.e0/this_run->anow-1.e0)
                        +ov*(SQR(this_run->anow)-1.e0))/this_run->anow;

}
