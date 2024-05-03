#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "constants.h"
#include "run_param.h"
#include "fluid.h"
#include "chemistry.h"

float gammaH2(struct fluid_mesh *mesh, struct run_param *this_run)
{
  float gamma_H2, gamma_H2_m1;
  float x;
  float tmpr, wmol;

  wmol = WMOL(mesh->chem);
  tmpr = mesh->uene*this_run->uenetok*wmol;

  x = 6100.0/tmpr;

  //gamma_H2_m1 = 2.0/(5.0+2.0*SQR(x)*expf(x)/SQR(expf(x)-1)); /* gamma - 1 */
  gamma_H2_m1 = 2.0/(5.0+2.0*SQR(x)*exp(-x)/SQR(1.0-exp(-x)));
  
  gamma_H2 = gamma_H2_m1 + 1;

  return gamma_H2;
}

float gamma_total(struct fluid_mesh *mesh, struct run_param *this_run)
{
  float gamma_H2_m1_inv; /* 1/(gamma_H2-1) */

  float x;
  float tmpr, wmol;
  struct prim_chem *chem;

  wmol = WMOL(mesh->chem);
  tmpr = mesh->uene*this_run->uenetok*wmol;

  chem = &(mesh->chem);

  x = 6100.0/tmpr;
  
  if(x < 10.0) {
    //gamma_H2_m1_inv = 0.5*(5.0+2.0*SQR(x)*expf(x)/SQR(expf(x)-1));
    gamma_H2_m1_inv = 0.5*(5.0+2.0*SQR(x)*exp(-x)/SQR(1.0-exp(-x)));
  }else{
    gamma_H2_m1_inv = 2.5;
  }

  float sum,denom;

  sum = (chem->fHI+chem->fHII)/GAMM1_MONOATOMIC; /* atomic hydrogen */
  denom = chem->fHI+chem->fHII;

#ifdef __HELIUM__
  sum += HELIUM_FACT/GAMM1_MONOATOMIC;
  denom += HELIUM_FACT;
#endif /* __HELIUM__ */

#ifdef __HYDROGEN_MOL__
  sum += chem->fHM/GAMM1_MONOATOMIC;
  sum += chem->fH2II/GAMM1_DIATOMIC;
  sum += chem->fH2I*gamma_H2_m1_inv;
  denom += (chem->fHM + chem->fH2I + chem->fH2II);
#endif
  
  float gamma_m1;
  gamma_m1 = denom/sum;

  return (gamma_m1+1.0);
  //return (1.4);
}
