#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "constants.h"
#include "run_param.h"
#include "chemistry.h"
#include "fluid.h"

#define LOG_T_MIN (0.5)
#define LOG_T_MAX (8.5)

#define SQRTFACT (1.0488088481) // sqrt(1.1)

#include "heatcool_rate.cu"

__device__ int advance_heatcool_dev(struct fluid_mesh *mesh, float *uene, 
				    struct prim_chem *chem, 
				    struct run_param *this_run,
				    float dtime, int *nrec, int *niter)
{
  double eunit, convfact;
  float wmol_min, wmol_max;
  float uene_min, uene_max;

  double heatcool, heatcool_prev;
  double nH, T;

  float wmol,anow3i;
  double uene_new, uene_up, uene_lo, uene_old, dens;

  int iter, err, ret_rec;

  eunit = SQR(this_run->lunit)/SQR(this_run->tunit);
  convfact = this_run->tunit/eunit*CUBE(this_run->lunit)/this_run->munit;
  wmol_min = 4.0/(5.0*XHYDROGEN+3.0);
  wmol_max = 1.0/(XHYDROGEN+0.25*YHELIUM);
  
  uene_min = pow(10.0,LOG_T_MIN+0.5)/this_run->uenetok/wmol_max;
  uene_max = pow(10.0,LOG_T_MAX-0.5)/this_run->uenetok/wmol_min;

  anow3i = 1.0/CUBE(this_run->anow);
  dens = mesh->dens*anow3i;

  err=0;
  ret_rec=0;

  //  uene_lo = uene_up = uene_old = mesh->uene+mesh->duene*dtime;
  uene_lo = uene_up = uene_old = *uene;

  if(uene_old < uene_min) {
    uene_up = uene_min;
    uene_lo = uene_min;
    uene_old = uene_min;
    ret_rec |= 1;
    err=-1;
  }else if(uene_old > uene_max) {
    uene_up = uene_max;
    uene_lo = uene_max;
    uene_old = uene_max;
    ret_rec |= 1;
  }

  wmol = WMOL(*chem);

  nH = dens*this_run->denstonh;
  //  T  = mesh->uene*this_run->uenetok*wmol;
  T  = (*uene)*this_run->uenetok*wmol;
#ifdef __TWOTEMP__
  T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */

  // First, we look for the upper and lower bounds to bracket the solution.
  heatcool = calc_heatcool_rate_dev(chem, this_run->znow, nH, T)/dens*convfact;
  heatcool_prev = heatcool;

  if(heatcool>0.0) { // heating
    uene_up = MIN(uene_up*SQRTFACT, uene_max);
    uene_lo = MAX(uene_lo/SQRTFACT, uene_min);
    T = uene_up*this_run->uenetok*wmol;
    //    T = 0.5*(uene_up+uene_old)*this_run->uenetok*wmol;
#ifdef __TWOTEMP__
    T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */

    heatcool = calc_heatcool_rate_dev(chem, this_run->znow, nH, T)/dens*convfact;

    while(uene_up-uene_old-heatcool*dtime<0.0){
      uene_up *= 1.1;
      uene_lo *= 1.1;
      
      if (uene_up > uene_max) {
        err = -1;
        break;
      }

      T = uene_up*this_run->uenetok*wmol;
#ifdef __TWOTEMP__
      T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */
      heatcool = calc_heatcool_rate_dev(chem, this_run->znow, nH, T)/dens*convfact;
    }
  }else{ // cooling
    uene_up = MIN(uene_up*SQRTFACT, uene_max);
    uene_lo = MAX(uene_lo/SQRTFACT, uene_min);
    T = uene_lo*this_run->uenetok*wmol;
#ifdef __TWOTEMP__
    T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */
    heatcool = calc_heatcool_rate_dev(chem, this_run->znow, nH, T)/dens*convfact;

    while(uene_lo-uene_old-heatcool*dtime>0.0){
      uene_up /= 1.1;
      uene_lo /= 1.1;

      if (uene_lo < uene_min) {
        err = -1;
        break;
      }

      T = uene_lo*this_run->uenetok*wmol;
#ifdef __TWOTEMP__
      T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */
      heatcool = calc_heatcool_rate_dev(chem, this_run->znow, nH, T)/dens*convfact;
    }
  }
  assert(uene_up>=0.0);
  assert(uene_lo>=0.0);

#if 0
  /* error handling in failing to bracketing the solution */
  if(err == -1) {
    if((*nrec)>=5) {
      if ( (heatcool_prev+mesh->duene)*dtime > -0.5*mesh->uene ){
        mesh->uene += (heatcool_prev+mesh->duene)*dtime;
      }else{
        mesh->uene *= 0.5;
      }
      mesh->uene = MAX(MIN(mesh->uene,uene_max),uene_min);
      //mesh->durad = (mesh->uene-uene_old)/dtime-mesh->duene;
      return (4);
    }else{
      (*nrec)++;
      ret_rec |= advance_heatcool(mesh, uene, chem, this_run, dtime/4.0, nrec, niter);
      ret_rec |= advance_heatcool(mesh, uene, chem, this_run, dtime/4.0, nrec, niter);
      ret_rec |= advance_heatcool(mesh, uene, chem, this_run, dtime/4.0, nrec, niter);
      ret_rec |= advance_heatcool(mesh, uene, chem, this_run, dtime/4.0, nrec, niter);
    }
  }
#endif

  iter = 0;
  do {
    uene_new = 0.5*(uene_up + uene_lo);

    T = 0.5*(uene_old+uene_new)*this_run->uenetok*wmol;
#ifdef __TWOTEMP__
    T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */
    heatcool = calc_heatcool_rate_dev(chem, this_run->znow, nH, T)/dens*convfact;

    if (uene_new - uene_old - heatcool*dtime > 0.0) {
      uene_up = uene_new;
    }else{
      uene_lo = uene_new;
    }

    iter++;
  }while( (uene_up-uene_lo) > 1.0e-5*uene_new && iter<32);

  *uene = uene_new;
  *niter = iter;

  if(iter==32) {
    return(2|ret_rec);
  }else{
    return(0|ret_rec);
  }
  
}
