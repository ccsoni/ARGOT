#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#include "constants.h"
#include "fluid.h"
#include "run_param.h"
#include "chemistry.h"

#define LOG_T_MIN (0.5)
#define LOG_T_MAX (8.5)

#ifndef MAX
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b) ((a)<(b) ? (a) : (b))
#endif

#define SQRTFACT (1.0488088481) // sqrt(1.1)

#define AE (5.4461E-4)
#define SQRTAE (2.3337E-2)

#ifdef __HEATCOOL__

int advance_heatcool(struct fluid_mesh *mesh, 
		     float *uene,
		     struct prim_chem *chem,
                     struct run_param *this_run, 
                     float dtime, 
		     int *nrec, int *niter)
{
#if 0
  static int initflag=1;
  static double eunit, convfact;
  static float wmol_min, wmol_max;
  static float uene_min, uene_max;
#else
  double eunit, convfact;
  float wmol_min, wmol_max;
  float uene_min, uene_max;
#endif

  double heatcool, heatcool_prev;
  double nH, T;

  float wmol,anow3i;
  double uene_new, uene_up, uene_lo, uene_old, dens;

  int iter,err, ret_rec;

  double ionfrac;
  float  lnLambda;
  float  ne, telec;
  double nHII, nHeII, nHeIII;
  double t_relax_norm;
  double t_relax, t_shock;
  double t_relax_HII,t_relax_HeII,t_relax_HeIII;
  double t_relax_HII_inv, t_relax_HeII_inv, t_relax_HeIII_inv;

#if 1
  eunit = SQR(this_run->lunit)/SQR(this_run->tunit);
  convfact = this_run->tunit/eunit*CUBE(this_run->lunit)/this_run->munit;
  wmol_min = 4.0/(5.0*XHYDROGEN+3.0);
  wmol_max = 1.0/(XHYDROGEN+0.25*YHELIUM);
  
  uene_min = pow(10.0,LOG_T_MIN+0.5)/this_run->uenetok/wmol_max;
  uene_max = pow(10.0,LOG_T_MAX-0.5)/this_run->uenetok/wmol_min;
#endif

#if 0
  if(initflag==1){
    eunit = SQR(this_run->lunit)/SQR(this_run->tunit);
    convfact = this_run->tunit/eunit*CUBE(this_run->lunit)/this_run->munit;
    wmol_min = 4.0/(5.0*XHYDROGEN+3.0);
    wmol_max = 1.0/(XHYDROGEN+0.25*YHELIUM);
    
    uene_min = pow(10.0,LOG_T_MIN+0.5)/this_run->uenetok/wmol_max;
    uene_max = pow(10.0,LOG_T_MAX-0.5)/this_run->uenetok/wmol_min;
    initflag=0;
  }
#endif

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
  T = (*uene)*this_run->uenetok*wmol;
#ifdef __TWOTEMP__
  T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */

  // First, we look for the upper and lower bounds to bracket the solution.
  heatcool = calc_heatcool_rate(chem, this_run->znow, nH, T)/dens*convfact;
  heatcool_prev = heatcool;

  if(heatcool>0.0) { // heating
    uene_up = MIN(uene_up*SQRTFACT, uene_max);
    uene_lo = MAX(uene_lo/SQRTFACT, uene_min);
    T = uene_up*this_run->uenetok*wmol;
    //    T = 0.5*(uene_up+uene_old)*this_run->uenetok*wmol;
#ifdef __TWOTEMP__
    T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */

    heatcool = calc_heatcool_rate(chem, this_run->znow, nH, T)/dens*convfact;

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
      heatcool = calc_heatcool_rate(chem, this_run->znow, nH, T)/dens*convfact;
    }
  }else{ // cooling
    uene_up = MIN(uene_up*SQRTFACT, uene_max);
    uene_lo = MAX(uene_lo/SQRTFACT, uene_min);
    T = uene_lo*this_run->uenetok*wmol;
#ifdef __TWOTEMP__
    T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */
    heatcool = calc_heatcool_rate(chem, this_run->znow, nH, T)/dens*convfact;

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
      heatcool = calc_heatcool_rate(chem, this_run->znow, nH, T)/dens*convfact;
    }
  }
  assert(uene_up>=0.0);
  assert(uene_lo>=0.0);

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

  iter = 0;
  do {
    uene_new = 0.5*(uene_up + uene_lo);

    T = 0.5*(uene_old+uene_new)*this_run->uenetok*wmol;
#ifdef __TWOTEMP__
    T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */
    heatcool = calc_heatcool_rate(chem, this_run->znow, nH, T)/dens*convfact;

    if (uene_new - uene_old - heatcool*dtime > 0.0) {
      uene_up = uene_new;
    }else{
      uene_lo = uene_new;
    }

    iter++;
  }while( (uene_up-uene_lo) > 1.0e-5*uene_new && iter<32);

  //  mesh->uene = uene_new;
  *uene = uene_new;
  *niter = iter;

  if(iter==32) {
    return(2|ret_rec);
  }else{
    return(0|ret_rec);
  }
}

#endif

#ifdef __DEBUG__
int main(int argc, char **argv) 
{
  struct run_param this_run;

  static struct SPH_Particle sph[1];

  this_run.lunit = 3.306429e+26;
  this_run.munit = 2.789651e+51;
  this_run.tunit = 4.407519e+17;

  this_run.masstonh = 
    this_run.munit*XHYDROGEN/mproton;

  this_run.denstonh = 
    this_run.masstonh/CUBE(this_run.lunit);

  this_run.uenetok  = 
    GAMM1*meanmu*mproton/kboltz*SQR(this_run.lunit)/SQR(this_run.tunit);

  this_run.process_file = stdout;

  this_run.npart = 1;
  this_run.ngas = 1;

  this_run.znow = 0.4;
  this_run.anow = 1.0/(1.0+this_run.znow);

  float nh,tmpr,dtime;
  int nrec;

  nh = 1.0e-3;
  tmpr = 1.0e+5;

  sph[0].wmol = 0.59;
  sph[0].dens = nh/this_run.denstonh*CUBE(this_run.anow);
  //  sph[0].dens = nh/this_run.denstonh*CUBE(this_run.anow);
  sph[0].uene = tmpr/this_run.uenetok;
  sph[0].duene = 1.0e-5;

  dtime = 1.0e-4;
  do {
    step_heatcool_ioneq(sph, &this_run, dtime);
    printf("%14.6e\n",sph[0].uene*this_run.uenetok);
  }while(1);
}
#endif
