#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#include "constants.h"
#include "particle.h"
#include "sph.h"
#include "run_param.h"
#include "chemistry.h"

#ifndef MAX
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b) ((a)<(b) ? (a) : (b))
#endif

#define SQRTFACT (1.0488088481) // sqrt(1.1)

#ifdef __HEATCOOL__

#define TOL_WMOL (1.e-3)
#define DTINY    (1.0e-60)

#include "heatcool_table.h"

void output_heatcool_table(struct heatcool_table *tbl){
  int inh, it;
  FILE *hctbl;

  hctbl = fopen("heatcool.tbl","w");
  for(it=0;it<N_T_BIN;it++) {
    for(inh=0;inh<N_NH_BIN;inh++) {
      fprintf(hctbl, "%14.6e %14.6e %14.6e %14.6e %14.6e\n",
	      tbl->lognh_tbl[inh],
	      tbl->logt_tbl[it],
	      tbl->heat_tbl[inh][it]-2.0*tbl->lognh_tbl[inh],
	      tbl->cool_tbl[inh][it]-2.0*tbl->lognh_tbl[inh],
	      tbl->wmol_tbl[inh][it]);
    }
    fprintf(hctbl,"\n");
  }
  fclose(hctbl);
}

int advance_heatcool_ioneq(struct SPH_Particle *sph,
			   struct run_param *this_run,
			   struct heatcool_table *tbl,
			   float dtime, int *nrec)
{

  static int initflag=1;
  static double eunit, convfact;
  static float wmol_min, wmol_max;
  static float uene_min, uene_max;

  double heatcool, heatcool_prev;
  double nH, T;

  float wmol_tmp, anow3i, frac;
  float uene, uene_up, uene_lo, uene_old, dens;

  float duad;

  int it, inh, iter, err, ret_rec;
  int niter_wmol;

  double logCool, logHeat;
  double Cool, Heat;

  float lognh;

  float *heat_tbl_1d, *cool_tbl_1d, *wmol_tbl_1d;

  anow3i = 1.0/CUBE(this_run->anow);

  heat_tbl_1d = (float *)malloc(sizeof(float)*N_T_BIN);
  cool_tbl_1d = (float *)malloc(sizeof(float)*N_T_BIN);
  wmol_tbl_1d = (float *)malloc(sizeof(float)*N_T_BIN);

  lognh = log10(sph->dens*this_run->denstonh*anow3i);
  lognh = MIN(LOG_NH_MAX,MAX(LOG_NH_MIN, lognh));
  inh = (int)((lognh-LOG_NH_MIN)/tbl->dlognh);
  inh = MIN(N_NH_BIN-2, MAX(0, inh));

  frac = (lognh-tbl->lognh_tbl[inh])/tbl->dlognh;

  for(it=0;it<N_T_BIN;it++) 
    heat_tbl_1d[it] = (1.0-frac)*tbl->heat_tbl[inh][it];
  for(it=0;it<N_T_BIN;it++) 
    heat_tbl_1d[it] += frac*tbl->heat_tbl[inh+1][it];
  
  for(it=0;it<N_T_BIN;it++) 
    cool_tbl_1d[it] = (1.0-frac)*tbl->cool_tbl[inh][it];
  for(it=0;it<N_T_BIN;it++) 
    cool_tbl_1d[it] += frac*tbl->cool_tbl[inh+1][it];
  
  for(it=0;it<N_T_BIN;it++) 
    wmol_tbl_1d[it] = (1.0-frac)*tbl->wmol_tbl[inh][it];
  for(it=0;it<N_T_BIN;it++) 
    wmol_tbl_1d[it] += frac*tbl->wmol_tbl[inh+1][it];  

  err=0;
  ret_rec=0;

  //  if(initflag==1){
  eunit = SQR(this_run->lunit)/SQR(this_run->tunit);
  convfact = this_run->tunit/eunit*CUBE(this_run->lunit)/this_run->munit;
  wmol_min = 4.0/(5.0*xhydrog+3.0)/meanmu;
  wmol_max = 1.0/(xhydrog+0.25*yhelium)/meanmu;
  
  uene_min = pow(10.0,LOG_T_MIN+0.5)/this_run->uenetok/wmol_max;
  uene_max = pow(10.0,LOG_T_MAX-0.5)/this_run->uenetok/wmol_min;
  
  initflag=0;
    //  }

  dens = sph->dens*anow3i;

#ifdef __ENTROPY__
  uene_lo = uene_up = uene_old = 
    (sph->etrp+sph->detrp*dtime)*pow(dens,GAMM1)/GAMM1;
#else
  uene_lo = uene_up = uene_old = sph->uene+sph->duene*dtime;
#endif

  duad = (sph->duene-sph->duvisc) + 0.5*sph->uene/dtime;

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
    err=-1;
  }

  nH = dens*this_run->denstonh;

#if 0
  niter_wmol=0;
  wmol_tmp = 1.0;
  do {
    sph->wmol = wmol_tmp;
    //    T  = sph->uene*this_run->uenetok*sph->wmol/meanmu;
    T  = uene_old*this_run->uenetok*sph->wmol/meanmu;
    it = (int)((log10(T)-LOG_T_MIN)/tbl->dlogt);
    it = MIN(N_T_BIN-2,MAX(0,it));
    frac = (log10(T)-tbl->logt_tbl[it])/tbl->dlogt;
    wmol_tmp = (1.0-frac)*wmol_tbl_1d[it]+frac*wmol_tbl_1d[it+1];
    wmol_tmp = 0.5*(wmol_tmp+sph->wmol);
    niter_wmol++;
  } while(fabs(wmol_tmp-sph->wmol) > TOL_WMOL*sph->wmol && niter_wmol<10);
#else
  sph->wmol = 0.59;
  //  T  = sph->uene*this_run->uenetok*sph->wmol/meanmu;
  T  = uene_old*this_run->uenetok*sph->wmol/meanmu;
  it = (int)((log10(T)-LOG_T_MIN)/tbl->dlogt);
  it = MIN(N_T_BIN-2,MAX(0,it));
  frac = (log10(T)-tbl->logt_tbl[it])/tbl->dlogt;
#endif


  // First, we look for the upper and lower bounds to bracket the solution.

  logHeat = (1.0-frac)*heat_tbl_1d[it]+frac*heat_tbl_1d[it+1];
  logCool = (1.0-frac)*cool_tbl_1d[it]+frac*cool_tbl_1d[it+1];
  Heat = pow(10.0,logHeat);
  Cool = pow(10.0,logCool);
  heatcool = (Heat-Cool)/dens*convfact;
  heatcool_prev = heatcool;

  if(heatcool>0.0) { // heating
    uene_up = MIN(uene_up*SQRTFACT, uene_max);
    uene_lo = MAX(uene_lo/SQRTFACT, uene_min);

    T = uene_up*this_run->uenetok*sph->wmol/meanmu;
    it = (int)((log10(T)-LOG_T_MIN)/tbl->dlogt);
    it = MIN(N_T_BIN-2,MAX(0,it));
    frac = (log10(T)-tbl->logt_tbl[it])/tbl->dlogt;

    logHeat = (1.0-frac)*heat_tbl_1d[it]+frac*heat_tbl_1d[it+1];
    logCool = (1.0-frac)*cool_tbl_1d[it]+frac*cool_tbl_1d[it+1];
    Heat = pow(10.0,logHeat);
    Cool = pow(10.0,logCool);
    heatcool = (Heat-Cool)/dens*convfact;

#ifndef __ENTROPY__
    heatcool = duad*heatcool/sqrt(SQR(duad)+SQR(heatcool));
#endif

    while(uene_up-uene_old-heatcool*dtime<0.0){
      uene_up *= 1.1;
      uene_lo *= 1.1;
      
      if (uene_up > uene_max) {
	err = -1;
	break;
      }

      T = uene_up*this_run->uenetok*sph->wmol/meanmu;
      it = (int)((log10(T)-LOG_T_MIN)/tbl->dlogt);
      it = MIN(N_T_BIN-2,MAX(0,it));
      frac = (log10(T)-tbl->logt_tbl[it])/tbl->dlogt;

      logHeat = (1.0-frac)*heat_tbl_1d[it]+frac*heat_tbl_1d[it+1];
      logCool = (1.0-frac)*cool_tbl_1d[it]+frac*cool_tbl_1d[it+1];
      Heat = pow(10.0,logHeat);
      Cool = pow(10.0,logCool);
      heatcool = (Heat-Cool)/dens*convfact;

#ifndef __ENTROPY__
      heatcool = duad*heatcool/sqrt(SQR(duad)+SQR(heatcool));
#endif

    }
  }else{ // cooling
    uene_up = MIN(uene_up*SQRTFACT, uene_max);
    uene_lo = MAX(uene_lo/SQRTFACT, uene_min);

    T = uene_lo*this_run->uenetok*sph->wmol/meanmu;
    it = (int)((log10(T)-LOG_T_MIN)/tbl->dlogt);
    it = MIN(N_T_BIN-2,MAX(0,it));
    frac = (log10(T)-tbl->logt_tbl[it])/tbl->dlogt;

    logHeat = (1.0-frac)*heat_tbl_1d[it]+frac*heat_tbl_1d[it+1];
    logCool = (1.0-frac)*cool_tbl_1d[it]+frac*cool_tbl_1d[it+1];
    Heat = pow(10.0,logHeat);
    Cool = pow(10.0,logCool);
    heatcool = (Heat-Cool)/dens*convfact;

#ifndef __ENTROPY__
    heatcool = duad*heatcool/sqrt(SQR(duad)+SQR(heatcool));
#endif

    while(uene_lo-uene_old-heatcool*dtime>0.0){
      uene_up /= 1.1;
      uene_lo /= 1.1;

      if (uene_lo < uene_min) {
	err = -1;
	break;
      }

      T = uene_lo*this_run->uenetok*sph->wmol/meanmu;
      it = (int)((log10(T)-LOG_T_MIN)/tbl->dlogt);
      it = MIN(N_T_BIN-2,MAX(0,it));
      frac = (log10(T)-tbl->logt_tbl[it])/tbl->dlogt;

      logHeat = (1.0-frac)*heat_tbl_1d[it]+frac*heat_tbl_1d[it+1];
      logCool = (1.0-frac)*cool_tbl_1d[it]+frac*cool_tbl_1d[it+1];
      Heat = pow(10.0,logHeat);
      Cool = pow(10.0,logCool);
      heatcool = (Heat-Cool)/dens*convfact;

#ifndef __ENTROPY__
      heatcool = duad*heatcool/sqrt(SQR(duad)+SQR(heatcool));
#endif

    }
  }
  assert(uene_up>=0.0);
  assert(uene_lo>=0.0);

  if(err == -1) {
    if((*nrec)>=10) {

#ifdef __ENTROPY__
      
#else
      if( (heatcool_prev+sph->duene)*dtime > -0.5*sph->uene ) {
	sph->uene += (heatcool_prev+sph->duene)*dtime;
      }else{
	sph->uene *= 0.5;
      }
      sph->uene = MAX(MIN(sph->uene,uene_max),uene_min);
      sph->durad = (sph->uene-uene_old)/dtime-sph->duene;
#endif

      free(heat_tbl_1d);
      free(cool_tbl_1d);
      free(wmol_tbl_1d);
      return (4);
    }else{
      (*nrec)++;
      ret_rec |= advance_heatcool_ioneq(sph, this_run, tbl, dtime/4.0, nrec);
      ret_rec |= advance_heatcool_ioneq(sph, this_run, tbl, dtime/4.0, nrec);
      ret_rec |= advance_heatcool_ioneq(sph, this_run, tbl, dtime/4.0, nrec);
      ret_rec |= advance_heatcool_ioneq(sph, this_run, tbl, dtime/4.0, nrec);
    }
  }

  iter = 0;
  do {
    uene = 0.5*(uene_up + uene_lo);

    //    T = uene*this_run->uenetok*sph->wmol/meanmu;
    //    T = 0.5*(sph->uene+uene)*this_run->uenetok*sph->wmol/meanmu;
    T = 0.5*(uene_old+uene)*this_run->uenetok*sph->wmol/meanmu;
    it = (int)((log10(T)-LOG_T_MIN)/tbl->dlogt);
    it = MIN(N_T_BIN-2,MAX(0,it));
    frac = (log10(T)-tbl->logt_tbl[it])/tbl->dlogt;

    logHeat = (1.0-frac)*heat_tbl_1d[it]+frac*heat_tbl_1d[it+1];
    logCool = (1.0-frac)*cool_tbl_1d[it]+frac*cool_tbl_1d[it+1];
    Heat = pow(10.0,logHeat);
    Cool = pow(10.0,logCool);
    heatcool = (Heat-Cool)/dens*convfact;

#ifndef __ENTROPY__
    heatcool = duad*heatcool/sqrt(SQR(duad)+SQR(heatcool));
#endif

    if (uene - uene_old - heatcool*dtime > 0.0) {
      uene_up = uene;
    }else{
      uene_lo = uene;
    }

    iter++;
  }while( (uene_up-uene_lo)/uene > 1.0e-4 && iter<40);

  assert(uene>=0.0);

#if 1
  sph->uene = uene;
#ifdef __ENTROPY__
  sph->etrp = GAMM1*uene/pow(dens,GAMM1);
#endif /* __ENTROPY__ */
#else
  if(uene < 0.5*sph->uene) {
    sph->uene *= 0.5;
  }else{
    sph->uene = uene;
  }
#endif
  sph->durad = (uene-uene_old)/dtime;


  free(heat_tbl_1d);
  free(cool_tbl_1d);
  free(wmol_tbl_1d);

  if(iter>=40) {
    return(2|ret_rec);
  }else{
    return(0|ret_rec);
  }

}



void step_heatcool_ioneq(struct Particle *particle, struct SPH_Particle *sph, 
			 struct run_param *this_run, float dtime)
{
  int p;

  static float zred_tbl=-1.0;
  static int init_flag=1;
  int err_dist0, err_dist1, err_dist2, err_dist3;

  static struct heatcool_table tbl;

  float anow3i;

  if(init_flag==1) {
    int inh, it;
    for(inh=0;inh<N_NH_BIN;inh++) {
      tbl.lognh_tbl[inh] = 
	LOG_NH_MIN + (LOG_NH_MAX-LOG_NH_MIN)/(float)(N_NH_BIN-1)*(float)inh;
    }
    tbl.dlognh = (LOG_NH_MAX-LOG_NH_MIN)/(float)(N_NH_BIN-1);

    for(it=0;it<N_T_BIN;it++) {
      tbl.logt_tbl[it] =
	LOG_T_MIN + (LOG_T_MAX-LOG_T_MIN)/(float)(N_T_BIN-1)*(float)it;
    }
    tbl.dlogt = (LOG_T_MAX-LOG_T_MIN)/(float)(N_T_BIN-1);
    
    init_flag=0;
  }

  if(zred_tbl != this_run->znow) {
    fprintf(this_run->process_file,
	    "Updating heating/cooling table at redshift %14.6e...",
	    this_run->znow);
    fflush(this_run->process_file);
    int inh;

#pragma omp parallel for schedule(dynamic,1)
    for(inh=0;inh<N_NH_BIN;inh++) {
      double nH,T;
      int it;

      nH = pow(10.0,tbl.lognh_tbl[inh]);

      for(it=0;it<N_T_BIN;it++) {
	struct prim_chem pchem;

	T = pow(10.0, tbl.logt_tbl[it]);

	calc_ioneq(&pchem, nH, T, this_run->znow);

	tbl.heat_tbl[inh][it] = 
	  log10(calc_heating_rate(&pchem,this_run->znow,nH,T)+DTINY);
	tbl.cool_tbl[inh][it] = 
	  log10(calc_cooling_rate(&pchem,this_run->znow,nH,T)+DTINY);
	tbl.wmol_tbl[inh][it] = WMOL(pchem);
      }
    }
    zred_tbl = this_run->znow;
    fprintf(this_run->process_file,
	    "Done\n");
    fflush(this_run->process_file);
    // if(this_run->mpi.rank == 0) 
    output_heatcool_table(&tbl);
  }

  anow3i = 1.0/CUBE(this_run->anow);

  int ndig;

  ndig = 0;
  err_dist0 = err_dist1 = err_dist2 = err_dist3 = 0;

#pragma omp parallel for schedule(dynamic,1) reduction(+:err_dist0,err_dist1,err_dist2,err_dist3, ndig)
  for(p=0;p<this_run->ngas;p++) {

    int nrec, err;

    nrec = 0;

    err = advance_heatcool_ioneq(&sph[p], this_run, &tbl, dtime, &nrec);

    if( err&1 ) err_dist3++;  // out of region
    if( err&2 ) err_dist1++;  // not converged
    if( err&4 ) err_dist2++;  // too much recursion
    if( (err&7) == 0 ) err_dist0++;

    if(nrec != 0) ndig++;

  }

  fprintf(this_run->process_file,
	  "Error code distribution                 normal: %d / %d\n",
	  err_dist0,this_run->ngas);
  fprintf(this_run->process_file,
	  "                                 not converged: %d\n",
	  err_dist1);
  fprintf(this_run->process_file,
	  "                            too much recursion: %d\n",
	  err_dist2);
  fprintf(this_run->process_file,
	  "               out of table and recursive call: %d\n",
	  err_dist3);
  fprintf(this_run->process_file,
	  "           # of particles with recursive calls: %d ",
	  ndig);

  fprintf(this_run->process_file,"\n");

}

#undef TOL_WMOL

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
    this_run.munit*xhydrog/mproton;

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
