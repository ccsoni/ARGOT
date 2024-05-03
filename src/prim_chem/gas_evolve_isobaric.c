#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "chemistry.h"
#include "constants.h"
#include "run_param.h"
#include "fluid.h"

void advance_reaction_and_heatcool(struct fluid_mesh*,
				   float*,
				   struct prim_chem*, 
				   struct run_param*,
				   float);

int advance_heatcool(struct fluid_mesh*,
		     float*,
		     struct prim_chem*, 
		     struct run_param*,
		     float, int*, int*);
void advance_reaction(struct fluid_mesh*, 
		      struct prim_chem*, 
		      struct run_param*, 
		      float);
float gamma_total(struct fluid_mesh*, struct run_param*);

float ztotime(float, struct cosmology);
float timetoz(float, struct cosmology);
void update_expansion(float, struct run_param*);

int main(int argc, char **argv) 
{
  struct run_param this_run;
  struct fluid_mesh mesh[1];
  struct prim_chem chem[1];

  float omega0, hubble, omegab;
  double dens, dens0, nH, nHe, wmol, T, uene, eunit, convfact;
  double heatcool;
  float dtime;

  int ierr, nrec, niter, p;
  double P0;

  float anow3;

  this_run.cosm.omega_m=1.0;
  this_run.cosm.omega_v=0.0;
  this_run.cosm.hubble=0.5;
  this_run.cosm.omega_b=0.1;

  omega0=this_run.cosm.omega_m;
  hubble=this_run.cosm.hubble;
  omegab=this_run.cosm.omega_b;

  this_run.lunit = mpc;
  //this_run.munit = 1.88e-29*hubble*hubble*pow(this_run.lunit,3.0)*8.0/3.0*PI;
  this_run.munit = 2.81e11*sunmass*SQR(hubble);
  this_run.tunit = 8.93e+17/(hubble*sqrt(omega0))*sqrt(3.e0*omega0/8.e0/PI);
  this_run.eunit = SQR(this_run.lunit)/SQR(this_run.tunit);
  convfact = this_run.tunit/this_run.eunit*CUBE(this_run.lunit)/this_run.munit;

  this_run.znow = 15.0;
  this_run.anow = 1.0/(1.0+this_run.znow);
  this_run.tnow = ztotime(this_run.znow, this_run.cosm);
  update_expansion(this_run.tnow, &this_run);
  anow3 = CUBE(this_run.anow);

  this_run.masstonh = this_run.munit*xhydrog/mproton;
  this_run.denstonh = this_run.masstonh/CUBE(this_run.lunit);
  this_run.uenetok  = GAMM1_MONOATOMIC*mproton/kboltz*this_run.eunit;

  chem[0].felec = 2.4e-4*0.05/(hubble*omegab);
  chem[0].fHI   = 1.0-chem[0].felec;
  chem[0].fHII  = chem[0].felec;

  chem[0].fHeIII= 6.0e-12/(yhelium/4.0);
  chem[0].fHeII = 6.0e-12/(yhelium/4.0);
  chem[0].fHeI  = 1.0-chem[0].fHeII-chem[0].fHeIII;

  chem[0].fHM   = 7.6e-21/xhydrog;
  chem[0].fH2I  = 1.5e-6/xhydrog;
  chem[0].fH2II = 7.6e-21/xhydrog;

  chem[0].felec = chem[0].felec-chem[0].fHM+chem[0].fH2II+HELIUM_FACT*(chem[0].fHeII+2.0*chem[0].fHeIII);

  wmol = WMOL(chem[0]);

  dens0= 178.0*rhoc*hubble*hubble*this_run.cosm.omega_b;
  //dens0= rhoc*hubble*hubble*this_run.cosm.omegab;
  dens = dens0*pow((1.0+this_run.znow),3.0);
  nH = dens*xhydrog/mproton;
  nHe = nH*HELIUM_FACT;
  mesh[0].dens = nH/this_run.denstonh*anow3;


  T=1.0e+3;
  //T=1.0e+2;
  mesh[0].uene = T/(this_run.uenetok*wmol);
  mesh[0].duene = 0.0;

  P0=dens*T;

  dtime = 1.0e-4;
  //dtime = 1.0e-5;
  //dtime = 1.0e-6;

  while(this_run.znow > 0.0) {
    float uene;
    nrec=0;
    niter=0;
    //    step_heatcool(sph, chem, &this_run, dtime/2.0);

    chem[0].GammaHI = GammaHI(this_run.znow);
    chem[0].GammaHeI = GammaHeI(this_run.znow);
    chem[0].GammaHeII = GammaHeII(this_run.znow);
    chem[0].GammaHM = GammaHM(this_run.znow);
    chem[0].GammaH2I_I = GammaH2I_I(this_run.znow);
    chem[0].GammaH2I_II = GammaH2I_II(this_run.znow);
    chem[0].GammaH2II_I = GammaH2II_I(this_run.znow);
    chem[0].GammaH2II_II = GammaH2II_II(this_run.znow);

    chem[0].HeatHI = HeatHI(this_run.znow);
    chem[0].HeatHeI = HeatHeI(this_run.znow);
    chem[0].HeatHeII = HeatHeII(this_run.znow);

#if 0
    advance_reaction(mesh, chem, &this_run, dtime);
    mesh[0].duene = 0.0;uene=mesh[0].uene;
    advance_heatcool(mesh, &uene, chem, &this_run, dtime, &nrec, &niter);
    mesh[0].uene = uene;
#else
    mesh[0].duene = 0.0;uene=mesh[0].uene;
    advance_reaction_and_heatcool(mesh, &uene, chem, &this_run, dtime); 
    mesh[0].uene = uene;
#endif

    T=mesh[0].uene*this_run.uenetok*wmol;
    dens = P0/T;
    //dens = dens0*pow((1.0+this_run.znow),3.0);

    nH = dens*xhydrog/mproton;
    mesh[0].dens = nH/this_run.denstonh*anow3;

    wmol = WMOL(chem[0]);
    mesh[0].chem = chem[0];
    float gamma_eos = gamma_total(&mesh[0], &this_run);
    
    this_run.tnow += dtime;
    update_expansion(this_run.tnow, &this_run);
    anow3 = CUBE(this_run.anow);

    printf("%14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e ", 
	   this_run.znow, this_run.tnow, T, nH, 
	   chem[0].fHI,chem[0].fHII,chem[0].fHM,chem[0].fH2I,chem[0].fH2II);
    printf("%14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n",
	   chem[0].fHeI,chem[0].fHeII,chem[0].fHeIII,chem[0].felec,wmol,gamma_eos);


  }
}
