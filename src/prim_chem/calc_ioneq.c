#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "chemistry.h"

#define NITERMAX (40)
#define  TOL (1.0E-4)
#define TINY (1.0e-30)

#define MIN_FRAC (2.0e-4)

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b) )
#endif 

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b) )
#endif

void hydrogen_mol_reaction(struct prim_chem*, double, double);
double calc_dtime_ioneq(struct prim_chem*, double, double);

void calc_ioneq(struct prim_chem *chem, double nH, double T, float zred)
{
  float  Gamma_HI;
  double K01,K02;

  double felec, fHI, fHII;
  double felec_prev;
  double ne;

#ifdef __HELIUM__
  float  Gamma_HeI, Gamma_HeII;
  double K03,K04,K05,K06;

  double fHeI, fHeII, fHeIII;
#endif  
#ifdef __HYDROGEN_MOL__
  float  Gamma_HM, Gamma_H2I_I, Gamma_H2I_II, Gamma_H2II_I, Gamma_H2II_II;
  double K07,K08,K09,K10;
  double K11,K12,K13,K14;
  double K15,K16,K17,K18,K19,K20,K21;

  double fH2I, fH2II, fHM;
#endif
 
  int iter;
  
  Gamma_HI   = GammaHI(zred);
#ifdef __HELIUM__
  Gamma_HeI  = GammaHeI(zred);
  Gamma_HeII = GammaHeII(zred);
#endif
#ifdef __HYDROGEN_MOL__
  Gamma_HM     = GammaHM(zred);
  Gamma_H2I_I  = GammaH2I_I(zred);
  Gamma_H2I_II = GammaH2I_II(zred);
  Gamma_H2II_I = GammaH2II_I(zred);
  Gamma_H2II_II = GammaH2II_II(zred);
#endif

  K01=k01(T);K02=k02(T);
#ifdef __HELIUM__
  K03=k03(T);K04=k04(T);K05=k05(T);K06=k06(T);
#endif
#ifdef __HYDROGEN_MOL__
  K07=k07(T);K08=k08(T);K09=k09(T);K10=k10(T);
  K11=k11(T);K12=k12(T);K13=k13(T);K14=k14(T);K15=k15(T);
  K16=k16(T);K17=k17(T);K18=k18(T);K19=k19(T);K20=k20(T);
  K21=k21(T);
#endif

  if(Gamma_HI == 0.0 
#ifdef __HELIUM__
     && Gamma_HeI == 0.0 && Gamma_HeII == 0.0
#endif
#ifdef __HYDROGEN_MOL__
     && Gamma_HM     == 0.0 && Gamma_H2I_I   == 0.0 && Gamma_H2I_II == 0.0 
     && Gamma_H2II_I == 0.0 && Gamma_H2II_II == 0.0
#endif
     ) {

    
    fHII = K01/(K01+K02);
    fHI = 1.e0-fHII;

#ifdef __HELIUM__
    fHeII = 1.0/(1.0+K04/K03+K05/K06);
    fHeIII = fHeII*K05/K06;
    fHeI = 1.e0-(fHeII+fHeIII);
#endif

    felec = fHII;
#ifdef __HELIUM__
    felec += HELIUM_FACT*(fHeII+2.0*fHeIII);
#endif

    chem->felec = felec;
    chem->fHI = fHI;
    chem->fHII = fHII;
#ifdef __HELIUM__
    chem->fHeI = fHeI;
    chem->fHeII = fHeII;
    chem->fHeIII = fHeIII;
#endif
#ifdef __HYDROGEN_MOL__
    // chem->fH2I   = 0.0;
    //  chem->fH2II  = 0.0;
    //  chem->fHM    = 0.0;

    hydrogen_mol_reaction(chem, nH, T);
#endif

    return;
  }

  felec_prev = 0.5;
  felec = 0.01;

  iter = 0;
  while(fabs(felec-felec_prev)/(felec_prev+1.0e-33)>TOL
	&& iter < NITERMAX) {
    felec_prev = felec;

    ne = felec*nH;

    fHII = (K01+Gamma_HI/ne)/(K01+K02+Gamma_HI/ne);
    fHI = 1.e0-fHII;
#ifdef __HELIUM__
    fHeII = 1.0/(1.0+K04/(K03+Gamma_HeI/ne)+(K05+Gamma_HeII/ne)/K06);
    fHeIII = fHeII*(K05+Gamma_HeII/ne)/K06;
    fHeI = 1.e0-(fHeII+fHeIII);
#endif
    
    fHI  = MAX(0.0, MIN(1.0, fHI));
    fHII = MAX(0.0, MIN(1.0, fHII));
#ifdef __HELIUM__
    fHeI   = MAX(0.0, MIN(1.0, fHeI));
    fHeII  = MAX(0.0, MIN(1.0, fHeII));
    fHeIII = MAX(0.0, MIN(1.0, fHeIII));
#endif
    
    if(fabs(fHI)<1.0E-20) fHI=0.0;
#ifdef __HELIUM__
    if(fabs(fHeI)<1.0E-20) fHeI=0.0;
    if(fabs(fHeIII)<1.0E-20) fHeIII=0.0;
#endif

    felec = fHII;
#ifdef __HELIUM__
    felec += HELIUM_FACT*(fHeII+2.0*fHeIII);
#endif
    felec = 0.5*(felec + felec_prev);

    iter++;

  }
  chem->felec = felec;
  chem->fHI   = fHI;
  chem->fHII  = fHII;
#ifdef __HELIUM__
  chem->fHeI   = fHeI;
  chem->fHeII  = fHeII;
  chem->fHeIII = fHeIII;
#endif
#ifdef __HYDROGEN_MOL__
  //  chem->fH2I   = 0.0;
  //  chem->fH2II  = 0.0;
  //  chem->fHM    = 0.0;

  hydrogen_mol_reaction(chem, nH, T);
#endif

  /*
  if(iter>=NITERMAX) {
    fprintf(stderr,"Loop not converged in calc_ioneq.c\n");
    exit(EXIT_FAILURE);
  } 
  */

#undef TOL  
#undef NITERMAX
}


#ifdef __HYDROGEN_MOL__

/* Abel 1997 fig3 based on Anninos and Norman (1996) */
double set_init_felec(double T)
{
  double felec;
  if(T < 150.0) {
    felec = (2.6e-4)*pow((T/100.0), 3.0);
  }else{
    felec = (1.5e-2)*pow((T/1000.0), 1.5);
  }

  return felec;
}

void hydrogen_mol_reaction(struct prim_chem *chem, double nH, double T)
{
  double nHI, nHII, nHe, nHeI, nHeII, nHeIII, nH2I, nH2II, nHM, ne;

  double scoeff, acoeff;
  double dtime_p; // dtime in the physical unit
  double dtime_safe; // safe timestep calculated in this function.

  double elec_rate_p, elec_rate_m, time_elapsed;
      
  double K_01, K_02, K_03, K_04, K_05, K_06, K_07, K_08, K_09, K_10;
  double K_11, K_12, K_13, K_14, K_15, K_16, K_17, K_18, K_19, K_20;
  double K_21;

  double correct_fact;
  
  chem->fH2I   = 1.0e-30;
  chem->fH2II  = 1.0e-30;
  chem->fHM    = 1.0e-30;

  if(chem->fHII<1.0e-30 || chem->felec<1.0e-30) {
    double min_frac=set_init_felec(T);
    if(chem->felec<min_frac)     chem->felec  = min_frac;
    
    chem->fHII = chem->felec;  //+chem->fHM-chem->fH2II;
    chem->fHI  -= chem->fHII;
#ifdef __HELIUM__

    chem->fHII   = min_frac;
    chem->fHeII  = min_frac;
    chem->fHeIII = 1.0e-10;
    chem->fHeI   = chem->fHeI - (chem->fHeII+chem->fHeIII);
#endif
  }
    
  nHI    = nH*chem->fHI;
  nHII   = nH*chem->fHII;
#ifdef __HYDROGEN_MOL__
  nH2I   = nH*chem->fH2I;
  nH2II  = nH*chem->fH2II;
  nHM    = nH*chem->fHM;
#endif
    
  nHe    = nH*HELIUM_FACT;
#ifdef __HELIUM__
  nHeI   = nHe*chem->fHeI;
  nHeII  = nHe*chem->fHeII;
  nHeIII = nHe*chem->fHeIII;
#endif

  ne = nH*chem->felec;

  // reaction rates
  K_01=k01(T);  K_02=k02(T);
#ifdef __HELIUM__
  K_03=k03(T);  K_04=k04(T);
  K_05=k05(T);  K_06=k06(T);
#endif

  K_07=k07(T);  K_08=k08(T);  K_09=k09(T);  K_10=k10(T);  K_11=k11(T);  K_12=k12(T);
  K_13=k13(T);  K_14=k14(T);  K_15=k15(T);  K_16=k16(T);  K_17=k17(T);  K_18=k18(T);
  K_19=k19(T);  K_20=k20(T);  K_21=k21(T);


  /*
    printf("K0 %e %e %e %e %e %e %e %e %e %e\n",
           K_01,K_02,K_03,K_04,K_05,K_06,K_07,K_08,K_09,K_10);
    printf("K1 %e %e %e %e %e %e %e %e %e %e %e\n",
           K_11,K_12,K_13,K_14,K_15,K_16,K_17,K_18,K_19,K_20,K_21);
  */

  
  //dtime_safe = calc_dtime(mesh, chem, this_run);
  dtime_safe = 1.0;

  time_elapsed=0.0;

  float felec_prev, fHM_prev, fH2I_prev, fH2II_prev;
  felec_prev = fHM_prev = fH2I_prev = fH2II_prev = 100.0;

  int step=0;
  
  while(1){
    time_elapsed += dtime_safe;
    // HI;
    scoeff = K_02*nHII*ne + 3.0*K_11*nH2I*nHI + K_12*nH2I*nHII 
      + (K_13 + 2.0*K_14)*nH2I*ne + 2.0*K_15*nH2I*nH2I + K_16*nHM*ne 
      + 2.0*nHM*(K_17*nHI + K_18*nHII) + 2.0*K_20*nH2II*ne
      + K_21*nH2II*nHM;
    acoeff = (K_01+K_07)*ne + K_08*nHM + K_09*nHII + K_10*nH2II 
      + K_11*nH2I + K_17*nHM;
    
    nHI = (scoeff*dtime_safe + nHI)/(1.0+acoeff*dtime_safe);
    
    // HII;
    scoeff = K_01*nHI*ne;
    acoeff = K_02*ne + K_09*nHI + K_12*nH2I + (K_18 + K_19)*nHM;
    nHII = (scoeff*dtime_safe + nHII)/(1.0 + acoeff*dtime_safe);
    
#ifdef __HELIUM__
    // HeI;
    scoeff = K_04*nHeII*ne;
    acoeff = K_03*ne;
    nHeI = (scoeff*dtime_safe + nHeI)/(1.0 + acoeff*dtime_safe);
    
    // HeII;
    scoeff = (K_03*nHeI+K_06*nHeIII)*ne;
    acoeff = (K_04+K_05)*ne;
    nHeII = (scoeff*dtime_safe + nHeII)/(1.0 + acoeff*dtime_safe);
    
    // HeIII;
    scoeff = K_05*nHeII*ne;
    acoeff = K_06*ne;
    nHeIII = (scoeff*dtime_safe + nHeIII)/(1.0 + acoeff*dtime_safe);
#endif
    
    // electron;
    scoeff = K_01*nHI*ne;
    acoeff = K_02*nHII;
#ifdef __HELIUM__
    scoeff += (K_03*nHeI*ne + K_05*nHeII*ne);
    acoeff += (K_04*nHeII + K_06*nHeIII);
#endif
	       
    scoeff += (K_08*nHM*nHI + K_16*nHM*ne + K_17*nHM*nHI + K_19*nHM*nHII);
    acoeff += (K_07*nHI + K_13*nH2I + K_20*nH2II);
    ne = (scoeff*dtime_safe+ne)/(1.0 + acoeff*dtime_safe);

    // H2I;
    scoeff = K_08*nHM*nHI + K_10*nH2II*nHI + K_21*nH2II*nHM;
    acoeff = K_11*nHI + K_12*nHII + K_14*ne;
    nH2I    = ( scoeff*dtime_safe + nH2I )/(1.0+acoeff*dtime_safe);

    // H2II; 
    scoeff = K_09*nHI*nHII + K_12*nH2I*nHII + K_19*nHM*nHII;
    acoeff = K_10*nHI + K_20*ne + K_21*nHM;
    nH2II = ( scoeff*dtime_safe + nH2II )/(1.0+acoeff*dtime_safe);    

    // H-;
    scoeff = K_07*nHI*ne + K_13*nH2I*ne;  
    acoeff = K_08*nHI + K_16*ne + K_17*nHI + (K_18 + K_19)*nHII + K_21*nH2II; 
    nHM = ( scoeff*dtime_safe + nHM )/(1.0+acoeff*dtime_safe);
    
    chem->fHI   = nHI/nH;
    chem->fHII  = nHII/nH;
    chem->fHM   = nHM/nH;
    chem->fH2I  = nH2I/nH;
    chem->fH2II = nH2II/nH;

    correct_fact = 
      1.0/(chem->fHI+chem->fHII+chem->fHM+2.0*(chem->fH2I+chem->fH2II)+TINY);

    chem->fHI    *= correct_fact;
    chem->fHII   *= correct_fact;
    chem->fHM    *= correct_fact;
    chem->fH2I   *= correct_fact;
    chem->fH2II  *= correct_fact;
    
#ifdef __HELIUM__
    chem->fHeI    = nHeI/(nHe+TINY);
    chem->fHeII   = nHeII/(nHe+TINY);
    chem->fHeIII  = nHeIII/(nHe+TINY);

    correct_fact = 1.0/(chem->fHeI+chem->fHeII+chem->fHeIII+TINY);
    chem->fHeI   *= correct_fact;
    chem->fHeII  *= correct_fact;
    chem->fHeIII *= correct_fact;
#endif

    chem->felec = chem->fHII-chem->fHM+chem->fH2II;
#ifdef __HELIUM__
    chem->felec += (HELIUM_FACT*(chem->fHeII+2.0*chem->fHeIII));
#endif
    
    float error_felec, error_fHM, error_fH2I, error_fH2II;
    error_felec = fabs(felec_prev-chem->felec)/(chem->felec+TINY);
    error_fHM = fabs(fHM_prev-chem->fHM)/(chem->fHM+TINY);
    error_fH2I = fabs(fH2I_prev-chem->fH2I)/(chem->fH2I+TINY);
    error_fH2II = fabs(fH2II_prev-chem->fH2II)/(chem->fH2II+TINY);

    float error=1.0e-4;
    if(error_felec<error && error_fHM<error &&
       error_fH2I<error && error_fH2II<error || step > 1000000) 
      break;

    /*
    printf("%8d %e %e %e %e %e %e %e %e %e %e %e\n",step,time_elapsed,dtime_safe,
	   chem->felec, chem->fHI, chem->fHII,
	   #ifdef __HELIUM__
	   chem->fHeI, chem->fHeII, chem->fHeIII,
	   #else
	   0,0,0,
	   #endif
	   chem->fHM,chem->fH2I,chem->fH2II);
    
    fflush(stdout);
    */
    
    felec_prev = chem->felec;
    fHM_prev   = chem->fHM;
    fH2I_prev  = chem->fH2I;
    fH2II_prev = chem->fH2II;

    dtime_safe = calc_dtime_ioneq(chem, nH, T);
    step++;
  }

}

double calc_dtime_ioneq(struct prim_chem *chem, double nH, double T)
{
  double nHe, ne;
  double nHI, nHII, nHM, nH2I, nH2II, nHeI, nHeII, nHeIII;

  double elec_rate_p, elec_rate_m;

  float K_01, K_02, K_03, K_04, K_05, K_06, K_07, K_08, K_09, K_10;
  float K_11, K_12, K_13, K_14, K_15, K_16, K_17, K_18, K_19, K_20;
  float K_21;

  nHI    = nH*chem->fHI;
  nHII   = nH*chem->fHII;

  nHM    = nH*chem->fHM;
  nH2I   = nH*chem->fH2I;
  nH2II  = nH*chem->fH2II;

    
  nHe    = nH*HELIUM_FACT;
#ifdef __HELIUM__
  nHeI   = nHe*chem->fHeI;
  nHeII  = nHe*chem->fHeII;
  nHeIII = nHe*chem->fHeIII;
#endif
  
  ne = nH*chem->felec;

  // reaction rates  
  K_01=k01(T);  K_02=k02(T);
#ifdef __HELIUM__
  K_03=k03(T);  K_04=k04(T);
  K_05=k05(T);  K_06=k06(T);
#endif /* __HELIUM__ */

  K_07=k07(T);  K_08=k08(T);  K_09=k09(T);  K_10=k10(T);
  K_11=k11(T);  K_12=k12(T);  K_13=k13(T);  K_14=k14(T);
  K_15=k15(T);  K_16=k16(T);  K_17=k17(T);  K_18=k18(T);
  K_19=k19(T);  K_20=k20(T);  K_21=k21(T);

  elec_rate_p = K_01*nHI*ne;
  elec_rate_m = K_02*nHII*ne;

#ifdef __HELIUM__
  elec_rate_p +=
    K_03*nHeI*ne + K_05*nHeII*ne;
  elec_rate_m +=
    K_04*nHeII*ne + K_06*nHeIII*ne;
#endif

  elec_rate_p += 
    K_08*nHM*nHI + K_16*nHM*ne + K_17*nHM*nHI + K_19*nHM*nHII;
  elec_rate_m +=  
    +K_07*nHI*ne + K_13*nH2I*ne + K_20*nH2II*ne;
  
  double dt_safe, dt_safe_elec;
  double dtmin;
  dtmin = 1.0;
  
  dt_safe_elec = ne/(fabs(elec_rate_p-elec_rate_m)+1.e-20f);
  dt_safe = 0.1*dt_safe_elec;
  dt_safe = MAX(dt_safe, dtmin);
  
  // safe timestep in the physical units.
  return dt_safe;
}
#endif




#if 0
int main(int argc, char **argv)
{
  struct prim_chem chem;
  double nH, T;
  float zred;

  nH = 1.0;
  zred = 0.0;
  
  for(int it=0; it<10000; it++) {
    T = 1.0 + it*10.0;
    calc_ioneq(&chem, nH, T, zred);

#if 0
    printf("%e %e %e %e\n",
	   T, chem.felec, chem.fHI, chem.fHII);
#elif 0
    printf("%e %e %e %e %e %e %e\n",
	   T, chem.felec, chem.fHI, chem.fHII,
	   chem.fHeI, chem.fHeII, chem.fHeIII);
#else
    printf("%e %e %e %e %e %e %e %e %e %e\n",
	   T, chem.felec, chem.fHI, chem.fHII,
	   chem.fHeI, chem.fHeII, chem.fHeIII,
	   chem.fH2I, chem.fH2II, chem.fHM);

#endif

  }
  
}
#endif
