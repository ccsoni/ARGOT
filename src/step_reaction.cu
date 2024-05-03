#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "run_param.h"
#include "chemistry.h"
#include "fluid.h"

#ifndef TINY
#define TINY (1.0e-31)
#endif

__device__ int advance_heatcool_dev(struct fluid_mesh*, float*, struct prim_chem*, 
                                    struct run_param*, float, int*, int*);

__device__ float calc_dtime_dev(struct fluid_mesh *mesh, 
				struct prim_chem *chem,
				struct run_param *this_run)
{
  float Gamma_HI;
#ifdef __HELIUM__
  double Gamma_HeI, Gamma_HeII;
#endif
#ifdef __HYDROGEN_MOL__
  double Gamma_H2I_I,  Gamma_H2I_II;
  double Gamma_H2II_I, Gamma_H2II_II, Gamma_HM;
#endif

  float nH, nHe, ne, T;
  float nHI, nHII;
#ifdef __HELIUM__
  float nHeI, nHeII, nHeIII;
#endif
#ifdef __HYDROGEN_MOL__  
  float nH2I, nH2II, nHM;
#endif

  double elec_rate_p, elec_rate_m;
  double HI_rate_p, HI_rate_m;

  float K_01, K_02;
#ifdef __HELIUM__
  float K_03, K_04, K_05, K_06;
#endif
#ifdef __HYDROGEN_MOL__
  float K_07, K_08, K_09, K_10, K_11, K_12, K_13, K_14;
  float K_15, K_16, K_17, K_18, K_19, K_20, K_21;
#endif

  Gamma_HI = chem->GammaHI;
#ifdef __HELIUM__
  Gamma_HeI = chem->GammaHeI;
  Gamma_HeII = chem->GammaHeII;
#endif
#ifdef __HYDROGEN_MOL__
  Gamma_HM = chem->GammaHM;
  Gamma_H2I_I = chem->GammaH2I_I;
  Gamma_H2I_II = chem->GammaH2I_II;
  Gamma_H2II_I = chem->GammaH2II_I;
  Gamma_H2II_II = chem->GammaH2II_II;
#endif

  float anow3i;

  anow3i = 1.0f/CUBE(this_run->anow);

  // number density
  nH = mesh->dens*this_run->denstonh*anow3i;

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

  // temperature
  T  = mesh->uene*this_run->uenetok*WMOL(*chem);
#ifdef __TWOTEMP__
  T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */

  // reaction rates
  K_01=k01_dev(T);  K_02=k02_dev(T);
#ifdef __HELIUM__
  K_03=k03_dev(T);  K_04=k04_dev(T);
  K_05=k05_dev(T);  K_06=k06_dev(T);
#endif
#ifdef __HYDROGEN_MOL__
  K_07=k07_dev(T);  K_08=k08_dev(T);
  K_09=k09_dev(T);  K_10=k10_dev(T);
  K_11=k11_dev(T);  K_12=k12_dev(T);
  K_13=k13_dev(T);  K_14=k14_dev(T);
  K_15=k15_dev(T);  K_16=k16_dev(T);
  K_17=k17_dev(T);  K_18=k18_dev(T);
  K_19=k19_dev(T);  K_20=k20_dev(T);
  K_21=k21_dev(T);
#endif

  // compute time step
  elec_rate_p = 
    K_01*nHI*ne           // HI    + e   -> HII   + 2e
    +Gamma_HI*nHI;        // HI    + p   -> HII   + e

  elec_rate_m =  
    K_02*nHII*ne;         // HII   + e   -> H     + p

#ifdef __HELIUM__
  elec_rate_p +=
    K_03*nHeI*ne          // HeI   + e   -> HeII  + 2e
    +K_05*nHeII*ne        // HeII  + e   -> HeIII + 2e
    +Gamma_HeI*nHeI       // HeI   + p   -> HeII  + e
    +Gamma_HeII*nHeII;    // HeII  + p   -> HeIII + e
  elec_rate_m +=
    K_04*nHeII*ne         // HeII  + e   -> HeI   + p
    +K_06*nHeIII*ne;      // HeIII + e   -> HeII  + p
#endif

#ifdef __HYDROGEN_MOL__
  elec_rate_p += 
    K_08*nHM*nHI          // HM    + HI  -> H2I*  + e
    +K_16*nHM*ne          // HM    + e   -> HI    + 2e
    +K_17*nHM*nHI         // HM    + HI  -> 2H    + e
    +K_19*nHM*nHII        // HM    + HII -> H2II  + e
    +Gamma_HM*nHM         // HM    + p   -> HI    + e
    +Gamma_H2I_I*nH2I     // H2I   + p   -> H2II  + e
    +Gamma_H2II_II*nH2II; // H2II  + p   -> 2HII  + e
  elec_rate_m +=  
    +K_07*nHI*ne          // HI    + e   -> HM    + p
    +K_13*nH2I*ne         // H2I   + e   -> HI    + HM
    +K_20*nH2II*ne;       // H2II  + e   -> 2HI
#endif

  HI_rate_p = K_02*nHII*ne;
  HI_rate_m = K_01*nHI*ne + Gamma_HI*nHI;

#ifdef __HYDROGEN_MOL__
  HI_rate_p += 3.0*K_11*nH2I*nHI + K_12*nH2I*nHII 
    + (K_13 + 2.0*K_14)*nH2I*ne + 2.0*K_15*nH2I*nH2I + K_16*nHM*ne 
    + 2.0*nHM*(K_17*nHI + K_18*nHII) + 2.0*K_20*nH2II*ne
    + K_21*nH2II*nHM + Gamma_HM*nHM + Gamma_H2I_I*nH2I 
    + 2.0*Gamma_H2I_II*nH2I;
  HI_rate_m += (K_07*ne + K_08*nHM + K_09*nHII + K_10*nH2II 
                + K_11*nH2I + K_17*nHM)*nHI;
#endif

  double dt_safe, dt_safe_elec, dt_safe_HI;

  double dtmin;

  /* shortest time step : 1 kyr */
  //dtmin = 1.0e3*year;

  /* shortest time step : 1 sec */
  dtmin = 1.0;
 
#if 0
  dt_safe_elec = 0.1*ne/(fabs(elec_rate_p-elec_rate_m)+1.e-20f);
  dt_safe_HI   = 0.1*nHI/(fabs(HI_rate_p-HI_rate_m)+1.e-20f);
  dt_safe = MIN(dt_safe_elec, dt_safe_HI);
  dt_safe = MAX(dt_safe, dtmin);

#else
  dt_safe_elec = ne/(fabs(elec_rate_p-elec_rate_m)+1.e-20f);
  dt_safe_HI   = nHI/(fabs(HI_rate_p-HI_rate_m)+1.e-20f);
  dt_safe = 0.1*dt_safe_elec + 0.00001*dt_safe_HI;
  dt_safe = MAX(dt_safe, dtmin);
#endif

  // safe timestep in the physical units.
  return dt_safe;

}

__device__ void advance_reaction_dev(struct fluid_mesh *mesh, 
				     struct prim_chem *chem,
				     struct run_param *this_run,
				     float dtime)
{
  float Gamma_HI;
#ifdef __HELIUM__
  double Gamma_HeI, Gamma_HeII;
#endif
#ifdef __HYDROGEN_MOL__
  double Gamma_H2I_I,  Gamma_H2I_II;
  double Gamma_H2II_I, Gamma_H2II_II, Gamma_HM;
#endif

  float nH, nHe, ne, T;
  float nHI, nHII;
#ifdef __HELIUM__
  float nHeI, nHeII, nHeIII;
#endif
#ifdef __HYDROGEN_MOL__  
  float nH2I, nH2II, nHM;
#endif

  //  double scoeff, acoeff;
  float scoeff, acoeff;
  double dtime_p, dtime_safe;

  double elec_rate_p, elec_rate_m, time_elapsed;

  float K_01, K_02;
#ifdef __HELIUM__
  float K_03, K_04, K_05, K_06;
#endif
#ifdef __HYDROGEN_MOL__
  float K_07, K_08, K_09, K_10, K_11, K_12, K_13, K_14;
  float K_15, K_16, K_17, K_18, K_19, K_20, K_21;
#endif

  float correct_fact;
  float anow3i;

  anow3i = 1.0f/CUBE(this_run->anow);

  Gamma_HI = chem->GammaHI;
#ifdef __HELIUM__
  Gamma_HeI = chem->GammaHeI;
  Gamma_HeII = chem->GammaHeII;
#endif
#ifdef __HYDROGEN_MOL__
  Gamma_HM = chem->GammaHM;
  Gamma_H2I_I = chem->GammaH2I_I;
  Gamma_H2I_II = chem->GammaH2I_II;
  Gamma_H2II_I = chem->GammaH2II_I;
  Gamma_H2II_II = chem->GammaH2II_II;
#endif

  dtime_p = dtime*this_run->tunit;
  
  // number density
  nH = mesh->dens*this_run->denstonh*anow3i;

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

  // temperature
  T  = mesh->uene*this_run->uenetok*WMOL(*chem);
#ifdef __TWOTEMP__
  T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */

  // reaction rates
  K_01=k01_dev(T);  K_02=k02_dev(T);
#ifdef __HELIUM__
  K_03=k03_dev(T);  K_04=k04_dev(T);
  K_05=k05_dev(T);  K_06=k06_dev(T);
#endif
#ifdef __HYDROGEN_MOL__
  K_07=k07_dev(T);  K_08=k08_dev(T);
  K_09=k09_dev(T);  K_10=k10_dev(T);
  K_11=k11_dev(T);  K_12=k12_dev(T);
  K_13=k13_dev(T);  K_14=k14_dev(T);
  K_15=k15_dev(T);  K_16=k16_dev(T);
  K_17=k17_dev(T);  K_18=k18_dev(T);
  K_19=k19_dev(T);  K_20=k20_dev(T);
  K_21=k21_dev(T);
#endif

  // compute time step
  elec_rate_p = 
    K_01*nHI*ne           // HI    + e   -> HII   + 2e
    +Gamma_HI*nHI;        // HI    + p   -> HII   + e

  elec_rate_m =  
    K_02*nHII*ne;         // HII   + e   -> H     + p

#ifdef __HELIUM__
  elec_rate_p +=
    K_03*nHeI*ne          // HeI   + e   -> HeII  + 2e
    +K_05*nHeII*ne        // HeII  + e   -> HeIII + 2e
    +Gamma_HeI*nHeI       // HeI   + p   -> HeII  + e
    +Gamma_HeII*nHeII;    // HeII  + p   -> HeIII + e
  elec_rate_m +=
    K_04*nHeII*ne         // HeII  + e   -> HeI   + p
    +K_06*nHeIII*ne;      // HeIII + e   -> HeII  + p
#endif

#ifdef __HYDROGEN_MOL__
  elec_rate_p += 
    K_08*nHM*nHI          // HM    + HI  -> H2I*  + e
    +K_16*nHM*ne          // HM    + e   -> HI    + 2e
    +K_17*nHM*nHI         // HM    + HI  -> 2H    + e
    +K_19*nHM*nHII        // HM    + HII -> H2II  + e
    +Gamma_HM*nHM         // HM    + p   -> HI    + e
    +Gamma_H2I_I*nH2I     // H2I   + p   -> H2II  + e
    +Gamma_H2II_II*nH2II; // H2II  + p   -> 2HII  + e
  elec_rate_m +=  
    +K_07*nHI*ne          // HI    + e   -> HM    + p
    +K_13*nH2I*ne         // H2I   + e   -> HI    + HM
    +K_20*nH2II*ne;       // H2II  + e   -> 2HI
#endif

  dtime_safe = 0.1*(ne/(fabs(elec_rate_p-elec_rate_m)+1.0e-20f));

  if(dtime_p < dtime_safe) dtime_safe = dtime_p;

  time_elapsed=0.0f;
  while(time_elapsed<dtime_p){
    if(time_elapsed + dtime_safe > dtime_p) dtime_safe = dtime_p-time_elapsed;
    time_elapsed += dtime_safe;

    // HI;
#ifdef __HYDROGEN_MOL__
    scoeff = K_02*nHII*ne + 3.0f*K_11*nH2I*nHI + K_12*nH2I*nHII 
      + (K_13 + 2.0f*K_14)*nH2I*ne + 2.0f*K_15*nH2I*nH2I + K_16*nHM*ne 
      + 2.0f*nHM*(K_17*nHI + K_18*nHII) + 2.0f*K_20*nH2II*ne
      + K_21*nH2II*nHM + Gamma_HM*nHM + Gamma_H2I_I*nH2I 
      + 2.0f*Gamma_H2I_II*nH2I;
    acoeff = (K_01+K_07)*ne + K_08*nHM + K_09*nHII + K_10*nH2II 
      + K_11*nH2I + K_17*nHM + Gamma_HI;
#else
    scoeff = K_02*nHII*ne;
    acoeff = K_01*ne + Gamma_HI;
#endif
    nHI = (scoeff*dtime_safe + nHI)/(1.0f+acoeff*dtime_safe);

    // HII;
#ifdef __HYDROGEN_MOL__
    scoeff = K_01*nHI*ne + Gamma_HI*nHI + Gamma_H2II_I*nH2II 
      + 2.0f*Gamma_H2II_II*nH2II;
    acoeff = K_02*ne + K_09*nHI + K_12*nH2I + (K_18 + K_19)*nHM;
#else
    scoeff = K_01*nHI*ne + Gamma_HI*nHI;
    acoeff = K_02*ne;
#endif
    nHII = (scoeff*dtime_safe + nHII)/(1.0f+ acoeff*dtime_safe);    

#ifdef __HELIUM__
    // HeI;
    scoeff = K_04*nHeII*ne;
    acoeff = K_03*ne+Gamma_HeI;
    nHeI = (scoeff*dtime_safe + nHeI)/(1.0f + acoeff*dtime_safe);
    
    // HeII;
    scoeff = (K_03*nHeI+K_06*nHeIII)*ne+Gamma_HeI*nHeI;
    acoeff = (K_04+K_05)*ne+Gamma_HeII;
    nHeII = (scoeff*dtime_safe + nHeII)/(1.0f + acoeff*dtime_safe);

    // HeIII;
    scoeff = K_05*nHeII*ne + Gamma_HeII*nHeII;
    acoeff = K_06*ne;
    nHeIII = (scoeff*dtime_safe + nHeIII)/(1.0f + acoeff*dtime_safe);
#endif

    // electron;
    scoeff = K_01*nHI*ne + Gamma_HI*nHI;
    acoeff = K_02*nHII;
#ifdef __HELIUM__
    scoeff += (K_03*nHeI*ne + K_05*nHeII*ne + Gamma_HeI*nHeI + Gamma_HeII*nHeII);
    acoeff += (K_04*nHeII + K_06*nHeIII);
#endif
#ifdef __HYDROGEN_MOL__
    scoeff += (K_08*nHM*nHI + K_16*nHM*ne + K_17*nHM*nHI + K_19*nHM*nHII
	       + Gamma_HM*nHM + Gamma_H2I_I*nH2I + Gamma_H2II_II*nH2II);
    acoeff += (K_07*nHI + K_13*nH2I + K_20*nH2II);
#endif
    ne = (scoeff*dtime_safe+ne)/(1.0f + acoeff*dtime_safe);

#ifdef __HYDROGEN_MOL__
    // H2I;
    scoeff = K_08*nHM*nHI + K_10*nH2II*nHI + K_21*nH2II*nHM;
    acoeff = K_11*nHI + K_12*nHII + K_14*ne + Gamma_H2I_I + Gamma_H2I_II;
    nH2I    = ( scoeff*dtime_safe + nH2I )/(1.0f+acoeff*dtime_safe);

    // H2II; 
    scoeff = K_09*nHI*nHII + K_12*nH2I*nHII + K_19*nHM*nHII + Gamma_H2I_I*nH2I; 
    acoeff = K_10*nHI + K_20*ne + K_21*nHM + Gamma_H2II_I + Gamma_H2II_II; 
    nH2II = ( scoeff*dtime_safe + nH2II )/(1.0f+acoeff*dtime_safe);    

    // H-;
    scoeff = K_07*nHI*ne + K_13*nH2I*ne;  
    acoeff = K_08*nHI + K_16*ne + K_17*nHI + (K_18 + K_19)*nHII + K_21*nH2II 
      + Gamma_HM; 
    nHM = ( scoeff*dtime_safe + nHM )/(1.0f+acoeff*dtime_safe);
#endif 


    chem->fHI   = nHI/nH;
    chem->fHII  = nHII/nH;
#ifdef __HYDROGEN_MOL__
    chem->fHM   = nHM/nH;
    chem->fH2I  = nH2I/nH;
    chem->fH2II = nH2II/nH;
#endif

#ifdef __HYDROGEN_MOL__
    correct_fact = 
      1.0f/(chem->fHI+chem->fHII+chem->fHM+2.0*(chem->fH2I+chem->fH2II));
#else
    correct_fact = 1.0f/(chem->fHI+chem->fHII);
#endif

    chem->fHI    *= correct_fact;
    chem->fHII   *= correct_fact;
#ifdef __HYDROGEN_MOL__
    chem->fHM    *= correct_fact;
    chem->fH2I   *= correct_fact;
    chem->fH2II  *= correct_fact;
#endif

#ifdef __HELIUM__
    chem->fHeI    = nHeI/(nHe+TINY);
    chem->fHeII   = nHeII/(nHe+TINY);
    chem->fHeIII  = nHeIII/(nHe+TINY);

    correct_fact = 1.0f/(chem->fHeI+chem->fHeII+chem->fHeIII+TINY);
    chem->fHeI   *= correct_fact;
    chem->fHeII  *= correct_fact;
    chem->fHeIII *= correct_fact;
#endif

#ifdef __HYDROGEN_MOL__
    chem->felec = chem->fHII-chem->fHM+chem->fH2II;
#else
    chem->felec = chem->fHII;
#endif
#ifdef __HELIUM__
    chem->felec += (HELIUM_FACT*(chem->fHeII+2.0*chem->fHeIII)+TINY);
#endif

  }

}

#define DtFactor (0.1) 

__device__ void advance_reaction_and_heatcool_dev(struct fluid_mesh *mesh, 
                                                  float *uene,
                                                  struct prim_chem *chem,
                                                  struct run_param *this_run,
                                                  float dtime)
{
  float Gamma_HI;
#ifdef __HELIUM__
  double Gamma_HeI, Gamma_HeII;
#endif
#ifdef __HYDROGEN_MOL__
  double Gamma_H2I_I,  Gamma_H2I_II;
  double Gamma_H2II_I, Gamma_H2II_II, Gamma_HM;
#endif

  float nH, nHe, ne, T;
  float nHI, nHII;
#ifdef __HELIUM__
  float nHeI, nHeII, nHeIII;
#endif
#ifdef __HYDROGEN_MOL__  
  float nH2I, nH2II, nHM;
#endif

  //  double scoeff, acoeff;
  float scoeff, acoeff;
  double dtime_p, dtime_safe;

  double time_elapsed;

  float K_01, K_02;
#ifdef __HELIUM__
  float K_03, K_04, K_05, K_06;
#endif
#ifdef __HYDROGEN_MOL__
  float K_07, K_08, K_09, K_10, K_11, K_12, K_13, K_14;
  float K_15, K_16, K_17, K_18, K_19, K_20, K_21;
#endif

  float correct_fact;
  float anow3i;

  anow3i = 1.0f/CUBE(this_run->anow);

  Gamma_HI = chem->GammaHI;
#ifdef __HELIUM__
  Gamma_HeI = chem->GammaHeI;
  Gamma_HeII = chem->GammaHeII;
#endif
#ifdef __HYDROGEN_MOL__
  Gamma_HM = chem->GammaHM;
  Gamma_H2I_I = chem->GammaH2I_I;
  Gamma_H2I_II = chem->GammaH2I_II;
  Gamma_H2II_I = chem->GammaH2II_I;
  Gamma_H2II_II = chem->GammaH2II_II;
#endif

  dtime_p = dtime*this_run->tunit;
  
  // number density
  nH = mesh->dens*this_run->denstonh*anow3i;

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

  // temperature
  T  = mesh->uene*this_run->uenetok*WMOL(*chem);
#ifdef __TWOTEMP__
  T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */

  // reaction rates
  K_01=k01_dev(T);  K_02=k02_dev(T);
#ifdef __HELIUM__
  K_03=k03_dev(T);  K_04=k04_dev(T);
  K_05=k05_dev(T);  K_06=k06_dev(T);
#endif
#ifdef __HYDROGEN_MOL__
  K_07=k07_dev(T);  K_08=k08_dev(T);
  K_09=k09_dev(T);  K_10=k10_dev(T);
  K_11=k11_dev(T);  K_12=k12_dev(T);
  K_13=k13_dev(T);  K_14=k14_dev(T);
  K_15=k15_dev(T);  K_16=k16_dev(T);
  K_17=k17_dev(T);  K_18=k18_dev(T);
  K_19=k19_dev(T);  K_20=k20_dev(T);
  K_21=k21_dev(T);
#endif


  dtime_safe = DtFactor * calc_dtime_dev(mesh, chem, this_run);

  if(dtime_p < dtime_safe) dtime_safe = dtime_p;

  time_elapsed=0.0f;
  while(time_elapsed<dtime_p){
    if(time_elapsed + dtime_safe > dtime_p) dtime_safe = dtime_p-time_elapsed;
    time_elapsed += dtime_safe;

    // HI;
#ifdef __HYDROGEN_MOL__
    scoeff = K_02*nHII*ne + 3.0f*K_11*nH2I*nHI + K_12*nH2I*nHII 
      + (K_13 + 2.0f*K_14)*nH2I*ne + 2.0f*K_15*nH2I*nH2I + K_16*nHM*ne 
      + 2.0f*nHM*(K_17*nHI + K_18*nHII) + 2.0f*K_20*nH2II*ne
      + K_21*nH2II*nHM + Gamma_HM*nHM + Gamma_H2I_I*nH2I 
      + 2.0f*Gamma_H2I_II*nH2I;
    acoeff = (K_01+K_07)*ne + K_08*nHM + K_09*nHII + K_10*nH2II 
      + K_11*nH2I + K_17*nHM + Gamma_HI;
#else
    scoeff = K_02*nHII*ne;
    acoeff = K_01*ne + Gamma_HI;
#endif
    nHI = (scoeff*dtime_safe + nHI)/(1.0f+acoeff*dtime_safe);

    // HII;
#ifdef __HYDROGEN_MOL__
    scoeff = K_01*nHI*ne + Gamma_HI*nHI + Gamma_H2II_I*nH2II 
      + 2.0f*Gamma_H2II_II*nH2II;
    acoeff = K_02*ne + K_09*nHI + K_12*nH2I + (K_18 + K_19)*nHM;
#else
    scoeff = K_01*nHI*ne + Gamma_HI*nHI;
    acoeff = K_02*ne;
#endif
    nHII = (scoeff*dtime_safe + nHII)/(1.0f+ acoeff*dtime_safe);    

#ifdef __HELIUM__
    // HeI;
    scoeff = K_04*nHeII*ne;
    acoeff = K_03*ne+Gamma_HeI;
    nHeI = (scoeff*dtime_safe + nHeI)/(1.0f + acoeff*dtime_safe);
    
    // HeII;
    scoeff = (K_03*nHeI+K_06*nHeIII)*ne+Gamma_HeI*nHeI;
    acoeff = (K_04+K_05)*ne+Gamma_HeII;
    nHeII = (scoeff*dtime_safe + nHeII)/(1.0f + acoeff*dtime_safe);

    // HeIII;
    scoeff = K_05*nHeII*ne + Gamma_HeII*nHeII;
    acoeff = K_06*ne;
    nHeIII = (scoeff*dtime_safe + nHeIII)/(1.0f + acoeff*dtime_safe);
#endif


    // electron;
    scoeff = K_01*nHI*ne + Gamma_HI*nHI;
    acoeff = K_02*nHII;
#ifdef __HELIUM__
    scoeff += (K_03*nHeI*ne + K_05*nHeII*ne + Gamma_HeI*nHeI + Gamma_HeII*nHeII);
    acoeff += (K_04*nHeII + K_06*nHeIII);
#endif
#ifdef __HYDROGEN_MOL__
    scoeff += (K_08*nHM*nHI + K_16*nHM*ne + K_17*nHM*nHI + K_19*nHM*nHII
               + Gamma_HM*nHM + Gamma_H2I_I*nH2I + Gamma_H2II_II*nH2II);
    acoeff += (K_07*nHI + K_13*nH2I + K_20*nH2II);
#endif
    ne = (scoeff*dtime_safe+ne)/(1.0f + acoeff*dtime_safe);

#ifdef __HYDROGEN_MOL__
    // H2I;
    scoeff = K_08*nHM*nHI + K_10*nH2II*nHI + K_21*nH2II*nHM;
    acoeff = K_11*nHI + K_12*nHII + K_14*ne + Gamma_H2I_I + Gamma_H2I_II;
    nH2I    = ( scoeff*dtime_safe + nH2I )/(1.0f+acoeff*dtime_safe);

    // H2II; 
    scoeff = K_09*nHI*nHII + K_12*nH2I*nHII + K_19*nHM*nHII + Gamma_H2I_I*nH2I; 
    acoeff = K_10*nHI + K_20*ne + K_21*nHM + Gamma_H2II_I + Gamma_H2II_II; 
    nH2II = ( scoeff*dtime_safe + nH2II )/(1.0f+acoeff*dtime_safe);    

    // H-;
    scoeff = K_07*nHI*ne + K_13*nH2I*ne;  
    acoeff = K_08*nHI + K_16*ne + K_17*nHI + (K_18 + K_19)*nHII + K_21*nH2II 
      + Gamma_HM; 
    nHM = ( scoeff*dtime_safe + nHM )/(1.0f+acoeff*dtime_safe);
#endif 


    chem->fHI   = nHI/nH;
    chem->fHII  = nHII/nH;
#ifdef __HYDROGEN_MOL__
    chem->fHM   = nHM/nH;
    chem->fH2I  = nH2I/nH;
    chem->fH2II = nH2II/nH;
#endif

#ifdef __HYDROGEN_MOL__
    correct_fact = 
      1.0f/(chem->fHI+chem->fHII+chem->fHM+2.0*(chem->fH2I+chem->fH2II));
#else
    correct_fact = 1.0f/(chem->fHI+chem->fHII);
#endif

    chem->fHI    *= correct_fact;
    chem->fHII   *= correct_fact;
#ifdef __HYDROGEN_MOL__
    chem->fHM    *= correct_fact;
    chem->fH2I   *= correct_fact;
    chem->fH2II  *= correct_fact;
#endif

#ifdef __HELIUM__
    chem->fHeI    = nHeI/(nHe+TINY);
    chem->fHeII   = nHeII/(nHe+TINY);
    chem->fHeIII  = nHeIII/(nHe+TINY);

    correct_fact = 1.0f/(chem->fHeI+chem->fHeII+chem->fHeIII+TINY);
    chem->fHeI   *= correct_fact;
    chem->fHeII  *= correct_fact;
    chem->fHeIII *= correct_fact;
#endif

#ifdef __HYDROGEN_MOL__
    chem->felec = chem->fHII-chem->fHM+chem->fH2II;
#else
    chem->felec = chem->fHII;
#endif
#ifdef __HELIUM__
    chem->felec += (HELIUM_FACT*(chem->fHeII+2.0*chem->fHeIII)+TINY);
#endif

    int nrec=0, niter;
    
    dtime_safe /= this_run->tunit;
    advance_heatcool_dev(mesh, uene, chem, this_run, dtime_safe, &nrec, &niter);
    //    mesh->prev_uene = (*uene);
    

    // new time-step 
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
    
    // temperature
    //T  = mesh->uene*this_run->uenetok*WMOL(*chem);
    T  = (*uene)*this_run->uenetok*WMOL(*chem);
#ifdef __TWOTEMP__
    T *= mesh->te_scaled;
#endif /* __TWOTEMP__ */
    
    // reaction rates
    K_01=k01_dev(T); K_02=k02_dev(T);
#ifdef __HELIUM__
    K_03=k03_dev(T); K_04=k04_dev(T); K_05=k05_dev(T); K_06=k06_dev(T);
#endif
#ifdef __HYDROGEN_MOL__
    K_07=k07_dev(T); K_08=k08_dev(T); K_09=k09_dev(T); K_10=k10_dev(T);
    K_11=k11_dev(T); K_12=k12_dev(T); K_13=k13_dev(T); K_14=k14_dev(T);
    K_15=k15_dev(T); K_16=k16_dev(T); K_17=k17_dev(T); K_18=k18_dev(T);
    K_19=k19_dev(T); K_20=k20_dev(T); K_21=k21_dev(T);
#endif

    dtime_safe = DtFactor * calc_dtime_dev(mesh, chem, this_run);

  }
}
