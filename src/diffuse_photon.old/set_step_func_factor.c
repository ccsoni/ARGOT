#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "diffuse_photon.h"

///type 0:HI 1:HeI 2:HeII 
///unit_nuL=1.0
///nuL = 13.6eV to Hz

/// crs[0][0] : ionization Iseg
/// crs[0][1] : ionization Sfunc
/// crs[1][0] : heating    Iseg
/// crs[1][1] : heating    Sfunc

#define unit_nHI   (HI_LYMAN_LIMIT)
#define unit_nHeI  (HeI_LYMAN_LIMIT)
#define unit_nHeII (HeII_LYMAN_LIMIT)

#define nuL_nHI   (3.2885e15) //13.6eV
#define nuL_nHeI  (5.9483e15) //24.6eV
#define nuL_nHeII (1.3154e16) //54.4eV

void zero_set_step_fact(float step_fact[2][2])
{
  step_fact[0][0] = 0.0;
  step_fact[0][1] = 0.0;
  step_fact[1][0] = 0.0;
  step_fact[1][1] = 0.0; 
}


void set_step_func_factor(struct step_func_factor *step_fact)
{
  double nu,nu_next; 
    
  double crs[2][2],crs_next[2][2];

  zero_set_step_fact(step_fact->HI);
#ifdef __HELIUM__
  zero_set_step_fact(step_fact->HeI);
  zero_set_step_fact(step_fact->HeII);
#endif //__HELIUM__

  /* recombination */
  double block = (double)STEP_FUNC_RANGE/NBIN_STEP_NU;

  for(int inu=0; inu<NBIN_STEP_NU; inu++) {
    nu      = unit_nHI + inu*block;
    nu_next = unit_nHI + (inu+1)*block;;

    if(nu==unit_nHI)      nu      = unit_nHI + 1e-7;
    if(nu_next==unit_nHI) nu_next = unit_nHI + 1e-7;

    crs[0][0]      = 4.0e0*PI*csectHI(nu)/(hplanck*(nu*nuL_nHI));
    crs_next[0][0] = 4.0e0*PI*csectHI(nu_next)/(hplanck*(nu_next*nuL_nHI));
    
    if(inu==0) crs[1][0] = 0.0e0;
    else crs[1][0] = crs[0][0]*(nu     -unit_nHI);
    crs_next[1][0] = crs[0][0]*(nu_next-unit_nHI);
 
    step_fact->HI[0][0] += (crs[0][0] + crs_next[0][0])*0.5e0; 
    step_fact->HI[1][0] += (crs[1][0] + crs_next[1][0])*0.5e0; 
     
#ifdef __HELIUM__
    nu      = unit_nHeI + inu*block;
    nu_next = unit_nHeI + (inu+1)*block;;   
    if(nu==unit_nHeI)      nu      = unit_nHeI + 1e-7;
    if(nu_next==unit_nHeI) nu_next = unit_nHeI + 1e-7;

    crs[0][0]      = 4.0e0*PI*csectHeI(nu)/(hplanck*(nu*nuL_nHeI));
    crs_next[0][0] = 4.0e0*PI*csectHeI(nu_next)/(hplanck*(nu_next*nuL_nHeI));
    
    if(inu==0) crs[1][0] = 0.0e0;
    else crs[1][0] = crs[0][0]*(nu     -unit_nHeI);
    crs_next[1][0] = crs[0][0]*(nu_next-unit_nHeI);
   
    step_fact->HeI[0][0] += (crs[0][0] + crs_next[0][0])*0.5e0; 
    step_fact->HeI[1][0] += (crs[1][0] + crs_next[1][0])*0.5e0; 

    nu      = unit_nHeII + inu*block;
    nu_next = unit_nHeII + (inu+1)*block;;
    if(nu==unit_nHeII)      nu      = unit_nHeII + 1e-7;
    if(nu_next==unit_nHeII) nu_next = unit_nHeII + 1e-7;

    crs[0][0]      = 4.0e0*PI*csectHeII(nu)/(hplanck*(nu*nuL_nHeII));
    crs_next[0][0] = 4.0e0*PI*csectHeII(nu_next)/(hplanck*(nu_next*nuL_nHeII));
    
    if(inu==0) crs[1][0] = 0.0e0;
    else crs[1][0] = crs[0][0]*(nu     -unit_nHeII);
    crs_next[1][0] = crs[0][0]*(nu_next-unit_nHeII);
 
    step_fact->HeII[0][0] += (crs[0][0] + crs_next[0][0])*0.5e0; 
    step_fact->HeII[1][0] += (crs[1][0] + crs_next[1][0])*0.5e0; 

#endif //__HELIUM__
  }

  /* recombination nu width : kT/h (T~1.0e+4K) : ~6.336325e-02 (nuL unit), 0.86 (eV) */
  double tempr, nu_width;
  tempr = 1.0e+4;
  nu_width = (kboltz*tempr/hplanck) / nuL_nHI;
  
  step_fact->HI[0][0] /= NBIN_STEP_NU;
  step_fact->HI[1][0] /= NBIN_STEP_NU/(hplanck*nuL_nHI);
  step_fact->HI[0][1] = 4.0e0*PI*csectHI(unit_nHI+1e-7) / (hplanck*nuL_nHI);
  step_fact->HI[1][1] = 2.0e0*PI*nu_width*csectHI(unit_nHI+1e-7) / unit_nHI;

#ifdef __HELIUM__
  step_fact->HeI[0][0]  /= NBIN_STEP_NU;
  step_fact->HeI[1][0]  /= NBIN_STEP_NU/(hplanck*nuL_nHeI);
  step_fact->HeI[0][1]  = 4.0e0*PI*csectHeI(unit_nHeI+1e-7) / (hplanck*nuL_nHeI);
  step_fact->HeI[1][1]  = 2.0e0*PI*nu_width*csectHeI(unit_nHeI+1e-7) / unit_nHeI;;

  step_fact->HeII[0][0] /= NBIN_STEP_NU;
  step_fact->HeII[1][0] /= NBIN_STEP_NU/(hplanck*nuL_nHeII);
  step_fact->HeII[0][1] = 4.0e0*PI*csectHeII(unit_nHeII+1e-7) / (hplanck*nuL_nHeII);
  step_fact->HeII[1][1] = 2.0e0*PI*nu_width*csectHeII(unit_nHeII+1e-7) / unit_nHeII;;

#endif //__HELIUM__

}

#undef unit_nHI
#undef unit_nHeI
#undef unit_nHeII

#undef nuL_nHI
#undef nuL_nHeI
#undef nuL_nHeII
