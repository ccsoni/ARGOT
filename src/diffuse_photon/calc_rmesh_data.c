#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "diffuse_photon.h"

#include "diffuse_chemistry.c"

#ifndef TINY
#define TINY (1.0e-31)
#endif

void zero_set_rmesh(struct radiation_mesh *rmesh)
{
  int ix;
  
  //#pragma omp parallel for schedule(dynamic,32)
#pragma omp parallel for schedule(auto)
  for(ix=0; ix<NMESH_LOCAL; ix++){

    rmesh[ix].length = 0.0;

    rmesh[ix].I_nu1 = 0.0;
#ifdef __HELIUM__
    rmesh[ix].I_nu2 = 0.0;
    rmesh[ix].I_nu3 = 0.0;
#ifdef __HELIUM_BB__
    rmesh[ix].I_nu4 = 0.0;
    rmesh[ix].I_nu5 = 0.0;
    rmesh[ix].I_nu6 = 0.0;
#endif
#endif //__HELIUM__
  } 
}


void calc_rmesh_data(struct fluid_mesh *mesh,
		     struct radiation_mesh *rmesh,
		     struct run_param *this_run)
{
  float csecHI   = csectHI(HI_LYMAN_LIMIT+1.0e-7);      //13.6eV
#ifdef __HELIUM__
  float csecHeI  = csectHeI(HeI_LYMAN_LIMIT+1.0e-7);    //24.6eV
  float csecHeII = csectHeII(HeII_LYMAN_LIMIT+1.0e-7);  //54.4eV
#endif //__HELIUM__

  //#pragma omp parallel for schedule(dynamic,32)
#pragma omp parallel for schedule(auto)
  for(int ix=0; ix<NMESH_LOCAL; ix++) {
 
    float wmol, temper;
    double emission; 
    double nH,nHI,nHII,ne;

    wmol   = WMOL(mesh[ix].prev_chem);
    temper = mesh[ix].prev_uene * this_run->uenetok *wmol;
    if(temper < 1.0) temper = 1.0;
    
    nH   = mesh[ix].dens * this_run->denstonh;
#ifdef __COSMOLOGICAL__
    nH  /= CUBE(this_run->anow); 
#endif

    nHI  = mesh[ix].prev_chem.fHI  * nH;
    nHII = mesh[ix].prev_chem.fHII * nH;
    ne   = nHII;

#ifdef __HELIUM__
    double nHeI,nHeII,nHeIII;
    nHeI  = mesh[ix].prev_chem.fHeI  * HELIUM_FACT*nH;
    nHeII = mesh[ix].prev_chem.fHeII * HELIUM_FACT*nH;
    nHeIII= mesh[ix].prev_chem.fHeIII* HELIUM_FACT*nH;

    ne += nHeII + 2.0e0*nHeIII;
#endif //__HELIUM__

    /* absorption */
    double absorptionHI_nu1 = csecHI*nHI;
    rmesh[ix].absorption_nu1 = absorptionHI_nu1 + TINY;   

#ifdef __HELIUM__
    double absorptionHeI_nu2, absorptionHeII_nu3;
    absorptionHeI_nu2 = csecHeI*nHeI;
    absorptionHeII_nu3 = csecHeII*nHeII;
    
    rmesh[ix].absorption_nu2 = absorptionHI_nu1*RCROSS_HI_nu2 + absorptionHeI_nu2 + TINY;
    rmesh[ix].absorption_nu3 = absorptionHI_nu1*RCROSS_HI_nu3 + absorptionHeI_nu2*RCROSS_HeI_nu3 + absorptionHeII_nu3 + TINY;
        
#ifdef __HELIUM_BB__
    rmesh[ix].absorption_nu4 = absorptionHI_nu1*RCROSS_HI_nu4 + TINY;
    rmesh[ix].absorption_nu5 = absorptionHI_nu1*RCROSS_HI_nu5 + TINY;
    rmesh[ix].absorption_nu6 = absorptionHI_nu1*RCROSS_HI_nu6 + absorptionHeI_nu2*RCROSS_HeI_nu6 + TINY;
#endif //__HELIUM_BB__    
#endif //__HELIUM__
    
    /* emissivity and source function */
    emission = (k02_A(temper)-k02_B(temper))*ne*nHII * nuLeV*HI_LYMAN_LIMIT*eV_to_erg/(4.0*PI);
    rmesh[ix].source_func_nu1 = emission/rmesh[ix].absorption_nu1;
    
#ifdef __HELIUM__
    emission = (k04_A(temper)-k04_B(temper))*ne*nHeII * nuLeV*HeI_LYMAN_LIMIT*eV_to_erg/(4.0*PI);
    rmesh[ix].source_func_nu2 = emission/rmesh[ix].absorption_nu2;
    
    emission = (k06_A(temper)-k06_B(temper))*ne*nHeIII * nuLeV*HeII_LYMAN_LIMIT*eV_to_erg/(4.0*PI);
    rmesh[ix].source_func_nu3 = emission/rmesh[ix].absorption_nu3;
    
#ifdef __HELIUM_BB__
    emission = 0.75*k04_B(temper)*ne*nHeII * nuLeV*HeI_BBT_ENG*eV_to_erg/(4.0*PI);
    rmesh[ix].source_func_nu4 = emission/rmesh[ix].absorption_nu4;

    emission = (1.0/6.0)*k04_B(temper)*ne*nHeII * nuLeV*HeI_BBS_ENG*eV_to_erg/(4.0*PI);
    rmesh[ix].source_func_nu5 = emission/rmesh[ix].absorption_nu5;

    emission = k06_B(temper)*ne*nHeIII * nuLeV*HeII_BB_ENG*eV_to_erg/(4.0*PI);
    rmesh[ix].source_func_nu6 = emission/rmesh[ix].absorption_nu6;
#endif //__HELIUM_BB__    
#endif //__HELIUM__
    
    ///zero_set
    rmesh[ix].GHI_tot   = 0.0;
    rmesh[ix].HHI_tot   = 0.0;
#ifdef __HELIUM__
    rmesh[ix].GHeI_tot  = 0.0;
    rmesh[ix].HHeI_tot  = 0.0;

    rmesh[ix].GHeII_tot = 0.0;
    rmesh[ix].HHeII_tot = 0.0;
#endif //__HELIUM__
  }
}
