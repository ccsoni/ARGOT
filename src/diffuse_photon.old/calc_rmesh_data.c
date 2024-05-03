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

    rmesh[ix].length = 0.0e0;

    rmesh[ix].IHI    = 0.0e0;    
#ifdef __HELIUM__
    rmesh[ix].IHeI   = 0.0e0;
    rmesh[ix].IHeII   = 0.0e0;
#endif //__HELIUM__
  }
  
}


void calc_rmesh_data(struct fluid_mesh *mesh,
		     struct radiation_mesh *rmesh,
		     struct run_param *this_run)
{
  int ix;
  float csecHI   = csectHI(HI_LYMAN_LIMIT+1.0e-7);      //13.6eV
#ifdef __HELIUM__
  float csecHeI  = csectHeI(HeI_LYMAN_LIMIT+1.0e-7);    //24.6eV
  float csecHeII = csectHeII(HeII_LYMAN_LIMIT+1.0e-7);  //54.4eV
#endif //__HELIUM__

  //#pragma omp parallel for schedule(dynamic,32)
#pragma omp parallel for schedule(auto)
  for(ix=0; ix<NMESH_LOCAL; ix++) {
 
    float wmol, temper, absorption;
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

    emission = (k02_A(temper)-k02_B(temper))*ne*nHII * nuLeV*HI_LYMAN_LIMIT*eV_to_erg/(4.0*PI);
    absorption = csecHI * nHI +TINY;
    rmesh[ix].absorptionHI  = absorption;
    rmesh[ix].source_funcHI = emission / absorption;
    
#ifdef __HELIUM__
    emission = (k04_A(temper)-k04_B(temper))*ne*nHeII * nuLeV*HeI_LYMAN_LIMIT*eV_to_erg/(4.0*PI);
    absorption = csecHeI * nHeI +TINY;
    rmesh[ix].absorptionHeI  = absorption;
    rmesh[ix].source_funcHeI = emission / absorption;
    
    emission = (k06_A(temper)-k06_B(temper))*ne*nHeIII * nuLeV*HeII_LYMAN_LIMIT*eV_to_erg/(4.0*PI);
    absorption = csecHeII * nHeII +TINY;
    rmesh[ix].absorptionHeII  = absorption;
    rmesh[ix].source_funcHeII = emission / absorption;

#endif //__HELIUM__
    
    ///zero_set
    rmesh[ix].GHI_tot   = 0.0e0;  rmesh[ix].HHI_tot   = 0.0e0;
#ifdef __HELIUM__
    rmesh[ix].GHeI_tot  = 0.0e0;  rmesh[ix].HHeI_tot  = 0.0e0;
    rmesh[ix].GHeII_tot = 0.0e0;  rmesh[ix].HHeII_tot = 0.0e0;
#endif //__HELIUM__
  }

}

