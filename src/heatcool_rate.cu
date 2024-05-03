#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "constants.h"
#include "chemistry.h"

#include "cooling_rate.cu"

__device__ double calc_heatcool_rate_dev(struct prim_chem *chem,
					 float zred, double nH, double T)
{
  double heat, cool;

  double nHe, nHI, nHII, nHeI, nHeII, nHeIII, nH2I, ne;
  double nHM, nH2II; 
  double Heat_HI;
  
  Heat_HI = chem->HeatHI;
#ifdef __HELIUM__
  double Heat_HeI, Heat_HeII;
  Heat_HeI  = chem->HeatHeI;
  Heat_HeII = chem->HeatHeII;
#endif 
#ifdef __HYDROGEN_MOL__
  double Heat_HM, Heat_H2I_I, Heat_H2I_II, Heat_H2II_I, Heat_H2II_II;
  Heat_HM      = chem->HeatHM;
  Heat_H2I_I   = chem->HeatH2I_I;
  Heat_H2I_II  = chem->HeatH2I_II;
  Heat_H2II_I  = chem->HeatH2II_I;
  Heat_H2II_II = chem->HeatH2II_II;
#endif

  nHI  = nH*chem->fHI;
  nHII = nH*chem->fHII;
#ifdef __HYDROGEN_MOL__
  nHM   = nH*chem->fHM; 
  nH2I  = nH*chem->fH2I;
  nH2II = nH*chem->fH2II; 
#endif
  
  nHe    = nH*HELIUM_FACT;
#ifdef __HELIUM__
  nHeI   = nHe*chem->fHeI;
  nHeII  = nHe*chem->fHeII;
  nHeIII = nHe*chem->fHeIII;
#endif

  //  ne = nHII+nHeII+2.0*nHeIII;
  ne = nH*chem->felec; 

  heat = Heat_HI*nHI;
#ifdef __HELIUM__
  heat += Heat_HeI*nHeI + Heat_HeII*nHeII;
#endif
#ifdef __HYDROGEN_MOL__
  heat += Heat_HM*nHM + (Heat_H2I_I+Heat_H2I_II)*nH2I + (Heat_H2II_I + Heat_H2II_II)*nH2II;
#endif

  cool = (creh2_dev(T)*nHII + cioh1_dev(T)*nHI + cexh1_dev(T)*nHI)*ne
    +    brems_dev(T)*ne*nHII
    +    compt_dev(T,zred)*ne;
#ifdef __HELIUM__
  cool += ((crehe2_dev(T)+crexihe2_dev(T))*nHeII+crehe3_dev(T)*nHeIII)*ne
    +    (ciohe1_dev(T)*nHeI+ciohe2_dev(T)*nHeII)*ne
    +    (cexhe1_dev(T)*ne*nHeI+cexhe2_dev(T)*nHeII)*ne
    +    brems_dev(T)*ne*(nHeII+nHeIII);
#endif
#ifdef __HYDROGEN_MOL__
  cool += h2mol_dev(T,nHI,nH2I);
#endif

  return (heat-cool);
}

