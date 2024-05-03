#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "constants.h"
#include "chemistry.h"

double calc_heatcool_rate(struct prim_chem *chem,
			  float zred, double nH, double T)
{
  double heat, cool;

  double nHe, nHI, nHII, nHeI, nHeII, nHeIII, ne;
  double nHM, nH2I, nH2II; 

#ifdef __UV_BACKGROUND__
  static float zred_prev=-1.0;
  static double Heat_HI=0.0, Heat_HeI=0.0, Heat_HeII=0.0;
  static double Heat_HM=0.0, Heat_H2I_I=0.0;
  static double Heat_H2I_II=0.0, Heat_H2II_I=0.0, Heat_H2II_II=0.0;
#else
  //  float zred_prev;
  double Heat_HI=0.0, Heat_HeI=0.0, Heat_HeII=0.0;
  double Heat_HM=0.0, Heat_H2I_I=0.0;
  double Heat_H2I_II=0.0, Heat_H2II_I=0.0, Heat_H2II_II=0.0;
#endif

#ifdef __UV_BACKGROUND__
  if(zred_prev != zred) {
    Heat_HI   = HeatHI(zred);
#ifdef __HELIUM__
    Heat_HeI  = HeatHeI(zred);
    Heat_HeII = HeatHeII(zred);
#endif
    zred_prev = zred;
  }
#else/* ! __UV_BACKGROUND__ */
  Heat_HI = chem->HeatHI;
#ifdef __HELIUM__
  Heat_HeI  = chem->HeatHeI;
  Heat_HeII = chem->HeatHeII;
#endif 
#ifdef __HYDROGEN_MOL__
  Heat_HM = chem->HeatHM;
  Heat_H2I_I = chem->HeatH2I_I;
  Heat_H2I_II = chem->HeatH2I_II;
  Heat_H2II_I = chem->HeatH2II_I;
  Heat_H2II_II = chem->HeatH2II_II;
#endif   
#endif /* __UV_BACKGROUND__ */

  nHI  = nH*chem->fHI;
  nHII = nH*chem->fHII;
#ifdef __HYDROGEN_MOL__
  nHM = nH*chem->fHM; 
  nH2I = nH*chem->fH2I;
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

  cool = (creh2(T)*nHII + cioh1(T)*nHI + cexh1(T)*nHI)*ne
    +    brems(T)*ne*nHII
    +    compt(T,zred)*ne;
#ifdef __HELIUM__
  cool += ((crehe2(T)+crexihe2(T))*nHeII+crehe3(T)*nHeIII)*ne
    +    (ciohe1(T)*nHeI+ciohe2(T)*nHeII)*ne
    +    (cexhe1(T)*ne*nHeI+cexhe2(T)*nHeII)*ne
    +    brems(T)*ne*(nHeII+nHeIII);
#endif
#ifdef __HYDROGEN_MOL__
  cool += h2mol(T,nHI,nH2I);
#endif

  return (heat-cool);
}

double calc_heating_rate(struct prim_chem *chem, 
			 float zred, double nH, double T)
{
  double heat;

  double nHe, nHI, nHII, nHeI, nHeII, nHeIII;
  double nHM, nH2I, nH2II; 

#ifdef __UV_BACKGROUND__
  static float zred_prev=-1.0;
  static double Heat_HI, Heat_HeI, Heat_HeII;
  static double Heat_HM=0.0, Heat_H2I_I=0.0;
  static double Heat_H2I_II=0.0, Heat_H2II_I=0.0, Heat_H2II_II=0.0;
#else
  //float zred_prev=-1.0;
  double Heat_HI=0.0, Heat_HeI=0.0, Heat_HeII=0.0;
  double Heat_HM=0.0, Heat_H2I_I=0.0;
  double Heat_H2I_II=0.0, Heat_H2II_I=0.0, Heat_H2II_II=0.0;
#endif

#ifdef __UV_BACKGROUND__
  if(zred_prev != zred) {
    Heat_HI   = HeatHI(zred);
#ifdef __HELIUM__
    Heat_HeI  = HeatHeI(zred);
    Heat_HeII = HeatHeII(zred);
#endif
    zred_prev = zred;
  }
#else /* ! __UV_BACKGROUND__ */
  Heat_HI = chem->HeatHI;
#ifdef __HELIUM__
  Heat_HeI  = chem->HeatHeI;
  Heat_HeII = chem->HeatHeII;
#endif 
#ifdef __HYDROGEN_MOL__
  Heat_HM = chem->HeatHM;
  Heat_H2I_I = chem->HeatH2I_I;
  Heat_H2I_II = chem->HeatH2I_II;
  Heat_H2II_I = chem->HeatH2II_I;
  Heat_H2II_II = chem->HeatH2II_II;
#endif
#endif

  nHI  = nH*chem->fHI;
  nHII = nH*chem->fHII;
  
  nHe    = nH*HELIUM_FACT;
#ifdef __HELIUM__
  nHeI   = nHe*chem->fHeI;
  nHeII  = nHe*chem->fHeII;
  nHeIII = nHe*chem->fHeIII;
#endif
#ifdef __HYDROGEN_MOL__
  nHM = nH*chem->fHM; 
  nH2I = nH*chem->fH2I;
  nH2II = nH*chem->fH2II; 
#endif

  heat = Heat_HI*nHI;
#ifdef __HELIUM__
  heat += Heat_HeI*nHeI + Heat_HeII*nHeII;
#endif
#ifdef __HYDROGEN_MOL__
  heat += Heat_HM*nHM + (Heat_H2I_I+Heat_H2I_II)*nH2I + (Heat_H2II_I + Heat_H2II_II)*nH2II;
#endif

  return heat;
}

double calc_cooling_rate(struct prim_chem *chem,
			 float zred, double nH, double T)
{
  double cool;
  
  float ne;
  float nHI, nHII, nH2I;
  float nHe, nHeI, nHeII, nHeIII;

  nHI  = nH*chem->fHI;
  nHII = nH*chem->fHII;
#ifdef __HYDROGEN_MOL__
  nH2I = nH*chem->fH2I;
#endif

  nHe    = nH*HELIUM_FACT;
#ifdef __HELIUM__
  nHeI   = nHe*chem->fHeI;
  nHeII  = nHe*chem->fHeII;
  nHeIII = nHe*chem->fHeIII;
#endif

  ne = nHII;
#ifdef __HELIUM__
  ne += nHeII+2.0*nHeIII;
#endif 

  cool = (creh2(T)*nHII + cioh1(T)*nHI + cexh1(T)*nHI)*ne
    +    brems(T)*ne*nHII
    +    compt(T,zred)*ne;
#ifdef __HELIUM__
  cool += ((crehe2(T)+crexihe2(T))*nHeII+crehe3(T)*nHeIII)*ne
    +    (ciohe1(T)*nHeI+ciohe2(T)*nHeII)*ne
    +    (cexhe1(T)*ne*nHeI+cexhe2(T)*nHeII)*ne
    +    brems(T)*ne*(nHeII+nHeIII);
#endif
#ifdef __HYDROGEN_MOL__
  cool += h2mol(T,nHI,nH2I);
#endif

#if 0
#ifdef __HYDROGEN_MOL__
  cool = (creh2(T)*nHII+(crehe2(T)+crexihe2(T))*nHeII+crehe3(T)*nHeIII)*ne
    +    (cioh1(T)*nHI+ciohe1(T)*nHeI+ciohe2(T)*nHeII)*ne
    +    (cexh1(T)*nHI+cexhe1(T)*ne*nHeI+cexhe2(T)*nHeII)*ne
    +    brems(T)*ne*(nHII+nHeII+nHeIII)
    +    compt(T,zred)*ne
    +    h2mol(T,nHI,nH2I);
#else
  cool = (creh2(T)*nHII+(crehe2(T)+crexihe2(T))*nHeII+crehe3(T)*nHeIII)*ne
    +    (cioh1(T)*nHI+ciohe1(T)*nHeI+ciohe2(T)*nHeII)*ne
    +    (cexh1(T)*nHI+cexhe1(T)*ne*nHeI+cexhe2(T)*nHeII)*ne
    +    brems(T)*ne*(nHII+nHeII+nHeIII)
    +    compt(T,zred)*ne;
#endif
#endif

  return cool;
}
