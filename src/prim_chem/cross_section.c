//=======================================================================
//
//  Cross section
//     csectHI(nu)      : HI   + photon -> HII   + e
//     csectHeI(nu)     : HeI  + photon -> HeII  + e
//     csectHeII(nu)    : HeII + photon -> HeIII + e
//     csectHM(nu)      : HM   + photon -> HI    + e
//     csectH2II_I(nu)  : H2II + photon -> HI    + HII
//     csectH2I_I(nu)   : H2I  + photon -> H2II  + e
//     csectH2II_II(nu) : H2II + photon -> 2HII  + e
//     csectH2I_II(nu)  : H2I  + photon -> 2HI
//
//     nu: in units of 3.28d+15 [Hz] (13.6eV)
//=======================================================================

#include <math.h>
#include "chemistry.h"
#include "constants.h"

double csectHI(double nu)
{
  double eps;

  if(nu < HI_LYMAN_LIMIT) {
    return (0.0);
  }else{
    eps = sqrt(nu-HI_LYMAN_LIMIT);
    return (6.3e-18/(nu*nu*nu*nu)*exp(4.0-4.0*atan(eps)/eps)
	    /(1.0-exp(-2.0*PI/eps)));
  }
}

double csectHeI(double nu)
{
  if(nu < HeI_LYMAN_LIMIT){
    return (0.0);
  }else{
    return (7.83e-18*(1.66*pow((nu/HeI_LYMAN_LIMIT),-2.05)-0.66*pow((nu/HeI_LYMAN_LIMIT),-3.05)));
  }
}

double csectHeII(double nu)
{
  double eps;

  if(nu < HeII_LYMAN_LIMIT) {
    return (0.0);
  }else{
    eps = sqrt(nu/HeII_LYMAN_LIMIT-1.0);
    return (6.3e-18*256.0/(nu*nu*nu*nu)*exp(4.0-4.0*atan(eps)/eps)
	    /(1.0-exp(-2.0*PI/eps))/4.0);
  }
}

//Abel et al. (1997): process (23) based on De Jong (1972)
double csectHM(double nu) 
{
  double nu1,nuT;
  
  nuT = 0.755;
  nu1 = nu*nuL/eV_to_Hz;

  if(nu1 < 0.755) {
    return (0.0);
  }else{
    return (2.11e-16*pow((nu1-nuT),1.5)/(nu1*nu1*nu1));
  }
}

//Shapiro & Kang (1987) based on Dunn (1968)
double csectH2II_I(double nu)
{
  double nu1, nua, nub, nuc;

  nu1 = nu*nuL/eV_to_Hz;
  nua = 2.65;
  nub = 11.27;
  nuc = 21.0;
  
  if(nu1 > nua && nu1 < nub) {
    return (pow(10.0,(-40.97+6.03*nu1-0.504*nu1*nu1+1.387e-2*nu1*nu1*nu1)));
  }else if(nu1 > nub && nu1 < nuc) {
    return (pow(10.0,(-30.26+2.79*nu1-0.184*nu1*nu1+3.535e-3*nu1*nu1*nu1)));
  }else{
    return (0.0);
  }
}

//Abel et al. (1997): process (24) based on O'Neil & Reinhardt (1978)
double csectH2I_I(double nu)
{
  double nu1, nua, nub, nuc;
  
  nu1 = nu*nuL/eV_to_Hz;
  nua = 15.42;
  nub = 16.5;
  nuc = 17.7;

  if(nu1>nua && nu1<=nub){
    return (6.2e-18*nu1 - 9.4e-17);
  }else if(nu1>nub && nu1<=nuc) {
    return (1.4e-18*nu1 - 1.48e-17);
  }else if(nu1>nuc) {
    return (2.5e-14*pow(nu1,-2.71));
  }else{
    return (0.0);
  }
}

//Abel et al. (1997): process (26) based on Shapiro & Kang (1994)
double csectH2II_II(double nu)
{
  double nu1, nua, nub;

  nu1 = nu*nuL/eV_to_Hz;
  nua = 30.0;
  nub = 90.0;

  if(nu1>=nua && nu1<nub) {
    return (pow(10.0,(-16.926-4.528e-2*nu1+2.238e-4*nu1*nu1+4.245e-7*nu1*nu1*nu1)));
  }else{
    return (0.e0);
  }
}

// the Solomon process 
double csectH2I_II(double nu)
{
  double nu1, nua, nub;

  nu1 = nu*nuL/eV_to_Hz;
  nua = 11.27;
  nub = 13.6;
  
  if(nu1>nua && nu1<nub) {
    return (3.71e-18);
  }else{
    return (0.0);
  }
}

double shielding_func_H2(double NH2, double tmpr) 
{
  double x, bth, fsh;

#if 0 
  // Draine & Bertoldi (1996)
  x = NH2/1.0e14;
  fsh = x < 1.0 ? 1.0 : pow(x,-0.75);
#else
  // Wolcott-Green, Haiman, Bryan (2011)
  x = NH2/5.0e14;
  bth = 0.90854e4*sqrt(tmpr); // (2*kB*T/m)^(1/2)
  bth /= 1.0e5;
  
  fsh = 0.965/pow(1.0+x/bth, 1.1) + 0.035/sqrt(1.0+x)*exp(-8.5e4*sqrt(1.0+x));
#endif

  return fsh;
}

double shielding_func_HI(double NHI)
{
  double fsh, x;

  x = NHI/2.85e23;

  fsh = exp(-0.15*x)/pow(1.0+x, 1.6);

  return fsh;
}
