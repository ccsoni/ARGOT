//=======================================================================
//  Photoionization rates
//=======================================================================
//     Photoionization processes
//-----------------------------------------------------------------------
//     HI   + photon -> HII   + e
//     HeII + photon -> HeIII + e
//     HeI  + photon -> HeII  + e
//     HM   + photon -> HI    + e
//     H2II + photon -> HI    + HII
//     H2I  + photon -> H2II  + e
//     H2II + photon -> 2HII  + e
//     H2I  + photon -> 2HI
//=======================================================================
//     J_UV(nu)             : UV background radiation in units of 
//                                              10^{-22} erg/src/cm/sr/Hz
//-----------------------------------------------------------------------
//     Gamma(specie)        : Photoionization rate
//     photofuncHI(nu)      : J(nu)*\sigma(nu) for csectHI
//     photofuncHeI(nu)     : J(nu)*\sigma(nu) for csectHeI
//     photofuncHeII(nu)    : J(nu)*\sigma(nu) for csectHeII
//     photofuncHM(nu)      : J(nu)*\sigma(nu) for csectHM
//     photofuncH2I_I(nu)   : J(nu)*\sigma(nu) for csectH2I_I
//     photofuncH2I_II(nu)  : J(nu)*\sigma(nu) for csectH2I_II
//     photofuncH2II_I(nu)  : J(nu)*\sigma(nu) for csectH2II_I
//     photofuncH2II_II(nu) : J(nu)*\sigma(nu) for csectH2II_II
//-----------------------------------------------------------------------
//     Heat(specie)         : Heating rate
//     heatfuncHI(nu)       : J(nu)*\sigma(nu)*(nu-nui) for csectHI
//     heatfuncHeI(nu)      : J(nu)*\sigma(nu)*(nu-nui) for csectHeI
//     heatfuncHeII(nu)     : J(nu)*\sigma(nu)*(nu-nui) for csectHeII
//=======================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "constants.h"
#include "chemistry.h"

extern double csectHI(double);
extern double csectHeI(double);
extern double csectHeII(double);
extern double csectHM(double);
extern double csectH2I_I(double);
extern double csectH2I_II(double);
extern double csectH2II_I(double);
extern double csectH2II_II(double);


double midexp(double (*funk)(double,float),double aa,double bb,int n,float zred)
{
  double x,tnm,sum,del,ddel,a,b;
  static double s;
  int it,j;

  b=exp(-aa);
  a=0.0;
  if (n == 1) {
    //return (s=(b-a)*FUNC(0.5*(a+b)));
    return (s=(b-a)*(*funk)(-log(0.5*(a+b)),zred)/(0.5*(a+b)));
  } else {
    for(it=1,j=1;j<n-1;j++) it *= 3;
    tnm=it;
    del=(b-a)/(3.0*tnm);
    ddel=del+del;
    x=a+0.5*del;
    sum=0.0;
    for (j=1;j<=it;j++) {
      //sum += FUNC(x);
      sum += (*funk)(-log(x),zred)/x;
      x += ddel;
      //sum += FUNC(x);
      sum += (*funk)(-log(x),zred)/x;
      x += del;
    }
    s=(s+(b-a)*sum/tnm)/3.0;
    return s;
  }
}

double midpnt(double (*func)(double,float),double a,double b,int n,float zred)
{
  double x,tnm,sum,del,ddel;
  static double s;
  int it,j;

  if (n == 1) {
    //return (s=(b-a)*FUNC(0.5*(a+b)));
    return (s=(b-a)*(*func)(0.5*(a+b),zred));
  } else {
    for(it=1,j=1;j<n-1;j++) it *= 3;
    tnm=it;
    del=(b-a)/(3.0*tnm);
    ddel=del+del;
    x=a+0.5*del;
    sum=0.0;
    for (j=1;j<=it;j++) {
      //sum += FUNC(x);
      sum += (*func)(x,zred);
      x += ddel;
      //sum += FUNC(x);
      sum += (*func)(x,zred);
      x += del;
    }
    s=(s+(b-a)*sum/tnm)/3.0;
    return s;
  }
}

double CUBA_J_UV(double nu, float zred)
{

  // nu is in units of 3.28d15 Hz (13.6eV)
  // intensity at nu=nuL in units of 10^{-22} erg/src/cm^{-2}/sr^{-1}/Hz^{-1}
#define CUBA_NU_NBIN (432)
#define CUBA_ZRED_NBIN (49)
  void hunt(float*, int, float, int*);

  static float cuba_table[CUBA_ZRED_NBIN][CUBA_NU_NBIN];
  static float cuba_lognu[CUBA_NU_NBIN];
  static float cuba_zred[CUBA_NU_NBIN];

  float lognu, logJ;
  float fnu, fzred;

  int inu, izred;

  static FILE *CUBA_data_file=NULL;

  if(CUBA_data_file == NULL) {
    CUBA_data_file = fopen("CUBA_background.dat", "r");

    for(izred=0;izred<CUBA_ZRED_NBIN;izred++) {
      for(inu=0;inu<CUBA_NU_NBIN;inu++) { 
        fscanf(CUBA_data_file,"%f %f %f\n", 
	       &cuba_zred[izred],
	       &cuba_lognu[inu],
	       &cuba_table[izred][inu]);
      }

    }
    FILE *fl; 
    fl = fopen("uvb.txt", "w"); 
    for (inu = 0; inu < CUBA_NU_NBIN; inu++) {
      fprintf(fl, "%e %e\n", pow(1.e1, cuba_lognu[inu]), pow(1.e1, cuba_table[0][inu])); 
    }
    fclose(fl); 
    //exit(1); 
  }

  lognu = log10(nu*3.28e15);

  hunt(cuba_zred,  CUBA_ZRED_NBIN, zred,  &izred);
  hunt(cuba_lognu, CUBA_NU_NBIN,   lognu, &inu);

  if( inu >= 0 && inu < (CUBA_NU_NBIN-2) ) {
    fnu = (lognu-cuba_lognu[inu])/(cuba_lognu[inu+1]-cuba_lognu[inu]);
    if( izred >= 0 && izred <= (CUBA_ZRED_NBIN-2)) {
      fzred = (zred-cuba_zred[izred])/(cuba_zred[izred+1]-cuba_zred[izred]);
      
      logJ = (1.0-fnu)*(1.0-fzred)*cuba_table[izred][inu] 
        +    fnu*(1.0-fzred)*cuba_table[izred][inu+1] 
        +    (1.0-fnu)*fzred*cuba_table[izred+1][inu] 
        +    fnu*fzred*cuba_table[izred+1][inu+1];
    }else if(izred >= CUBA_ZRED_NBIN-1) {
      return 0.0;
    }else if(izred < 0) {
      izred = 0;
      logJ = (1.0-fnu)*cuba_table[izred][inu] + fnu*cuba_table[izred][inu+1];
    }
    return (pow(10.0, logJ)/1.e-22);
  }else{
    return (0.0);
  }
  
#undef CUBA_ENE_NBIN
#undef CUBA_ZRED_NBIN
}

double NULL_J_UV(double nu, float zred)
{
  return (0.0);
}

double J_UV(double nu, float zred)
{
  // nu is in units of 3.28d15 Hz (13.6eV)

  // intensity at nu=nuL in units of 10^{-22} erg/src/cm^{-2}/sr^{-1}/Hz^{-1}
  double J0; 

  if (zred>6.0){
    J0=0.0;
  }else if (zred>4.0){
    J0=(3.0*(1.0 - zred/6.0))*(3.0*(1.0 - zred/6.0));
  }else if (zred>2.0){
    J0=1.0;
  }else{
    J0=(1.0 + zred)*(1.0 + zred)*(1.0 + zred)*(1.0 + zred)/81.0;
  }

  if(zred<0.0) J0=0.0;

  return (J0/nu);
}

#define NITER (5)
#define ALLOWED_ERR (1.e-5)

#ifdef __CUBA__
#define J_UV CUBA_J_UV
#elif __NOUV__
#define J_UV NULL_J_UV
#endif

double photofuncHI(double nu, float zred)
{
  return (4.0*PI*J_UV(nu, zred)*csectHI(nu)/nu);
}

double photofuncHeI(double nu, float zred)
{
  return (4.0*PI*J_UV(nu, zred)*csectHeI(nu)/nu);
}

double photofuncHeII(double nu, float zred)
{
  return (4.0*PI*J_UV(nu, zred)*csectHeII(nu)/nu);
}

double photofuncHM(double nu, float zred)
{
  return (4.0*PI*J_UV(nu, zred)*csectHM(nu)/nu);
}

double photofuncH2I_I(double nu, float zred)
{
  return (4.0*PI*J_UV(nu, zred)*csectH2I_I(nu)/nu);
}

double photofuncH2II_I(double nu, float zred)
{
  return (4.0*PI*J_UV(nu, zred)*csectH2II_I(nu)/nu);
}

double photofuncH2I_II(double nu, float zred)
{
  return (4.0*PI*J_UV(nu, zred)*csectH2I_II(nu)/nu);
}

double photofuncH2II_II(double nu, float zred)
{
  return (4.0*PI*J_UV(nu, zred)*csectH2II_II(nu)/nu);
}

double GammaHI(float zred)
{
  int p;
  double Gamma, Gamma_old;
  
  for(p=1;p<=NITER;p++) {
    if (p > 1) Gamma_old = Gamma; 
    Gamma = midexp(photofuncHI, 1.0, 1.0, p, zred);
    if (p > 1) {
      if (fabs(Gamma - Gamma_old) < ALLOWED_ERR * fabs(Gamma_old)) 
        return (Gamma*1.e-22/hplanck);
    }
  }
  
  // it hasn't converged. who cares?
  return (Gamma*1.e-22/hplanck);
}

double GammaHeI(float zred)
{
  int p;
  double Gamma, nuT, Gamma_old;

  nuT=5.945e+15;
 
  for(p=1;p<=NITER;p++) {
    if (p > 1) Gamma_old = Gamma; 
    Gamma = midexp(photofuncHeI, nuT/nuL, 1.0, p, zred);
    if (p > 1) {
      if (fabs(Gamma - Gamma_old) < ALLOWED_ERR * fabs(Gamma_old)) 
        return (Gamma*1.e-22/hplanck);
    }
  }

  return (Gamma*1.e-22/hplanck);
}

double GammaHeII(float zred)
{
  int p;
  double Gamma, Gamma_old;
 
  for(p=1;p<=NITER;p++) {
    if (p > 1) Gamma_old = Gamma; 
    Gamma = midexp(photofuncHeII, 4.0, 1.0, p, zred);
    if (p > 1) {
      if (fabs(Gamma - Gamma_old) < ALLOWED_ERR * fabs(Gamma_old)) 
        return (Gamma*1.e-22/hplanck);
    }
  }
  return (Gamma*1.e-22/hplanck);
}

double GammaHM(float zred)
{
  int p;
  double Gamma, nuT, Gamma_old;
 
  nuT=1.826e14;

  for(p=1;p<=NITER;p++) {
    if (p > 1) Gamma_old = Gamma; 
    Gamma = midexp(photofuncHM, nuT/nuL, 1.0, p, zred);
    if (p > 1) {
      if (fabs(Gamma - Gamma_old) < ALLOWED_ERR * fabs(Gamma_old)) 
        return (Gamma*1.e-22/hplanck);
    }
  }
  return (Gamma*1.e-22/hplanck);
}

double GammaH2I_I(float zred)
{
  int p;
  //double Gamma, nuA, nuB, Gamma_old;
  double Gamma, nuA, Gamma_old;
 
  nuA = 1.1338235; // 15.42eV / 13.6eV
  //nuB = 1.3014705; // 17.7eV / 13.6eV

  for(p=1;p<=NITER;p++) {
    if (p > 1) Gamma_old = Gamma; 
    // Gamma = midpnt(photofuncH2I_I, nuA, nuB, p, zred);
    Gamma = midexp(photofuncH2I_I, nuA, 1.0, p, zred);
    if (p > 1) {
      if (fabs(Gamma - Gamma_old) < ALLOWED_ERR * fabs(Gamma_old)) 
        return (Gamma*1.e-22/hplanck);
    }
  }

  return (Gamma*1.e-22/hplanck);
}

double GammaH2I_II(float zred)
{
  int p;
  double Gamma, nuA, nuB, Gamma_old;
 
  nuA = 0.82867647; // 11.27eV / 13.6eV
  nuB = 1.0; // 13.6eV / 13.6eV

  for(p=1;p<=NITER;p++) {
    if (p > 1) Gamma_old = Gamma; 
    Gamma = midpnt(photofuncH2I_II, nuA, nuB, p, zred);
    if (p > 1) {
      if (fabs(Gamma - Gamma_old) < ALLOWED_ERR * fabs(Gamma_old)) 
        return (Gamma*1.e-22/hplanck);
    }
  }

  return (Gamma*1.e-22/hplanck);
}

double GammaH2II_I(float zred)
{
  int p;
  double Gamma, nuA, nuB, Gamma_old;
 
  nuA = 0.1948529411; // 2.65eV / 13.6eV
  nuB = 1.544117647; // 21.0eV / 13.6eV

  for(p=1;p<=NITER;p++) {
    if (p > 1) Gamma_old = Gamma; 
    Gamma = midpnt(photofuncH2II_I, nuA, nuB, p, zred);
    if (p > 1) {
      if (fabs(Gamma - Gamma_old) < ALLOWED_ERR * fabs(Gamma_old)) 
        return (Gamma*1.e-22/hplanck);
    }
  }

  return (Gamma*1.e-22/hplanck);
}

double GammaH2II_II(float zred)
{
  int p;
  double Gamma, nuA, nuB, Gamma_old;
 
  nuA = 2.2058823529; // 30.0eV / 13.6eV
  nuB = 5.1470588235; // 70.0eV / 13.6eV

  for(p=1;p<=NITER;p++) {
    if (p > 1) Gamma_old = Gamma; 
    Gamma = midpnt(photofuncH2II_II, nuA, nuB, p, zred);
    if (p > 1) {
      if (fabs(Gamma - Gamma_old) < ALLOWED_ERR * fabs(Gamma_old)) 
        return (Gamma*1.e-22/hplanck);
    }
  }

  return (Gamma*1.e-22/hplanck);
}

double heatfuncHI(double nu, float zred)
{
  return (4.0*PI*J_UV(nu, zred)*csectHI(nu)*(nu-1.0)/nu);
}

double heatfuncHeI(double nu, float zred)
{
  double nuT;
  //nuT = 1.8125; // 5.945/3.28
  nuT = HeI_LYMAN_LIMIT;
  return (4.0*PI*J_UV(nu, zred)*csectHeI(nu)*(nu-nuT)/nu);
}

double heatfuncHeII(double nu, float zred)
{
  double nuT;
  //nuT = 4.0; // 13.12/3.28
  nuT = HeII_LYMAN_LIMIT;
  return (4.0*PI*J_UV(nu, zred)*csectHeII(nu)*(nu-nuT)/nu);
}

#if 0
double heatfuncHM(double nu, float zred) 
{
  double nuT; 
  nuT=1.826e14/nuL; 
  return (4*PI*J_UV(nu, zred)*csectHM(nu)*(nu - nuT)/nu); 
}

double heatfuncH2I_I(double nu, float zred)
{
  double nuA = 1.1338235; // 15.42eV / 13.6eV
  return (4*PI*J_UV(nu, zred)*csectH2I_I(nu)*(nu - nuA)/nu); 
}

double heatfuncH2I_II(double nu, float zred)
{
  double nuT = 0.82867647; // 11.27/13.6
  return (4*PI*J_UV(nu, zred)*csectH2I_II(nu)*(nu - nuT)/nu); 
}

double heatfuncH2II_I(double nu, float zred)
{
  double nuT = 0.1948529411; // 2.65eV / 13.6eV
  return (4*PI*J_UV(nu, zred)*csectH2II_I(nu)*(nu - nuT)/nu); 
}

double heatfuncH2II_II(double nu, float zred)
{
  double nuT = 30./13.6; 
  return (4*PI*J_UV(nu, zred)*csectH2II_II(nu)*(nu - nuT)/nu); 
}
#endif

double HeatHI(float zred)
{
  int p;
  double Heat, Heat_old;

  for(p=1;p<=NITER;p++) {
    if (p > 1) Heat_old = Heat; 
    Heat = midexp(heatfuncHI, 1.0, 1.0, p, zred);
    if (p > 1) {
      if (fabs(Heat - Heat_old) < ALLOWED_ERR * fabs(Heat_old)) 
        return (Heat*nuL); 
    }
  }

  return (Heat*nuL*1.0e-22);
}

double HeatHeI(float zred)
{
  int p;
  double Heat, nuT, Heat_old;
  nuT = HeI_LYMAN_LIMIT;
  
  for(p=1;p<=NITER;p++) {
    if (p > 1) Heat_old = Heat; 
    Heat = midexp(heatfuncHeI, nuT, 1.0, p, zred); // nuT = 5.945e+15; nuT/nuL
    if (p > 1) {
      if (fabs(Heat - Heat_old) < ALLOWED_ERR * fabs(Heat_old)) 
        return (Heat*nuL); 
    }
  }

  return (Heat*nuL)*1.0e-22;
}

double HeatHeII(float zred)
{ 
  int p;
  double Heat, nuT, Heat_old;
  nuT = HeII_LYMAN_LIMIT;
  
  for(p=1;p<=NITER;p++) {
    if (p > 1) Heat_old = Heat; 
    Heat = midexp(heatfuncHeII, nuT, 1.0, p, zred);
    if (p > 1) {
      if (fabs(Heat - Heat_old) < ALLOWED_ERR * fabs(Heat_old)) 
        return (Heat*nuL);
    }
  }

  return (Heat*nuL*1.0e-22);
}

#if 0
double HeatHM(float zred)
{
  int p; 
  double Heat, nuT, Heat_old; 

  nuT=1.826e14;

  for (p = 1; p <= NITER; p++) {
    if (p > 1)  Heat_old = Heat; 
    Heat = midexp(heatfuncHM, nuT/nuL, 1.0, p, zred); 
    if (p > 1) {
      if (fabs(Heat - Heat_old) < ALLOWED_ERR * fabs(Heat_old)) 
        return (Heat*nuL);
    }
  }
  return (Heat*nuL*1.0e-22);
}

double HeatH2I_I(float zred)
{
  int p; 
  //double Heat, nuA, nuB, Heat_old; 
  double Heat, nuA, Heat_old; 
  nuA = 1.1338235; // 15.42eV / 13.6eV
  //nuB = 1.3014705; // 17.7eV / 13.6eV
  for (p = 1; p <= NITER; p++) {
    if (p > 1) Heat_old = Heat; 
    //Heat = midpnt(heatfuncH2I_I, nuA, nuB, p, zred); 
    Heat = midexp(heatfuncH2I_I, nuA, 0, p, zred); 
    if (p > 1) {
      if (fabs(Heat - Heat_old) < ALLOWED_ERR * fabs(Heat_old)) 
        return (Heat*nuL);
    }
  }
  return (Heat*nuL*1.0e-22);
}

double HeatH2I_II(float zred)
{
  int p;
  double Heat, nuA, nuB, Heat_old;
 
  nuA = 0.82867647; // 11.27eV / 13.6eV
  nuB = 1.0; // 13.6eV / 13.6eV

  for(p=1;p<=NITER;p++) {
    if (p > 1) Heat_old = Heat; 
    Heat = midpnt(heatfuncH2I_II, nuA, nuB, p, zred);
    if (p > 1) {
      if (fabs(Heat - Heat_old) < ALLOWED_ERR * fabs(Heat_old)) 
        return (Heat*nuL);
    }
  }
  return (Heat*nuL*1.0e-22);
}

double HeatH2II_I(float zred) 
{
  int p; 
  double Heat, nuA, nuB, Heat_old; 

  nuA = 0.1948529411; // 2.65eV / 13.6eV
  nuB = 1.544117647; // 21.0eV / 13.6eV

  for (p = 1; p <= NITER; p++) {
    if (p > 1) Heat_old = Heat; 
    Heat = midpnt(heatfuncH2II_I, nuA, nuB, p, zred);
    if (p > 1) {
      if (fabs(Heat - Heat_old) < ALLOWED_ERR * fabs(Heat_old)) 
        return (Heat*nuL);
    }
  }
  return (Heat*nuL*1.0e-22);
}

double HeatH2II_II(float zred) 
{
  int p; 
  double Heat, nuA, nuB, Heat_old; 
  nuA = 2.2058823529; // 30.0eV / 13.6eV
  nuB = 5.1470588235; // 70.0eV / 13.6eV
  for (p = 1; p <= NITER; p++) {
    if (p > 1) Heat_old = Heat; 
    Heat = midpnt(heatfuncH2II_II, nuA, nuB, p, zred);
    if (p > 1) {
      if (fabs(Heat - Heat_old) < ALLOWED_ERR * fabs(Heat_old)) 
        return (Heat*nuL);
    }
  }
  return (Heat*nuL*1.0e-22);
}
#endif

void hunt(float xx[], int n, float x, int *jlo)
{
  int jm,jhi,inc;
  int ascnd;

  ascnd=(xx[n-1] > xx[0]);
  if (*jlo <= -1 || *jlo > n-1) {
    *jlo=-1;
    jhi=n;
  } else {
    inc=1;
    if (x >= xx[*jlo] == ascnd) {
      if (*jlo == n - 1) return;
      jhi=(*jlo)+1;
      while (x >= xx[jhi] == ascnd) { 
        *jlo=jhi; 
        inc += inc; 
        jhi=(*jlo)+inc; 
        if (jhi > n-1) { 
          jhi=n; 
          break; 
        }
      }
    } else {
      if (*jlo == 0) { 
        *jlo=-1; 
        return; 
      }
      jhi=(*jlo)--;
      while (x < xx[*jlo] == ascnd) { 
        jhi=(*jlo); 
        inc <<= 1; 
        if (inc >= jhi) { 
          *jlo=-1; 
          break; 
        } else *jlo=jhi-inc;
      }
    }
  }
  while (jhi-(*jlo) != 1) {
    jm=(jhi+(*jlo)) >> 1;
    if (x > xx[jm] == ascnd)
      *jlo=jm;
    else
      jhi=jm;
  }
}
