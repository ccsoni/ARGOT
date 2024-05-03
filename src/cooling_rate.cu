//=======================================================================
//  Functions for cooling rates
//=======================================================================
//  Recombination cooling rate [erg/cm**3/sec]
//     creh2(T)*ne*nHII     : H II
//     crehe2(T)*ne*nHeII   : He II
//     crexihe2(T)*ne*nHeII : He II (dielectronic recombination)
//     crehe3(T)*ne*nHeIII  : He III
//
//  Ionization cooling rate [erg/cm**3/sec]
//     cioh1(T)*ne*nHI    : H I
//     ciohe1(T)*ne*nHeI  : He I
//     ciohe2(T)*ne*nHeII : He II
//
//  Collisional excitation cooling rate [erg/cm**3/sec]
//     cexh1(T)*ne*nHI      : H I
//     cexhe1(T)*ne*ne*nHeI : He I     
//     cexhe2(T)*ne*nHeII   : He II
//
//  Bremsstrahlung cooling rate [erg/cm**3/sec]
//     brems(T)*ne*(nHII+nHeII+nHeIII)
//
//  Compton cooling rate [erg/cm**3/sec]
//     compt(T,z)*ne
//  H_2 cooling rate [erg/cm**3/sec] (Lepp & Shull 1983)
//     h2mol(T,nHI,nH2)
//=======================================================================

#include <math.h>

__device__ double creh2_dev(double t)
{
  // Cen 
  // return (8.7e-27*sqrt(t)/pow((t/1.e3),0.2));
#ifndef __CASE_B__
  return (6.5e-27*sqrt(t)/pow((t/1.e3),0.2)/(1.0+pow((t/1.e6),0.7)));
#else
  double lambda = 2.0*157807.0/t;
  return (3.435e-30*t*pow(lambda, 1.97)/pow(1.0+pow(lambda/2.25, 0.376),3.72));
#endif
}

__device__ double crehe2_dev(double t)
{

#ifndef __CASE_B__
  return (1.55e-26*pow(t,0.3647));
#else /* __CASE_B__ */
  if(t<980.0) {
    return (1.7e-26*pow(t,0.35));
  }else{ 
    return (1.9e-25*pow(t/1.0e3, 0.15));
  }
#endif
}

__device__ double crexihe2_dev(double t)
{
  return (1.24e-13/pow(t,1.5)*exp(-4.7/(t*1.e-5))*(1.0+0.3*exp(-0.94/(t*1.e-5))));
}

__device__ double crehe3_dev(double t)
{
#ifndef __CASE_B__
  return (3.48e-26*sqrt(t)/pow((t/1.e3),0.2)/(1.e0+pow((t/4.e6),0.7)));

  // Cen
  // return (3.48e-26*sqrt(t)/pow((t/1.e3),0.2)/(1.d0+pow((t/1.e6),0.7)));
#else
  double lambda =  2.0*(631515.0/t);
  return (3.435e-30*t*pow(lambda, 1.97)/pow(1.0+pow(lambda/2.25, 0.376),3.72));
#endif
}

__device__ double cioh1_dev(double t)
{
  return (1.27e-21*sqrt(t)/(1.0+sqrt(t*1.e-5))*exp(-1.578091/(t*1.e-5)));
}

__device__ double ciohe1_dev(double t)
{
  return (9.38e-22*sqrt(t)/(1.0+sqrt(t*1.e-5))*exp(-2.853354/(t*1.e-5)));
}

__device__ double ciohe2_dev(double t)
{
  return (4.95e-22*sqrt(t)/(1.0+sqrt(t*1.e-5))*exp(-6.31515/(t*1.e-5)));
}

__device__ double cexh1_dev(double t)
{
  return (7.5e-19/(1.0+sqrt(t*1.e-5))*exp(-1.18348/(t*1.e-5)));
}

__device__ double cexhe1_dev(double t)
{
  return (9.1e-27*pow(t,-0.1687)/(1.0+sqrt(t*1.e-5))*exp(-0.13179/(t*1.e-5)));
}

__device__ double cexhe2_dev(double t)
{
  return (5.54e-17*pow(t,-0.397)/(1.0+sqrt(t*1.e-5))*exp(-4.73638/(t*1.e-5)));
}

__device__ double brems_dev(double t)
{
  double gaunt;

  gaunt = 1.1+0.34*exp(-(5.5-log10(t))*(5.5-log10(t))/3.0);
  return (1.42e-27*gaunt*sqrt(t));
}

__device__ double compt_dev(double t,float z)
{
  double z14,z1;
  double tgamma;

  z1  = 1.0+z;
  z14 = z1*z1*z1*z1;

  tgamma = 2.73*z1;

  /*
  if(t>tgamma) {
    return (5.406e-36*z14*(t-tgamma));
  }else{
    return (0.0);
  }
  */
  return (5.406e-36*z14*(t-tgamma));
}

__device__ double h2mol_dev(double t, double HI, double H2I)
{
  double crh, crl, cvh, cvl, kHI, kH2, xx;
  
  xx = log10(t)-4.e0;
  if(t > 1635.0) {
    kHI = 1.0e-12*sqrt(t)*exp(1000.0/t);
  }else{
    kHI = 1.4e-13*exp(t/125.e0-(t/577.e0)*(t/577.e0));
  }
  kHI *= (exp(-8.152e-13/(1.38e-16*t)));

  kH2 = 8.152e-13*(4.2/(1.38e-6*(t+1190.0))+1.0/(1.38e-16*t));
  kH2 = 1.45e-12*sqrt(t)*exp(-kH2);

  if(t > 1087.0) {
    crh = 3.9e-19*exp(-6118.0/t);
  }else{
    crh = pow(10.0,(-19.24+(0.474-1.247*xx)*xx));
  }

  if(t > 4031.0) {
    crl = 1.38e-22*exp(-9243.0/t);
  }else{
    crl = pow(10.0,(-22.9-(0.553+1.148*xx)*xx));
  }
  crl *= (pow(H2I,0.77)+1.2*pow(HI,0.77));

  cvh = 1.1e-18*exp(-6744.0/t);
  cvl = 8.18e-13*(HI*kHI+H2I*kH2);

  if (cvl ==0.0 || crl ==0.0) {
    return 0.0;
  }else{
    return ((cvh/(1.0+cvh/cvl) + crh/(1.0+crh/crl))*H2I);
  }
}
