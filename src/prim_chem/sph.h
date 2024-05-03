#ifndef __SPH__
#define __SPH__

#if !defined(__HKVISC__) && !defined(__MBVISC__) && !defined(__G2VISC__)
#define __HKVISC__            /* default artificial viscousity */
#endif 

#if defined(__MBVISC__) || defined(__G2VISC__)
#define __ROTV__
#endif

/*--- adiabatic index ---*/
#define GAMMA  (1.666666666666)
#define GAMM1  (0.666666666666)
#define SQGGM1 (1.054092553389)

/*--- coefficients for artificial viscousity ---*/
#ifdef __MBVISC__
#define SPH_ALPHA (1.0)
#define SPH_BETA  (2.0)
#endif

#ifdef __HKVISC__
#define SPH_ALPHA (0.5)
#define SPH_BETA  (0.5)
#endif

#ifdef __G2VISC__
#define SPH_ALPHA (1.0)
#endif

#define NSM (20.0)
#define FHVMIN (0.2)
#define FHVMAX (5.0)
#define FVISCMIN (0.05)

#define SPH_CFL (0.4)

struct SPH_Particle {
  int   pindx; // index of the particle structure

  float dens;
  float uene;

  float divv;
#ifndef __HKVISC__
  float rotvx;
  float rotvy;
  float rotvz;
  float fvisc;
#endif /* __HKVISC__ */
  float hsm;
  float duene;
  float nnb;
  float fhv;

#ifdef __ENTROPY__
  float etrp;
  float detrp;
#endif

#ifdef __MBVISC__
  float mumax;
#endif

#ifdef __G2VISC__
  float vsigmax;
#endif

#ifdef __HEATCOOL__
  float durad;
  float wmol;
#endif /* __HEATCOOL__ */

#ifdef __TWOTEMP__
  float te_scaled;
#endif /* __TWOTEMP__ */
  float duvisc;

};

#define SPH_NINTERP (16382)
#define SPH_TBL_SIZE (16384)

struct SPH_Kernel {
  float (*W)[2];
  float (*dW)[2];
};
#endif /*__SPH__*/
