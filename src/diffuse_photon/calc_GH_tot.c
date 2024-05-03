#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "diffuse_photon.h"

#ifndef TINY
#define TINY (1.0e-31)
#endif

static inline double calc_I_seg_GammaHI(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;
    
  I_seg_in_bar = rmesh->I_nu1/rmesh->absorption_nu1;

#ifdef __HELIUM__
  I_seg_in_bar += rmesh->I_nu2*RINT_J_HI_nu2/rmesh->absorption_nu2;
  I_seg_in_bar += rmesh->I_nu3*RINT_J_HI_nu3/rmesh->absorption_nu3;
#ifdef __HELIUM_BB__
  I_seg_in_bar += rmesh->I_nu4*RINT_J_HI_nu4/rmesh->absorption_nu4;
  I_seg_in_bar += rmesh->I_nu5*RINT_J_HI_nu5/rmesh->absorption_nu5;
  I_seg_in_bar += rmesh->I_nu6*RINT_J_HI_nu6/rmesh->absorption_nu6;
#endif //__HELIUM_BB__
#endif //__HELIUM__

  return I_seg_in_bar;
}

static inline double calc_I_seg_HeatHI(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;
    
  I_seg_in_bar = rmesh->I_nu1/rmesh->absorption_nu1;

#ifdef __HELIUM__
  I_seg_in_bar += rmesh->I_nu2*RINT_H_HI_nu2/rmesh->absorption_nu2;
  I_seg_in_bar += rmesh->I_nu3*RINT_H_HI_nu3/rmesh->absorption_nu3;
#ifdef __HELIUM_BB__
  I_seg_in_bar += rmesh->I_nu4*RINT_H_HI_nu4/rmesh->absorption_nu4;
  I_seg_in_bar += rmesh->I_nu5*RINT_H_HI_nu5/rmesh->absorption_nu5;
  I_seg_in_bar += rmesh->I_nu6*RINT_H_HI_nu6/rmesh->absorption_nu6;
#endif //__HELIUM_BB__
#endif //__HELIUM__

  return I_seg_in_bar;
}

static inline double calc_source_GammaHI(struct radiation_mesh *rmesh)
{
  double source_func;

  source_func = rmesh->source_func_nu1;

#ifdef __HELIUM__
  source_func += rmesh->source_func_nu2*RINT_J_HI_nu2;
  source_func += rmesh->source_func_nu3*RINT_J_HI_nu3;
#ifdef __HELIUM_BB__
  source_func += rmesh->source_func_nu4*RINT_J_HI_nu4;
  source_func += rmesh->source_func_nu5*RINT_J_HI_nu5;
  source_func += rmesh->source_func_nu6*RINT_J_HI_nu6;
#endif //__HELIUM_BB__
#endif //__HELIUM__

  return source_func;
}

static inline double calc_source_HeatHI(struct radiation_mesh *rmesh)
{
  double source_func;

  source_func = rmesh->source_func_nu1;

#ifdef __HELIUM__
  source_func += rmesh->source_func_nu2*RINT_H_HI_nu2;
  source_func += rmesh->source_func_nu3*RINT_H_HI_nu3;
#ifdef __HELIUM_BB__
  source_func += rmesh->source_func_nu4*RINT_H_HI_nu4;
  source_func += rmesh->source_func_nu5*RINT_H_HI_nu5;
  source_func += rmesh->source_func_nu6*RINT_H_HI_nu6;
#endif //__HELIUM_BB__
#endif //__HELIUM__

  return source_func;
}

#ifdef __HELIUM__
static inline double calc_I_seg_GammaHeI(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;
    
  I_seg_in_bar  = rmesh->I_nu2/rmesh->absorption_nu2;
  I_seg_in_bar += rmesh->I_nu3*RINT_J_HeI_nu3/rmesh->absorption_nu3;
#ifdef __HELIUM_BB__
  I_seg_in_bar += rmesh->I_nu6*RINT_J_HeI_nu6/rmesh->absorption_nu6;
#endif //__HELIUM_BB__

  return I_seg_in_bar;
}

static inline double calc_I_seg_HeatHeI(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;
    
  I_seg_in_bar  = rmesh->I_nu2/rmesh->absorption_nu2;
  I_seg_in_bar += rmesh->I_nu3*RINT_H_HeI_nu3/rmesh->absorption_nu3;
#ifdef __HELIUM_BB__
  I_seg_in_bar += rmesh->I_nu6*RINT_H_HeI_nu6/rmesh->absorption_nu6;
#endif //__HELIUM_BB__

  return I_seg_in_bar;
}

static inline double calc_source_GammaHeI(struct radiation_mesh *rmesh)
{
  double source_func;

  source_func  = rmesh->source_func_nu2;
  source_func += rmesh->source_func_nu3*RINT_J_HeI_nu3;
#ifdef __HELIUM_BB__
  source_func += rmesh->source_func_nu6*RINT_J_HeI_nu6;
#endif //__HELIUM_BB__

  return source_func;
}

static inline double calc_source_HeatHeI(struct radiation_mesh *rmesh)
{
  double source_func;

  source_func  = rmesh->source_func_nu2;
  source_func += rmesh->source_func_nu3*RINT_H_HeI_nu3;
#ifdef __HELIUM_BB__
  source_func += rmesh->source_func_nu6*RINT_H_HeI_nu6;
#endif //__HELIUM_BB__

  return source_func;
}


static inline double calc_I_seg_GammaHeII(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;   
  I_seg_in_bar = rmesh->I_nu3/rmesh->absorption_nu3;
  return I_seg_in_bar;
}

static inline double calc_I_seg_HeatHeII(struct radiation_mesh *rmesh)
{
  double I_seg_in_bar;
  I_seg_in_bar = rmesh->I_nu3/rmesh->absorption_nu3;
  return I_seg_in_bar;
}

static inline double calc_source_GammaHeII(struct radiation_mesh *rmesh)
{
  double source_func;
  source_func = rmesh->source_func_nu3;
  return source_func;
}

static inline double calc_source_HeatHeII(struct radiation_mesh *rmesh)
{
  double source_func;
  source_func = rmesh->source_func_nu3;
  return source_func;
}
#endif //__HELIUM__


void calc_GH_tot(struct radiation_mesh *rmesh, struct step_func_factor *step_fact)
{
  int ix;
  //#pragma omp parallel for schedule(dynamic,32)
#pragma omp parallel for schedule(auto)
  for(ix=0; ix<NMESH_LOCAL; ix++){
  
    double length_tot;
    double Iseg_Gamma, Iseg_Heat;
    double source_Gamma, source_Heat;
    
    length_tot = rmesh[ix].length;

    /* HI */
    Iseg_Gamma  = calc_I_seg_GammaHI(&rmesh[ix]);
    Iseg_Heat  = calc_I_seg_HeatHI(&rmesh[ix]);

    Iseg_Gamma /= length_tot;
    Iseg_Heat  /= length_tot;
    
    source_Gamma = calc_source_GammaHI(&rmesh[ix]);
    source_Heat  = calc_source_HeatHI(&rmesh[ix]);
    
    rmesh[ix].GHI_tot += (Iseg_Gamma + source_Gamma)*step_fact->HI[0];
    rmesh[ix].HHI_tot += (Iseg_Heat  + source_Heat) *step_fact->HI[1];

#ifdef __HELIUM__
    /* HeI */
    Iseg_Gamma = calc_I_seg_GammaHeI(&rmesh[ix]);
    Iseg_Heat  = calc_I_seg_HeatHeI(&rmesh[ix]);

    Iseg_Gamma /= length_tot;
    Iseg_Heat  /= length_tot;
    
    source_Gamma = calc_source_GammaHeI(&rmesh[ix]);
    source_Heat  = calc_source_HeatHeI(&rmesh[ix]);
    
    rmesh[ix].GHeI_tot += (Iseg_Gamma + source_Gamma)*step_fact->HeI[0];
    rmesh[ix].HHeI_tot += (Iseg_Heat  + source_Heat) *step_fact->HeI[1];

    /* HeII */
    Iseg_Gamma = calc_I_seg_GammaHeII(&rmesh[ix]);
    Iseg_Heat  = calc_I_seg_HeatHeII(&rmesh[ix]);

    Iseg_Gamma /= length_tot;
    Iseg_Heat /= length_tot;
    
    source_Gamma = calc_source_GammaHeII(&rmesh[ix]);
    source_Heat  = calc_source_HeatHeII(&rmesh[ix]);
    
    rmesh[ix].GHeII_tot += (Iseg_Gamma + source_Gamma)*step_fact->HeII[0];
    rmesh[ix].HHeII_tot += (Iseg_Heat  + source_Heat) *step_fact->HeII[1];
#endif //__HELIUM__

  }
}


void calc_GH_sum(struct fluid_mesh *mesh, struct radiation_mesh *rmesh)
{
  int ix;
  float r_n_ang = 1.0e0/N_ANG;
  
  //#pragma omp parallel for schedule(dynamic,32)
#pragma omp parallel for schedule(auto)
  for(ix=0; ix<NMESH_LOCAL; ix++){
    mesh[ix].prev_chem.GammaHI  += rmesh[ix].GHI_tot * r_n_ang; 
    mesh[ix].prev_chem.HeatHI   += rmesh[ix].HHI_tot * r_n_ang; 
    
#ifdef __HELIUM__
    mesh[ix].prev_chem.GammaHeI  += rmesh[ix].GHeI_tot * r_n_ang;
    mesh[ix].prev_chem.HeatHeI   += rmesh[ix].HHeI_tot * r_n_ang;   
    mesh[ix].prev_chem.GammaHeII += rmesh[ix].GHeII_tot * r_n_ang;
    mesh[ix].prev_chem.HeatHeII  += rmesh[ix].HHeII_tot * r_n_ang;
#endif //__HELIUM__
  }
}
