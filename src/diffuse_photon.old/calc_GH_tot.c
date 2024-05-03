#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "diffuse_photon.h"

#ifndef TINY
#define TINY (1.0e-31)
#endif

void calc_GH_tot(struct radiation_mesh *rmesh, struct step_func_factor *step_fact)
{
  int ix;
  //#pragma omp parallel for schedule(dynamic,32)
#pragma omp parallel for schedule(auto)
  for(ix=0; ix<NMESH_LOCAL; ix++){
  
    float length_tot,I_tot;
    float I_seg_in_bar;

    length_tot = rmesh[ix].length;
    I_tot    = rmesh[ix].IHI;
    I_seg_in_bar = I_tot/(length_tot*rmesh[ix].absorptionHI+TINY); 
    rmesh[ix].GHI_tot += I_seg_in_bar*step_fact->HI[0][0] + rmesh[ix].source_funcHI*step_fact->HI[0][1];
    rmesh[ix].HHI_tot += I_seg_in_bar*step_fact->HI[1][0] + rmesh[ix].source_funcHI*step_fact->HI[1][1];
  
#ifdef __HELIUM__
    I_tot    = rmesh[ix].IHeI;
    I_seg_in_bar = I_tot/(length_tot*rmesh[ix].absorptionHeI+TINY); 
    rmesh[ix].GHeI_tot += I_seg_in_bar*step_fact->HeI[0][0] + rmesh[ix].source_funcHeI*step_fact->HeI[0][1];
    rmesh[ix].HHeI_tot += I_seg_in_bar*step_fact->HeI[1][0] + rmesh[ix].source_funcHeI*step_fact->HeI[1][1];

    I_tot    = rmesh[ix].IHeII;
    I_seg_in_bar = I_tot/(length_tot*rmesh[ix].absorptionHeII+TINY); 
    rmesh[ix].GHeII_tot += I_seg_in_bar*step_fact->HeII[0][0] + rmesh[ix].source_funcHeII*step_fact->HeII[0][1];
    rmesh[ix].HHeII_tot += I_seg_in_bar*step_fact->HeII[1][0] + rmesh[ix].source_funcHeII*step_fact->HeII[1][1];
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
