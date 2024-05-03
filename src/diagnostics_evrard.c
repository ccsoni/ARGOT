#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>

#include "run_param.h"
#include "fluid.h"
#include "mpi.h"

float wallclock_timing(struct timeval, struct timeval);

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

#define R_LIMIT (1.0)
//#define R_LIMIT (6.5)
//#define R_LIMIT (5.0)
#define __EVRARD_CENTER__

float fluid_kinetic_energy(struct fluid_mesh *mesh, struct run_param *this_run)
{
  double KE, KE_total;

  KE = 0.0;

  float cx_pos,cy_pos,cz_pos;
#ifndef __EVRARD_CENTER__
  cx_pos = 0.5e0*(this_run->xmax+this_run->xmin) - 0.5*this_run->delta_x;
  cy_pos = 0.5e0*(this_run->ymax+this_run->ymin) - 0.5*this_run->delta_y;
  cz_pos = 0.5e0*(this_run->zmax+this_run->zmin) - 0.5*this_run->delta_z;
#else
  cx_pos = 0.5e0*(this_run->xmax+this_run->xmin);
  cy_pos = 0.5e0*(this_run->ymax+this_run->ymin);
  cz_pos = 0.5e0*(this_run->zmax+this_run->zmin);
#endif

  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	
	struct fluid_mesh *tgt;
	tgt = &MESH(ix,iy,iz);

	float ix_pos = this_run->xmin_local + (ix+0.5)*this_run->delta_x;
	float iy_pos = this_run->ymin_local + (iy+0.5)*this_run->delta_y;
	float iz_pos = this_run->zmin_local + (iz+0.5)*this_run->delta_z;
	
	float radius = sqrt(SQR(ix_pos-cx_pos)+SQR(iy_pos-cy_pos)+SQR(iz_pos-cz_pos));
	
	if(radius <= R_LIMIT) {
	  KE += 0.5*NORML2(tgt->momx,tgt->momy,tgt->momz)/tgt->dens;
	}

      }
    }
  }	

  KE_total = 0.0;
  MPI_Allreduce(&KE, &KE_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  KE_total *= (this_run->delta_x*this_run->delta_y*this_run->delta_z);

  return ((float)KE_total);
}

float fluid_thermal_energy(struct fluid_mesh *mesh, struct run_param *this_run)
{
  double TE, TE_total;
  
  TE = 0.0;

  float cx_pos,cy_pos,cz_pos;
#ifndef __EVRARD_CENTER__
  cx_pos = 0.5e0*(this_run->xmax+this_run->xmin) - 0.5*this_run->delta_x;
  cy_pos = 0.5e0*(this_run->ymax+this_run->ymin) - 0.5*this_run->delta_y;
  cz_pos = 0.5e0*(this_run->zmax+this_run->zmin) - 0.5*this_run->delta_z;
#else
  cx_pos = 0.5e0*(this_run->xmax+this_run->xmin);
  cy_pos = 0.5e0*(this_run->ymax+this_run->ymin);
  cz_pos = 0.5e0*(this_run->zmax+this_run->zmin);
#endif
  
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	
	struct fluid_mesh *tgt;
	tgt = &MESH(ix,iy,iz);
	
	float ix_pos = this_run->xmin_local + (ix+0.5)*this_run->delta_x;
	float iy_pos = this_run->ymin_local + (iy+0.5)*this_run->delta_y;
	float iz_pos = this_run->zmin_local + (iz+0.5)*this_run->delta_z;
	
	float radius = sqrt(SQR(ix_pos-cx_pos)+SQR(iy_pos-cy_pos)+SQR(iz_pos-cz_pos));
	
	if(radius <= R_LIMIT) {
	  TE += tgt->uene*tgt->dens;
	}
	
      }
    }
  }	
  

  TE_total = 0.0;
  MPI_Allreduce(&TE, &TE_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  TE_total *= (this_run->delta_x*this_run->delta_y*this_run->delta_z);

  return ((float)TE_total);
}


float fluid_potential_energy(struct fluid_mesh *mesh, struct run_param *this_run)
{
  double PE, PE_total;
 
  PE = 0.0;

  float cx_pos,cy_pos,cz_pos;
#ifndef __EVRARD_CENTER__
  cx_pos = 0.5e0*(this_run->xmax+this_run->xmin) - 0.5*this_run->delta_x;
  cy_pos = 0.5e0*(this_run->ymax+this_run->ymin) - 0.5*this_run->delta_y;
  cz_pos = 0.5e0*(this_run->zmax+this_run->zmin) - 0.5*this_run->delta_z;
#else
  cx_pos = 0.5e0*(this_run->xmax+this_run->xmin);
  cy_pos = 0.5e0*(this_run->ymax+this_run->ymin);
  cz_pos = 0.5e0*(this_run->zmax+this_run->zmin);
#endif

  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	
	struct fluid_mesh *tgt;
	tgt = &MESH(ix,iy,iz);
	
	float ix_pos = this_run->xmin_local + (ix+0.5)*this_run->delta_x;
	float iy_pos = this_run->ymin_local + (iy+0.5)*this_run->delta_y;
	float iz_pos = this_run->zmin_local + (iz+0.5)*this_run->delta_z;
	
	float radius = sqrt(SQR(ix_pos-cx_pos)+SQR(iy_pos-cy_pos)+SQR(iz_pos-cz_pos));
	
	if(radius <= R_LIMIT) {
	  PE += 0.5*tgt->pot*tgt->dens;
	}
	
      }
    }
  }	


  PE_total = 0.0;
  MPI_Allreduce(&PE, &PE_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  PE_total *= (this_run->delta_x*this_run->delta_y*this_run->delta_z);

  return ((float)PE_total);
}


float fluid_mass(struct fluid_mesh *mesh, struct run_param *this_run) 
{
  double mass, mass_total;

  mass = 0.0;

  float cx_pos,cy_pos,cz_pos;
#ifndef __EVRARD_CENTER__
  cx_pos = 0.5e0*(this_run->xmax+this_run->xmin) - 0.5*this_run->delta_x;
  cy_pos = 0.5e0*(this_run->ymax+this_run->ymin) - 0.5*this_run->delta_y;
  cz_pos = 0.5e0*(this_run->zmax+this_run->zmin) - 0.5*this_run->delta_z;
#else
  cx_pos = 0.5e0*(this_run->xmax+this_run->xmin);
  cy_pos = 0.5e0*(this_run->ymax+this_run->ymin);
  cz_pos = 0.5e0*(this_run->zmax+this_run->zmin);
#endif
  
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	
	struct fluid_mesh *tgt;
	tgt = &MESH(ix,iy,iz);
	
	float ix_pos = this_run->xmin_local + (ix+0.5)*this_run->delta_x;
	float iy_pos = this_run->ymin_local + (iy+0.5)*this_run->delta_y;
	float iz_pos = this_run->zmin_local + (iz+0.5)*this_run->delta_z;
	
	float radius = sqrt(SQR(ix_pos-cx_pos)+SQR(iy_pos-cy_pos)+SQR(iz_pos-cz_pos));
	
	if(radius <= R_LIMIT) {
	  mass += tgt->dens;
	}
	
      }
    }
  }	
  
  mass_total = 0.0;
  MPI_Allreduce(&mass, &mass_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  mass_total *= (this_run->delta_x*this_run->delta_y*this_run->delta_z);

  return ((float)mass_total);
}


void output_diagnostics(struct fluid_mesh *mesh, struct run_param *this_run, 
			float dtime)
{
  static struct timeval prev_tv, now_tv;
  static float elapsed_wall_time=0.0;

  float walltime;

  gettimeofday(&now_tv,NULL);

  if(this_run->step == 0) {
    if(this_run->mpi_rank == 0) {
      fprintf(this_run->diag_file,
	      "#step   time        dt          mass           KinE           TheE           PotE           TotE           split_time  elapsed_wall_time\n");
      fflush(this_run->diag_file);
    }
  }else{
    walltime = wallclock_timing(prev_tv, now_tv);
    elapsed_wall_time += walltime;
    float fdum;
    float mass_fluid, KE_fluid, TE_fluid, PE_fluid;
    fdum = 0.0;
    KE_fluid = fluid_kinetic_energy(mesh, this_run);
    TE_fluid = fluid_thermal_energy(mesh, this_run);
    PE_fluid = fluid_potential_energy(mesh, this_run);
    mass_fluid = fluid_mass(mesh, this_run);
    if(this_run->mpi_rank == 0) {
      fprintf(this_run->diag_file,
	      "%5d %11.3e %11.3e %14.6e %14.6e %14.6e %14.6e %14.6e %11.3e %11.3e\n",
	      this_run->step, this_run->tnow, dtime, mass_fluid, KE_fluid, TE_fluid, PE_fluid, KE_fluid+TE_fluid+PE_fluid, 
	      walltime, elapsed_wall_time);
      fflush(this_run->diag_file);

      if(isnan(mass_fluid) || isnan(KE_fluid) || isnan(TE_fluid)   || isnan(PE_fluid) ||
	 isinf(mass_fluid) || isinf(KE_fluid) || isinf(TE_fluid)   || isinf(PE_fluid)) {
	fprintf(stderr, "Inf or NaN appeared at tnow=%e\n",this_run->tnow);
	fprintf(stderr, "mass=%e, KE=%e, TE=%e, PE=%e\n",mass_fluid,KE_fluid,TE_fluid,PE_fluid);
	fflush(stderr);
        exit(EXIT_FAILURE);
      }
      
    }
  }

  prev_tv = now_tv;
}
