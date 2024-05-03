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


float fluid_kinetic_energy(struct fluid_mesh *mesh, struct run_param *this_run)
{
  double KE, KE_total;

  KE = 0.0;

  for(int imesh=0;imesh<NMESH_LOCAL;imesh++) {
    KE += 0.5*NORML2(mesh[imesh].momx,mesh[imesh].momy,mesh[imesh].momz)/mesh[imesh].dens;
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

  for(int imesh=0;imesh<NMESH_LOCAL;imesh++) {
    TE += mesh[imesh].uene*mesh[imesh].dens;
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

  for(int imesh=0;imesh<NMESH_LOCAL;imesh++) {
    PE += 0.5*mesh[imesh].pot*mesh[imesh].dens;
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

  for(int imesh=0;imesh<NMESH_LOCAL;imesh++) {
    mass += mesh[imesh].dens;
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
#ifdef __COSMOLOGICAL__
      fprintf(this_run->diag_file,
	      "#step   time        a(t)        dt          mass           KE             TE             PE             TOTAL          split_time  elapsed_wall_time\n");
#else /* !__COSMOLOGICAL__ */
      fprintf(this_run->diag_file,
	      "#step   time        dt          mass           KE             TE             PE             TOTAL          split_time  elapsed_wall_time\n");
#endif
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
#ifdef __COSMOLOGICAL__
      fprintf(this_run->diag_file,
	      "%5d %11.3e %11.3e %11.3e %14.6e %14.6e %14.6e %14.6e %14.6e %11.3e %11.3e\n",
	      this_run->step, this_run->tnow, this_run->anow, dtime, mass_fluid, KE_fluid, TE_fluid, PE_fluid, KE_fluid+TE_fluid+PE_fluid,
	      walltime, elapsed_wall_time);
#else /* !__COSMOLOGICAL__ */
      fprintf(this_run->diag_file,
	      "%5d %11.3e %11.3e %14.6e %14.6e %14.6e %14.6e %14.6e %11.3e %11.3e\n",
	      this_run->step, this_run->tnow, dtime, mass_fluid, KE_fluid, TE_fluid, PE_fluid, KE_fluid+TE_fluid+PE_fluid,
	      walltime, elapsed_wall_time);
#endif /* __COSMOLOGICAL__ */
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
