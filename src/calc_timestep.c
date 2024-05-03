#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <mpi.h>

#include "constants.h"
#include "run_param.h"
#include "fluid.h"
#include "cuda_mem_space.h"

#include "prototype.h"

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

double calc_dtime_gpu(struct cuda_mem_space*, struct cuda_param*, 
		      struct run_param*);

#ifdef __USE_GPU__
double calc_timestep_chem(struct cuda_mem_space *cuda_mem, 
			  struct cuda_param *this_cuda, 
			  struct run_param *this_run)
{
  double dt, dt_min;

  dt = calc_dtime_gpu(cuda_mem, this_cuda, this_run);

  MPI_Allreduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  return (dt_min/this_run->tunit);
}

#else
double calc_timestep_chem(struct fluid_mesh *mesh, struct run_param *this_run)
{
  int ix, iy, iz;

  double dt, dt_min;

  dt = DBL_MAX;

  for(ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(iz=0;iz<NMESH_Z_LOCAL;iz++) {

	double dti;
	dti = calc_dtime(&MESH(ix,iy,iz), &(MESH(ix,iy,iz).prev_chem), this_run);
	dt = MIN(dt, dti);
	
      }
    }
  }

  MPI_Allreduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  return (dt_min/this_run->tunit);
}
#endif


double calc_timestep_fluid(struct fluid_mesh *mesh, struct run_param *this_run) 
{
  double dtime_min;

  dtime_min = DBL_MAX;
  
#pragma omp parallel for schedule(auto) reduction(min:dtime_min)
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++){
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++){
	struct fluid_mesh *target_mesh;
	float cs, gamma, velx, vely, velz;
	float dtime_x, dtime_y, dtime_z, dtime;

	target_mesh = &MESH(ix,iy,iz);

	gamma = gamma_total(target_mesh, this_run);
	cs = sqrt((gamma-1.0)*gamma*target_mesh->uene);
	velx = fabsf(target_mesh->momx/target_mesh->dens);
	vely = fabsf(target_mesh->momy/target_mesh->dens);
	velz = fabsf(target_mesh->momz/target_mesh->dens);

	dtime_x = this_run->delta_x/(velx+cs);
	dtime_y = this_run->delta_y/(vely+cs);
	dtime_z = this_run->delta_z/(velz+cs);
	
	dtime = fminf(dtime_x, fminf(dtime_y, dtime_z));
	dtime_min = fminf(dtime, dtime_min);
      }
    }
  }

#ifndef __SERIAL__
  double dtime_min_local;
  dtime_min_local = dtime_min;

  MPI_Allreduce(&dtime_min_local, &dtime_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

  return (COURANT_FACT*dtime_min);

}


#ifdef __USE_GPU__
double calc_timestep(struct fluid_mesh *mesh, struct cuda_mem_space *cuda_mem, 
		     struct cuda_param *this_cuda, struct run_param *this_run)
{
  double dt, dt_hydro, dt_chem;

  dt_hydro = calc_timestep_fluid(mesh, this_run);
  dt_chem  = calc_timestep_chem(cuda_mem, this_cuda, this_run);  

  dt = MIN(dt_hydro, DTFACT_RAD*dt_chem);

  fprintf(this_run->proc_file,"# dt = %e , dt_hydro = %e , dt_chem*DTFACT_RAD = %e\n", 
          dt, dt_hydro, DTFACT_RAD*dt_chem);
  
  return dt;

}
#else
double calc_timestep(struct fluid_mesh *mesh, struct run_param *this_run)
{

  double dt, dt_hydro, dt_chem;

  dt_hydro = calc_timestep_fluid(mesh, this_run);
  dt_chem  = calc_timestep_chem(mesh, this_run);

  dt = MIN(dt_hydro, DTFACT_RAD*dt_chem);
  
  return dt;

}
#endif
