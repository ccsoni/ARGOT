#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>
#include <omp.h>

#include "constants.h"
#include "run_param.h"
#include "mpi_param.h"
#include "radiation.h"

#include "prototype.h"

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])
#define GAMMA_ION(ix,iy,iz) (gamma[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

void step_chemistry_gpu(struct cuda_mem_space*, struct cuda_param*,
			struct run_param*, float*, float*, float);

float diff_chem(struct prim_chem *chem1, struct prim_chem *chem2)
{
  float diff_H, diff_elec;

#if 1
  diff_H = fabs(chem1->GammaHI-chem2->GammaHI)/(chem1->GammaHI+1.0e-30);
  diff_elec = fabs(chem1->felec - chem2->felec)/(chem1->felec+1.0e-30);
#else 
  if(chem1->fHI < 0.5) {
    diff_H = fabs(chem1->fHI - chem2->fHI)/(chem1->fHI+1.0e-30);
  }else{
    diff_H = fabs(chem1->fHII - chem2->fHII)/(chem1->fHII+1.0e-30);
  }
#endif

  return MAX(diff_H,diff_elec);
}

float diff_uene(float uene1, float uene2)
{
  float diff;

  diff = fabs(uene1-uene2)/(MAX(uene1, uene2)+1.0e-20);

  return diff;
}


#ifdef __USE_GPU__
void step_chemistry(struct cuda_mem_space *cuda_mem, struct cuda_param *this_cuda,
		    struct run_param *this_run, float dtime, float *max_diff_chem, float *max_diff_uene)
{
  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);
#endif

  float max_diff_local_chem, max_diff_local_uene;

  step_chemistry_gpu(cuda_mem, this_cuda, this_run, &max_diff_local_chem, &max_diff_local_uene, dtime);

  //  fprintf(this_run->proc_file,"# max_diff = %14.6e\n",max_diff_local);

  MPI_Allreduce(&max_diff_local_chem, max_diff_chem, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

  MPI_Allreduce(&max_diff_local_uene, max_diff_uene, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

#ifdef __ARGOT_PROFILE__
    times(&end_tms);
    gettimeofday(&end_tv, NULL);

    fprintf(this_run->proc_file,
	    "# step_chemistry     : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	    timing(start_tms, end_tms), wallclock_timing(start_tv, end_tv));
    fprintf(this_run->proc_file,
	    "# max difference (chem) : %14.6e\n", *max_diff_chem);
    fprintf(this_run->proc_file,
	    "# max difference (uene) : %14.6e\n", *max_diff_uene);
    fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */

}
#else /* ! __USE_GPU__ */
void step_chemistry(struct fluid_mesh *mesh, struct run_param *this_run, 
		    float dtime, float *max_diff_chem, float *max_diff_uene)
{
  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);
#endif

  //  int ix, iy, iz;
  float max_diff_uene_local, max_diff_chem_local;
  float max_diff_uene_global, max_diff_chem_global;

  max_diff_chem_local = 0.0;
  max_diff_uene_local = 0.0;

  int nrec_max, niter_max;

  nrec_max = 0;
  niter_max = 0;
#pragma omp parallel for schedule(auto) reduction(max:nrec_max), reduction(max:niter_max), reduction(max:max_diff_chem_local), reduction(max:max_diff_uene_local)
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	struct fluid_mesh *target_mesh;
	struct prim_chem prev_chem;
	float prev_uene;
	int nrec, niter;

	target_mesh = &(MESH(ix,iy,iz));

	prev_chem = target_mesh->prev_chem;
	prev_uene = target_mesh->prev_uene;

	target_mesh->duene = 0.0;nrec=0;
#if 0 /* solve chem. reactions and radiative heating/cooling separately */
	advance_reaction(target_mesh, &prev_chem, this_run, dtime);
#ifdef __HEATCOOL__
	advance_heatcool(target_mesh, &prev_uene, &prev_chem, this_run, dtime, &nrec, &niter);
#endif
#else /* solve chem. reactions and radiative heating/cooling simultaneously */
#ifdef __HEATCOOL__
	advance_reaction_and_heatcool(target_mesh, &prev_uene, &prev_chem, this_run, dtime, 
				      &nrec, &niter);
#else /* !__HEATCOOL__ */
	advance_reaction(target_mesh, &prev_chem, this_run, dtime);
#endif /* __HEATCOOL__ */
#endif

	max_diff_chem_local = MAX(max_diff_chem_local,
				  diff_chem(&prev_chem, &(target_mesh->chem)));
	max_diff_uene_local = MAX(max_diff_uene_local,
				  diff_uene(prev_uene, target_mesh->uene));

	nrec_max = MAX(nrec_max, nrec);
	niter_max = MAX(niter_max, niter);

	target_mesh->chem = prev_chem;
	target_mesh->uene = prev_uene;


      }
    }
  }
#ifdef __ARGOT_PROFILE__
  fprintf(this_run->proc_file,
	  "# nrec_max = %d / niter_max = %d\n", nrec_max, niter_max);
  fprintf(this_run->proc_file,
	  "# local max difference (chem)    : %14.6e\n", max_diff_chem_local);
  fprintf(this_run->proc_file,
	  "# local max difference (uene)    : %14.6e\n", max_diff_uene_local);
  fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__*/

  MPI_Allreduce(&max_diff_chem_local, &max_diff_chem_global, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&max_diff_uene_local, &max_diff_uene_global, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);  
  *max_diff_chem = max_diff_chem_global;
  *max_diff_uene = max_diff_uene_global;

#ifdef __ARGOT_PROFILE__
    times(&end_tms);
    gettimeofday(&end_tv, NULL);
    fprintf(this_run->proc_file,
	    "# max difference (chem)    : %14.6e\n", *max_diff_chem);
    fprintf(this_run->proc_file,
	    "# max difference (uene)    : %14.6e\n", *max_diff_uene);
    fprintf(this_run->proc_file,
	    "# step_chemistry     : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	    timing(start_tms, end_tms), wallclock_timing(start_tv, end_tv));
    fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */

}
#endif

