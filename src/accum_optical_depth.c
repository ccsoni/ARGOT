#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <sys/time.h>
#include <sys/times.h>

#include "run_param.h"
#include "constants.h"
#include "radiation.h"
#include "prototype.h"

void accum_optical_depth(struct light_ray *ray, struct ray_segment *seg, 
			 struct run_param *this_run, struct omp_param *this_omp)
{
  int irank;
  float (*optical_depth_send)[NCHANNEL];
  float (*optical_depth_accum)[NCHANNEL];

  MPI_Win win_nray, win_depth;
  MPI_Info info;

  static uint64_t nray_in[NNODE]; /* number of light rays in each domain */

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

#ifdef __ARGOT_PROFILE__
    times(&start_tms);
    gettimeofday(&start_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

#if 1
  MPI_Allgather(&(this_run->nray), 1, MPI_UINT64_T, 
		nray_in, 1, MPI_UINT64_T, MPI_COMM_WORLD);
#else
  MPI_Gather(&(this_run->nray), 1, MPI_UINT64_T,
	     nray_in, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(nray_in, NNODE, MPI_UINT64_T, 0, MPI_COMM_WORLD);
#endif

#if 0
  int iprc;
  for(iprc=0;iprc<NNODE;iprc++) {
    fprintf(this_run->proc_file, "nray_in[%d] = %llu\n", iprc, nray_in[iprc]);
    fflush(this_run->proc_file);
  }
#endif

  MPI_Info_create(&info);
  MPI_Info_set(info, "no_locks", "true");

  /* start accumulating the optical depth of rays */
  /* whose targets are in the irank-th domain */
  for(irank=0;irank<NNODE;irank++) {
    uint64_t iray;

    optical_depth_send = 
      (float (*)[NCHANNEL]) malloc(sizeof(float)*nray_in[irank]*NCHANNEL);
    for(iray=0;iray<nray_in[irank];iray++) {
      for(int ichan=0;ichan<NCHANNEL;ichan++) optical_depth_send[iray][ichan]=0.0;
    }

    omp_set_nest_lock(&(this_omp->omp_lock));
    
    if(this_run->mpi_rank == irank) { /* sending the optical depths to myself */
      optical_depth_accum = 
	(float (*)[NCHANNEL]) malloc(sizeof(float)*nray_in[irank]*NCHANNEL);
      for(iray=0;iray<nray_in[irank];iray++) {
	for(int ichan=0;ichan<NCHANNEL;ichan++) optical_depth_accum[iray][ichan]=0.0;
      }

      MPI_Win_create(optical_depth_accum, nray_in[irank]*NCHANNEL*sizeof(float), sizeof(float), info,
		     MPI_COMM_WORLD, &win_depth);
    }else{
      MPI_Win_create(NULL, 0, sizeof(int), info,
		     MPI_COMM_WORLD, &win_depth);
    }

    omp_unset_nest_lock(&(this_omp->omp_lock));
    
    MPI_Win_fence(MPI_MODE_NOPRECEDE, win_depth);

    uint64_t iseg;
    // select the ray segment whose optical depths 
    // should be sent to the irank-th domain
//#pragma omp parallel for schedule(dynamic,256)
#pragma omp parallel for schedule(auto)
    for(iseg=0;iseg<this_run->nseg;iseg++) {
      if(seg[iseg].target_rank == irank) {
	int ichan=0;
	optical_depth_send[seg[iseg].ray_indx][ichan++] = seg[iseg].optical_depth_HI;
#ifdef __HELIUM__
	optical_depth_send[seg[iseg].ray_indx][ichan++] = seg[iseg].optical_depth_HeI;
	optical_depth_send[seg[iseg].ray_indx][ichan++] = seg[iseg].optical_depth_HeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
	optical_depth_send[seg[iseg].ray_indx][ichan++] = seg[iseg].optical_depth_HM;
	optical_depth_send[seg[iseg].ray_indx][ichan++] = seg[iseg].optical_depth_H2I;
	optical_depth_send[seg[iseg].ray_indx][ichan++] = seg[iseg].optical_depth_H2II;
#endif /* __HYDROGEN_MOL__ */
      }
    }

    if(this_run->mpi_rank != irank) {      
      MPI_Accumulate(optical_depth_send, nray_in[irank]*NCHANNEL, MPI_FLOAT, 
		    irank, 0, nray_in[irank]*NCHANNEL, MPI_FLOAT, MPI_SUM, win_depth);
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win_depth);
    MPI_Win_free(&win_depth);

    if(this_run->mpi_rank == irank) {
#pragma omp parallel for schedule(auto)
      for(iray=0;iray<this_run->nray;iray++) {
	int ichan=0;
	ray[iray].optical_depth_HI = 
	  optical_depth_accum[iray][ichan] + optical_depth_send[iray][ichan];
#ifdef __HELIUM__
	ichan+=1;
	ray[iray].optical_depth_HeI = 
	  optical_depth_accum[iray][ichan] + optical_depth_send[iray][ichan];
	ichan+=1;
	ray[iray].optical_depth_HeII = 
	  optical_depth_accum[iray][ichan] + optical_depth_send[iray][ichan];
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
	ichan+=1;
	ray[iray].optical_depth_HM = 
	  optical_depth_accum[iray][ichan] + optical_depth_send[iray][ichan];
	ichan+=1;
	ray[iray].optical_depth_H2I = 
	  optical_depth_accum[iray][ichan] + optical_depth_send[iray][ichan];
	ichan+=1;
	ray[iray].optical_depth_H2II = 
	  optical_depth_accum[iray][ichan] + optical_depth_send[iray][ichan];
#endif /* __HYDROGEN_MOL__ */
      }
      free(optical_depth_accum);
    }

    free(optical_depth_send);
  }

  MPI_Info_free(&info);

#ifdef __ARGOT_PROFILE__
    times(&end_tms);
    gettimeofday(&end_tv, NULL);

    fprintf(this_run->proc_file,
	    "# accum_optical_depth : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	    timing(start_tms, end_tms), wallclock_timing(start_tv, end_tv));
    fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */

#if 0
#ifdef __HELIUM__
    if(this_run->mpi_rank == 0) {
      for(int iray=0;iray<1000;iray++) {
	printf("%d %e %e %e\n",iray,ray[iray].optical_depth_HI,ray[iray].optical_depth_HeI,ray[iray].optical_depth_HeII);
      }
    }
#endif /* __HELIUM__ */
#endif
}

