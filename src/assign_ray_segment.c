#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>
#include <mpi.h>

#include "run_param.h"
#include "constants.h"
#include "mpi_param.h"
#include "omp_param.h"
#include "radiation.h"
#include "prototype.h"

#define __MPI_BYTE_COMM__

#ifdef __USE_GPU__
#define __TEMP_SEG_MEM__
#endif

void count_ray_segment(struct light_ray *ray, struct run_param *this_run, struct omp_param *this_omp)
{
  MPI_Info info;
  MPI_Win win_nseg;

  uint64_t iray;
  int irank;

  // # of segments that should be exported to other domains
  static uint64_t nseg_to[NNODE]; 
  // # of segments imported from other domains
  static uint64_t nseg_from[NNODE];

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

  for(irank=0;irank<NNODE;irank++) {
    nseg_to[irank]=0;
    nseg_from[irank]=0;
  }

  /* compute the number of ray segments for each domain */
  for(iray=0;iray<this_run->nray;iray++) {
    uint64_t iseg;
    for(iseg=0;iseg<ray[iray].num_segment;iseg++) {
      nseg_to[ray[iray].segment[iseg].local_rank]++;
    }
  }

  /* collect the number of ray segments from each domain */
  MPI_Info_create(&info);
  MPI_Info_set(info, "no_locks", "true");

  omp_set_nest_lock(&(this_omp->omp_lock));
  
#ifdef __MPI_BYTE_COMM__
  MPI_Win_create(nseg_from, NNODE*sizeof(uint64_t), sizeof(char), info, 
		 MPI_COMM_WORLD, &win_nseg);
#else
  MPI_Win_create(nseg_from, NNODE*sizeof(uint64_t), sizeof(uint64_t), info, 
		 MPI_COMM_WORLD, &win_nseg);
#endif

  omp_unset_nest_lock(&(this_omp->omp_lock));
  
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_nseg);
  for(irank=0;irank<NNODE;irank++) {
#ifdef __MPI_BYTE_COMM__
    MPI_Put(nseg_to+irank, sizeof(uint64_t), MPI_BYTE, irank, 
	    this_run->mpi_rank*sizeof(uint64_t), sizeof(uint64_t), MPI_BYTE, win_nseg);
#else
    MPI_Put(nseg_to+irank, 1, MPI_UINT64_T, irank, 
	    this_run->mpi_rank, 1, MPI_UINT64_T, win_nseg);
#endif
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_nseg);

  MPI_Info_free(&info);
  MPI_Win_free(&win_nseg);

#ifdef __ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__
  for(irank=0;irank<NNODE;irank++) {
    fprintf(this_run->proc_file,"nseg_to[%d] = %llu\n", 
	    irank, nseg_to[irank]);
  }
  fflush(this_run->proc_file);
  for(irank=0;irank<NNODE;irank++) {
    fprintf(this_run->proc_file,"nseg_from[%d] = %llu\n", 
	    irank, nseg_from[irank]);
  }
  fflush(this_run->proc_file);
#endif /* __ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__ */
  
  this_run->nseg = 0;
  for(irank=0;irank<NNODE;irank++) this_run->nseg += nseg_from[irank];

}

void assign_ray_segment(struct light_ray *ray, struct ray_segment *seg_orig, 
			struct run_param *this_run, struct mpi_param *this_mpi, struct omp_param *this_omp) 
{
  MPI_Info info;
  MPI_Win win_nseg, win_seg;

  // # of segments that should be exported to other domains
  static uint64_t nseg_to[NNODE]; 
  // # of segments imported from other domains
  static uint64_t nseg_from[NNODE];

  // address of the data sent from other domains
  static uint64_t adr[NNODE];

  // buffer of ray segments for MPI communications
  struct ray_segment *seg_buf;

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

  struct ray_segment *seg;
  
#ifdef __TEMP_SEG_MEM__
  seg = (struct ray_segment*)malloc(sizeof(struct ray_segment)*this_run->nseg);

  /* copy pinned mem to host malloc mem */
#pragma omp parallel for schedule(auto)
  for(uint64_t iseg=0; iseg<this_run->nseg; iseg++) {
    seg[iseg] = seg_orig[iseg];
  }

#else
  seg = seg_orig;
#endif
  
#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

  for(int irank=0;irank<NNODE;irank++) {
    nseg_to[irank]=0;
    nseg_from[irank]=0;
  }

  /* compute the number of ray segments for each domain */
  for(uint64_t iray=0;iray<this_run->nray;iray++) {
    for(uint64_t iseg=0;iseg<ray[iray].num_segment;iseg++) {
      nseg_to[ray[iray].segment[iseg].local_rank]++;
    }
  }

  /* collect the number of ray segments from each domain */
  MPI_Info_create(&info);
  MPI_Info_set(info, "no_locks", "true");

  omp_set_nest_lock(&(this_omp->omp_lock));
  
#ifdef __MPI_BYTE_COMM__
  MPI_Win_create(nseg_from, NNODE*sizeof(uint64_t), sizeof(char), info, 
		 MPI_COMM_WORLD, &win_nseg);
#else
  MPI_Win_create(nseg_from, NNODE*sizeof(uint64_t), sizeof(uint64_t), info, 
		 MPI_COMM_WORLD, &win_nseg);
#endif

  omp_unset_nest_lock(&(this_omp->omp_lock));
  
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win_nseg);
  for(int irank=0;irank<NNODE;irank++) {

#ifdef __MPI_BYTE_COMM__
    MPI_Put(nseg_to+irank, sizeof(uint64_t), MPI_BYTE, irank, 
	    this_run->mpi_rank*sizeof(uint64_t), sizeof(uint64_t), MPI_BYTE, win_nseg);
#else
    MPI_Put(nseg_to+irank, 1, MPI_UINT64_T, irank, 
	    this_run->mpi_rank, 1, MPI_UINT64_T, win_nseg);
#endif
  }
  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_nseg);
  MPI_Info_free(&info);
  MPI_Win_free(&win_nseg);

#ifdef __ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__
  for(int irank=0;irank<NNODE;irank++) {
    fprintf(this_run->proc_file,"nseg_to[%d] = %llu\n", 
	    irank, nseg_to[irank]);
  }
  fflush(this_run->proc_file);
  for(int irank=0;irank<NNODE;irank++) {
    fprintf(this_run->proc_file,"nseg_from[%d] = %llu\n", 
	    irank, nseg_from[irank]);
  }
  fflush(this_run->proc_file);
#endif /* __ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__ */
  
  this_run->nseg = 0;
  for(int irank=0;irank<NNODE;irank++) this_run->nseg += nseg_from[irank];

  //  assert(this_run->nseg <= NSEG_MAX);

  fprintf(this_run->proc_file, "# of the ray segment: %llu \n",this_run->nseg);
  fflush(this_run->proc_file);

  /* collect the ray segments for this domain */
  uint64_t num_seg;
  num_seg = 0;
  for(uint64_t iray=0;iray<this_run->nray;iray++) {
    for(uint64_t iseg=0;iseg<ray[iray].num_segment;iseg++) {
      if(ray[iray].segment[iseg].local_rank == this_run->mpi_rank) {
	seg[num_seg] = ray[iray].segment[iseg];
	num_seg++;
      }
    }
  }
  
#ifdef __ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__
  fprintf(this_run->proc_file, 
	  "# of the ray segment inside this domain : %d \n",num_seg);
  fflush(this_run->proc_file);
#endif /* __ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__ */ 

  MPI_Info_create(&info);
  MPI_Info_set(info, "no_locks", "true");

  /* start sending segments data to the irank-th domain */
  for(int irank=0;irank<NNODE;irank++) {
    //    if(this_run->mpi_rank == 0) fprintf(this_run->proc_file,"%d\n",irank);
    uint64_t iadr;

    if(this_run->mpi_rank != irank) {
      seg_buf = (struct ray_segment *) 
	malloc(sizeof(struct ray_segment)*nseg_to[irank]);
      assert(seg_buf);
      
      // first gather the ray segments to irank-th domain
      uint64_t nseg_to_irank;
      nseg_to_irank=0;
      for(uint64_t iray=0;iray<this_run->nray;iray++) {
	for(uint64_t iseg=0;iseg<ray[iray].num_segment;iseg++) {
	  if(ray[iray].segment[iseg].local_rank == irank) {
	    seg_buf[nseg_to_irank] = ray[iray].segment[iseg];
	    nseg_to_irank++;
	  }
	}
      }
    }
    
    /* communicate the address of the data sent from other domain */

    omp_set_nest_lock(&(this_omp->omp_lock));
    
#ifdef __MPI_BYTE_COMM__
    if(this_run->mpi_rank == irank) {
      MPI_Win_create(seg, this_run->nseg*sizeof(struct ray_segment), sizeof(char), 
		     info, MPI_COMM_WORLD, &win_seg);
    }else{
      MPI_Win_create(&iadr, 1*sizeof(uint64_t), sizeof(char), info, MPI_COMM_WORLD, &win_seg);
    }
#else
    if(this_run->mpi_rank == irank) {
      MPI_Win_create(seg, this_run->nseg*sizeof(struct ray_segment), sizeof(struct ray_segment), 
		     info, MPI_COMM_WORLD, &win_seg);
    }else{
      MPI_Win_create(&iadr, 1*sizeof(uint64_t), sizeof(uint64_t), info, MPI_COMM_WORLD, &win_seg);
    }
#endif

    omp_unset_nest_lock(&(this_omp->omp_lock));
    
    MPI_Win_fence(MPI_MODE_NOPRECEDE, win_seg);

    if(this_run->mpi_rank == irank) {
      /* compute the address in the seg[] for the data sent by the other process */
      uint64_t ii;
      adr[0] = 0;
      for(ii=0;ii<NNODE-1;ii++){
	adr[ii+1] = adr[ii] + nseg_from[ii];
      }
      adr[irank]=0;
      for(ii=0;ii<irank;ii++) adr[ii] += nseg_from[irank];

      for(ii=0;ii<NNODE;ii++) {
#ifdef __MPI_BYTE_COMM__
	if(ii != irank) MPI_Put(adr+ii, sizeof(uint64_t), MPI_BYTE, ii, 0, sizeof(uint64_t), 
				MPI_BYTE, win_seg);
#else
	if(ii != irank) MPI_Put(adr+ii,1, MPI_UINT64_T, ii, 0, 1, 
				MPI_UINT64_T, win_seg);
#endif
      }

    }

    MPI_Win_fence(MPI_MODE_NOSTORE, win_seg);


#if __ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__
    if(this_run->mpi_rank != irank) {
      fprintf(this_run->proc_file,"iadr = %llu\n", iadr);
      fflush(this_run->proc_file);
    }
#endif /* __ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__ */
    /* send the ray segment to the irank-th domain */
    if(this_run->mpi_rank != irank && nseg_to[irank] != 0) {
#ifdef __MPI_BYTE_COMM__
      MPI_Put(seg_buf,nseg_to[irank]*sizeof(struct ray_segment),MPI_BYTE, irank,
	      iadr*sizeof(struct ray_segment), nseg_to[irank]*sizeof(struct ray_segment),MPI_BYTE,win_seg);
#else
      MPI_Put(seg_buf,nseg_to[irank],this_mpi->segment_type, irank,
	      iadr, nseg_to[irank],this_mpi->segment_type,win_seg);
#endif
    }

    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win_seg);

    MPI_Win_free(&win_seg);

    if(this_run->mpi_rank != irank)     free(seg_buf);
  }

  MPI_Info_free(&info);

  
#ifdef __TEMP_SEG_MEM__
  /* copy host malloc mem to pinned mem */
#pragma omp parallel for schedule(auto)
  for(uint64_t iseg=0; iseg<this_run->nseg; iseg++) {
    seg_orig[iseg] = seg[iseg];
  }

  free(seg);
#endif

  
#ifdef __ARGOT_PROFILE__
    times(&end_tms);
    gettimeofday(&end_tv, NULL);

    fprintf(this_run->proc_file,
	    "# assign_ray_segment : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
	    timing(start_tms, end_tms), wallclock_timing(start_tv, end_tv));
    fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */

#ifdef __ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__
  if(this_run->mpi_rank == 4) {

    for(uint64_t iseg=num_seg;iseg<this_run->nseg;iseg++) {
      fprintf(this_run->proc_file,"%lu %lu %lu\n",iseg,this_run->nseg,this_run->nray);
      fprintf(this_run->proc_file, "%d %d %d %d\n", this_run->mpi_rank,
	      seg[iseg].ray_indx, seg[iseg].local_rank, seg[iseg].target_rank, seg[iseg].src_rank);
      fprintf(this_run->proc_file, "AAA %14.6e %14.6e %14.6e : %14.6e %14.6e %14.6e", 
	      seg[iseg].xpos_start,seg[iseg].ypos_start,seg[iseg].zpos_start,
	      seg[iseg].xpos_end,seg[iseg].ypos_end,seg[iseg].zpos_end);
      fprintf(this_run->proc_file, "%14.6e %14.6e %14.6e\n", 
	      seg[iseg].xpos_end-seg[iseg].xpos_start,
	      seg[iseg].ypos_end-seg[iseg].ypos_start,
	      seg[iseg].zpos_end-seg[iseg].zpos_start);
    }
  }
#endif /* __ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__ */

}
