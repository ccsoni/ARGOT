#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <unistd.h>

#include "run_param.h"
#include "mpi_param.h"
#include "radiation.h"

#include "prototype.h"

int main(int argc, char **argv) 
{
  static struct run_param this_run;
  static struct mpi_param this_mpi;

  static struct ray_segment send_seg[10], recv_seg[10];

  MPI_Init(&argc, &argv);
  init_mpi(&this_run,&this_mpi);

  init_run(&this_run);

  sleep(1);

  if(this_run.mpi_rank == 0) {
    printf("Done.\n");
    fflush(stdout);
  }
  //  MPI_Finalize();
  //  exit(0);

  int ii;
  for(ii=0;ii<10;ii++) {
    send_seg[ii].ray_indx = ii;
    send_seg[ii].local_rank = 0;
    send_seg[ii].target_rank = ii;
    send_seg[ii].xpos_start = 0.0*ii;
    send_seg[ii].ypos_start = 0.1*ii;
    send_seg[ii].zpos_start = 0.2*ii;
    send_seg[ii].xpos_end = 0.3*ii;
    send_seg[ii].ypos_end = 0.4*ii;
    send_seg[ii].zpos_end = 0.5*ii;
    send_seg[ii].optical_depth = 0.6*ii;
  }

#if 0
  MPI_Status status;

  if(this_run.mpi_rank==0) {
    MPI_Send(send_seg, 10, this_mpi.segment_type, 1, 1, MPI_COMM_WORLD);
  }else if(this_run.mpi_rank==1) {
    MPI_Recv(recv_seg, 10, this_mpi.segment_type, 0, 1, MPI_COMM_WORLD, &status);
  }

  if(this_run.mpi_rank==1) {
    int iproc;
    for(iproc=0;iproc<10;iproc++) {
      fprintf(this_run.proc_file,"%llu %d\n",
	      recv_seg[iproc].ray_indx, recv_seg[iproc].local_rank);
      fprintf(this_run.proc_file,"START: %14.6e %14.6e %14.6e\n",
	      recv_seg[iproc].xpos_start, 
	      recv_seg[iproc].ypos_start, 
	      recv_seg[iproc].zpos_start);
      fprintf(this_run.proc_file,"END: %14.6e %14.6e %14.6e\n",
	      recv_seg[iproc].xpos_end, 
	      recv_seg[iproc].ypos_end, 
	      recv_seg[iproc].zpos_end);
      fprintf(this_run.proc_file,"DEPTH: %14.6e\n",
	      recv_seg[iproc].optical_depth);
    }
  }
#else 

  MPI_Info info;
  MPI_Win  win;

  MPI_Info_create(&info);
  MPI_Info_set(info, "no_locks", "true");

  MPI_Win_create(recv_seg, 10, sizeof(struct ray_segment), info,
		 MPI_COMM_WORLD, &win);

  MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
  if(this_run.mpi_rank == 0) {
    MPI_Put(send_seg, 10, this_mpi.segment_type, 1, 
	    0, 10, this_mpi.segment_type, win);
  }
  MPI_Win_fence(0, win);
  MPI_Win_free(&win);
  MPI_Info_free(&info);

  if(this_run.mpi_rank==1) {
    int iproc;
    for(iproc=0;iproc<10;iproc++) {
      fprintf(this_run.proc_file,"%d %d\n",
	      recv_seg[iproc].ray_indx, recv_seg[iproc].local_rank);
      fprintf(this_run.proc_file,"START: %14.6e %14.6e %14.6e\n",
	      recv_seg[iproc].xpos_start, 
	      recv_seg[iproc].ypos_start, 
	      recv_seg[iproc].zpos_start);
      fprintf(this_run.proc_file,"END: %14.6e %14.6e %14.6e\n",
	      recv_seg[iproc].xpos_end, 
	      recv_seg[iproc].ypos_end, 
	      recv_seg[iproc].zpos_end);
      fprintf(this_run.proc_file,"DEPTH: %14.6e\n",
	      recv_seg[iproc].optical_depth);
      fflush(this_run.proc_file);
    }
  }

#endif

  MPI_Finalize();
  exit(0);

  
}
