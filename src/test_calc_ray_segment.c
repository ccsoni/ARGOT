#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "constants.h"
#include "run_param.h"
#include "mpi_param.h"
#include "radiation.h"

#include "prototype.h"

int main(int argc, char **argv) 
{
  static struct run_param this_run;
  static struct mpi_param this_mpi;

  struct radiation_src src1;
  static struct light_ray this_ray;

  static struct fluid_mesh mesh[NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL];
  static struct radiation_src src[NSOURCE_MAX];

  MPI_Init(&argc, &argv);

  init_mpi(&this_run,&this_mpi);

  input_data(mesh, src, &this_run, argv[1]);

  init_run(&this_run);

  src1.xpos = 0.499;
  src1.ypos = 0.51;
  src1.zpos = 0.51;
  src1.L = 1.0e10;

  //  this_ray.src = src[0];
  this_ray.src = src1;
  this_ray.ix_target = 60;
  this_ray.iy_target = 40;
  this_ray.iz_target = 60;

  float xpos_target, ypos_target, zpos_target;
  xpos_target = this_run.xmin + this_run.delta_x*((float)this_ray.ix_target+0.5);
  ypos_target = this_run.ymin + this_run.delta_y*((float)this_ray.iy_target+0.5);
  zpos_target = this_run.zmin + this_run.delta_z*((float)this_ray.iz_target+0.5);

  fprintf(this_run.proc_file,"# target position : (%14.6e, %14.6e, %14.6e) \n",
	  xpos_target, ypos_target, zpos_target);
  fflush(this_run.proc_file);

  this_run.nray = 1;
  calc_ray_segment(&this_ray,&this_run);

  if(this_run.mpi_rank == 0) {
    int i;
    //    printf("%d\n",this_ray.num_segment);
    for(i=0;i<this_ray.num_segment;i++) {
#if 1
      printf("%d %14.6e %14.6e %14.6e\n",i,
	     this_ray.segment[i].xpos_start,
	     this_ray.segment[i].ypos_start,
	     this_ray.segment[i].zpos_start);
#endif
      printf("%d %14.6e %14.6e %14.6e %14.6e\n",this_ray.segment[i].local_rank,
	     this_ray.segment[i].xpos_end,
	     this_ray.segment[i].ypos_end,
	     this_ray.segment[i].zpos_end,
	     this_ray.segment[i].optical_depth);
      printf("\n");
    }
  }


  if(this_run.mpi_rank == this_ray.segment[1].local_rank) {
    calc_optical_depth(&(this_ray.segment[1]), mesh, &this_run);
    fprintf(this_run.proc_file,"%14.6e \n",this_ray.segment[2].optical_depth);
  }

  MPI_Finalize();
  exit(EXIT_SUCCESS);
  
}
