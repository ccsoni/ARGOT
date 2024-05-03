#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#include "run_param.h"
#include "fluid.h"
#include "source.h"

#define FILENAME_LENGTH (256)

void make_directory(char*);

void output_mesh_single(struct fluid_mesh *mesh, struct run_param *this_run,
			char *filename)
{

  FILE *output_fp;

  output_fp = fopen(filename,"w");

  fwrite(this_run, sizeof(struct run_param), 1, output_fp);
#if 0
  fwrite(mesh, sizeof(struct fluid_mesh), 
	 NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL,
	 output_fp);
#else
  int imesh;
  for(imesh=0;imesh<NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL;imesh++) {
    struct fluid_mesh_io tmp_mesh;
    tmp_mesh.dens = mesh[imesh].dens;
    tmp_mesh.eneg = mesh[imesh].eneg;
    tmp_mesh.momx = mesh[imesh].momx;
    tmp_mesh.momy = mesh[imesh].momy;
    tmp_mesh.momz = mesh[imesh].momz;
    tmp_mesh.uene = mesh[imesh].uene;
    tmp_mesh.pot  = mesh[imesh].pot;
    tmp_mesh.chem = mesh[imesh].chem;
    fwrite(&tmp_mesh, sizeof(struct fluid_mesh_io), 1, output_fp);
  }
#endif
  fclose(output_fp);

}

void output_mesh(struct fluid_mesh *mesh, struct run_param *this_run,
		 char *prefix)
{

  static char filename[FILENAME_LENGTH];

  sprintf(filename,"%s_%03d_%03d_%03d",prefix, 
	  this_run->rank_x, this_run->rank_y, this_run->rank_z);

  output_mesh_single(mesh, this_run, filename);
}


void output_src(struct radiation_src *src, struct run_param *this_run,
		char *prefix)
{
  FILE *output_fp;
  static char filename[FILENAME_LENGTH];

  sprintf(filename,"%s_src.dat",prefix);

  output_fp = fopen(filename, "w");
  fwrite(&(this_run->nsrc), sizeof(uint64_t), 1, output_fp);
  fwrite(&this_run->freq, sizeof(struct freq_param), 1, output_fp);
  fwrite(src, sizeof(struct radiation_src), this_run->nsrc, output_fp);
  fclose(output_fp);

}

void output_data(struct fluid_mesh *mesh, 
		 struct radiation_src *src,
		 struct run_param *this_run, 
		 char *prefix)
{
  output_mesh(mesh, this_run, prefix);
  output_src(src, this_run, prefix);
}

void output_data_in_run(struct fluid_mesh *mesh,
			struct radiation_src *src,
			struct run_param *this_run,
			char *prefix)
{
  static char prefix_stamp[256];
  static char dirname[128];

  if(this_run->output_indx >= this_run->noutput) {
    return;
  }

  if(this_run->step % 20 == 0) {
    make_directory("dmp");
    sprintf(prefix_stamp, "dmp/%s-dmp",prefix);
    output_data(mesh, src, this_run, prefix_stamp);
  }

  if(this_run->tnow > this_run->output_timing[this_run->output_indx]){
    sprintf(dirname,"%s-%02d",prefix, this_run->output_indx);
    make_directory(dirname);

    sprintf(prefix_stamp, "%s/%s-%02d",
            dirname,prefix,this_run->output_indx);

    output_data(mesh, src, this_run, prefix_stamp);
    this_run->output_indx++;
  }
}
