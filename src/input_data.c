#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#include "run_param.h"
#include "fluid.h"
#include "source.h"
#include "prototype.h"

#ifdef __COSMOLOGICAL__
#include "cosmology.h"
#endif

#define FILENAME_LENGTH (256)

void setup_data(struct run_param *this_run)
{
  float dx, dy, dz;

  this_run->xmin = this_run->ymin = this_run->zmin = 0.0;
  this_run->xmax = this_run->ymax = this_run->zmax = 1.0;

  dx = (this_run->xmax-this_run->xmin)/(float)NNODE_X;
  dy = (this_run->ymax-this_run->ymin)/(float)NNODE_Y;
  dz = (this_run->zmax-this_run->zmin)/(float)NNODE_Z;

  this_run->xmin_local = this_run->rank_x*dx;
  this_run->xmax_local = this_run->xmin_local+dx;
  
  this_run->ymin_local = this_run->rank_y*dy;
  this_run->ymax_local = this_run->ymin_local+dy;

  this_run->xmin_local = this_run->rank_x*dx;
  this_run->xmax_local = this_run->xmin_local+dx;
  
  this_run->ymin_local = this_run->rank_y*dy;
  this_run->ymax_local = this_run->ymin_local+dy;

  this_run->nmesh_x_total = this_run->nmesh_y_total = this_run->nmesh_z_total = 128;
  this_run->nmesh_x_local = this_run->nmesh_y_local = this_run->nmesh_z_local = 32;

  this_run->delta_x = (this_run->xmax-this_run->xmin)/(float)this_run->nmesh_x_total;
  this_run->delta_y = (this_run->ymax-this_run->ymin)/(float)this_run->nmesh_y_total;
  this_run->delta_z = (this_run->zmax-this_run->zmin)/(float)this_run->nmesh_z_total;
}

void input_mesh_header(struct run_param *this_run, char *filename)
{
  struct run_param input_run;
  FILE *input_fp;
  
  input_fp = fopen(filename,"r");
  fread(&input_run, sizeof(struct run_param), 1, input_fp);
  fclose(input_fp);

  if(input_run.nmesh_x_total != NMESH_X_TOTAL ||
     input_run.nmesh_y_total != NMESH_Y_TOTAL ||
     input_run.nmesh_z_total != NMESH_Z_TOTAL) {
    fprintf(stderr, "# Inconsistent size of the global mesh\n");
    fprintf(stderr, "# input_run.nmesh_x_total = %d\n", input_run.nmesh_x_total);
    fprintf(stderr, "# input_run.nmesh_y_total = %d\n", input_run.nmesh_y_total);
    fprintf(stderr, "# input_run.nmesh_z_total = %d\n", input_run.nmesh_z_total);
    exit(EXIT_FAILURE);
  }

  if(input_run.nmesh_x_local != NMESH_X_LOCAL ||
     input_run.nmesh_y_local != NMESH_Y_LOCAL ||
     input_run.nmesh_z_local != NMESH_Z_LOCAL) {
    fprintf(stderr, "# Inconsistent size of the local mesh\n");
    fprintf(stderr, "# input_run.nmesh_x_local = %d\n", input_run.nmesh_x_local);
    fprintf(stderr, "# input_run.nmesh_y_local = %d\n", input_run.nmesh_y_local);
    fprintf(stderr, "# input_run.nmesh_z_local = %d\n", input_run.nmesh_z_local);
    exit(EXIT_FAILURE);
  }

  if(input_run.mpi_nproc != NNODE_X*NNODE_Y*NNODE_Z) {
    fprintf(stderr, "# Inconsistent number of MPI processes\n");
    fprintf(stderr, "# input_run.mpi_nproc = %d\n", input_run.mpi_nproc);
    fprintf(stderr, "# this_run.mpi_nproc = %d\n", NNODE_X*NNODE_Y*NNODE_Z);
    exit(EXIT_FAILURE);
  }

  if(input_run.nnode_x != NNODE_X ||
     input_run.nnode_y != NNODE_Y ||
     input_run.nnode_z != NNODE_Z) {
    fprintf(stderr, "# Inconsistent domain decomposition\n");
    fprintf(stderr, "# input_run.nnode_x = %d\n",input_run.nnode_x);
    fprintf(stderr, "# input_run.nnode_y = %d\n",input_run.nnode_y);
    fprintf(stderr, "# input_run.nnode_z = %d\n",input_run.nnode_z);
    exit(EXIT_FAILURE);
  }

  if(input_run.nspecies != NSPECIES) {
    fprintf(stderr, "# Inconsistent number of chemical species \n");
    fprintf(stderr, "# input_run.nspecies = %d\n", input_run.nspecies);
    exit(EXIT_FAILURE);
  }

  if(input_run.nchannel != NCHANNEL) {
    fprintf(stderr, "# Inconsistent channel number of radiation transfer \n");
    fprintf(stderr, "# input_run.nchannel = %d \n", input_run.nchannel);
    exit(EXIT_FAILURE);
  }

  input_fp = fopen(filename,"r");
  fread(this_run, sizeof(struct run_param), 1, input_fp);
  fclose(input_fp);

}

void input_mesh_single(struct fluid_mesh *mesh, struct run_param *this_run,
		       char *filename)
{
  FILE *input_fp;

  input_mesh_header(this_run,filename);
  
  input_fp = fopen(filename,"r");
  
  fread(this_run, sizeof(struct run_param), 1, input_fp);
#if 0
  fread(mesh, sizeof(struct fluid_mesh), 
	NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL,
	input_fp);
#else
  int imesh;
  for(imesh=0;imesh<NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL;imesh++) {
    struct fluid_mesh_io temp_mesh;
    fread(&temp_mesh, sizeof(struct fluid_mesh_io), 1, input_fp);
    mesh[imesh].dens = temp_mesh.dens;
    mesh[imesh].eneg = temp_mesh.eneg;
    mesh[imesh].momx = temp_mesh.momx;
    mesh[imesh].momy = temp_mesh.momy;
    mesh[imesh].momz = temp_mesh.momz;
    mesh[imesh].uene  = temp_mesh.uene;
    mesh[imesh].pot  = temp_mesh.pot;
    mesh[imesh].chem = temp_mesh.chem;
    /*
    mesh[imesh].uene = (mesh[imesh].eneg - 
			0.5*(SQR(mesh[imesh].momx)+
			     SQR(mesh[imesh].momy)+
			     SQR(mesh[imesh].momz))/mesh[imesh].dens)/mesh[imesh].dens;
    */
    mesh[imesh].prev_chem = mesh[imesh].chem;
    mesh[imesh].prev_uene = mesh[imesh].uene;
  }
#endif
  fclose(input_fp);
}

void input_mesh(struct fluid_mesh *mesh, struct run_param *this_run,
		char *prefix)
{

  static char filename[FILENAME_LENGTH];

  sprintf(filename, "%s_%03d_%03d_%03d", prefix, 
	  this_run->rank_x, this_run->rank_y, this_run->rank_z);

  input_mesh_single(mesh, this_run, filename);

}

void input_src(struct radiation_src *src, struct run_param *this_run, 
	       char *prefix)
{
  static char filename[FILENAME_LENGTH];
  FILE *input_fp;

  uint64_t input_nsrc;

  sprintf(filename,"%s_src.dat", prefix);

  input_fp = fopen(filename, "r");
  
  if(input_fp == NULL) {
    fprintf(stderr, "# File %s not found\n", filename);
    exit(EXIT_FAILURE);
  }
  
  fread(&(this_run->nsrc), sizeof(uint64_t), 1, input_fp);

  if(this_run->nsrc > NSOURCE_MAX) {
    fprintf(stderr, "# Exceeds the max. number of the sources\n");
    fprintf(stderr, "# input_nsrc = %llu\n", this_run->nsrc);
    exit(EXIT_FAILURE);
  }

  fread(&this_run->freq, sizeof(struct freq_param), 1, input_fp);
  fread(src, sizeof(struct radiation_src), this_run->nsrc, input_fp);

  fclose(input_fp);
}

void input_src_file(struct radiation_src *src, struct run_param *this_run, 
                    char *src_file)
{
  FILE *input_fp;

  uint64_t input_nsrc;

  input_fp = fopen(src_file, "r");
  
  if(input_fp == NULL) {
    fprintf(stderr, "# File %s not found\n", src_file);
    exit(EXIT_FAILURE);
  }
  
  fread(&(this_run->nsrc), sizeof(uint64_t), 1, input_fp);

  if(this_run->nsrc > NSOURCE_MAX) {
    fprintf(stderr, "# Exceeds the max. number of the sources\n");
    fprintf(stderr, "# input_nsrc = %llu\n", this_run->nsrc);
    exit(EXIT_FAILURE);
  }

  fread(&this_run->freq, sizeof(struct freq_param), 1, input_fp);
  fread(src, sizeof(struct radiation_src), this_run->nsrc, input_fp);

  fclose(input_fp);
}


void input_data(struct fluid_mesh *mesh, struct radiation_src *src, 
		struct run_param *this_run, char *prefix)
{
  input_mesh(mesh, this_run, prefix);
  input_src(src, this_run, prefix);

#ifdef __COSMOLOGICAL__
  update_expansion(this_run->tnow, this_run);
#endif
}
