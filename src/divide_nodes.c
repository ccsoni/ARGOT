#include <stdio.h>
#include <stdlib.h>

#include "run_param.h"
#include "fluid.h"

void make_directory(char*);

#define MESH(ix,iy,iz) (mesh[(iz)+nz_total*((iy)+ny_total*(ix))])
#define MESHL(ix,iy,iz) (mesh_local[(iz)+nz_local*((iy)+ny_local*(ix))])

int mpi_rank_div(int rank_x, int rank_y, int rank_z, struct run_param *this_run)
{
  int ix, iy, iz;

#ifdef __ISOLATED__
  if(rank_x>=this_run->nnode_x || rank_x < 0 ||
     rank_y>=this_run->nnode_y || rank_y < 0 ||
     rank_z>=this_run->nnode_z || rank_z < 0){
    return (-1);
  }else{
    return ((rank_z)+this_run->nnode_z*((rank_y)+this_run->nnode_y*(rank_x)));
  }
#else /* __PERIODIC__ */
  ix = (rank_x + this_run->nnode_x) % this_run->nnode_x;
  iy = (rank_y + this_run->nnode_y) % this_run->nnode_y;
  iz = (rank_z + this_run->nnode_z) % this_run->nnode_z;

  return ((iz)+this_run->nnode_z*((iy)+this_run->nnode_y*(ix)));
#endif
}

void read_header_div(struct run_param *this_run, char *filename)
{
  FILE *input_fp;
  input_fp = fopen(filename,"r");
  fread(this_run, sizeof(struct run_param), 1, input_fp);
  fclose(input_fp);
}

void input_mesh_single_div(struct fluid_mesh *mesh, struct run_param *this_run,
			   char *filename)
{
  FILE *input_fp;

  read_header_div(this_run,filename);
  
  input_fp = fopen(filename,"r");
  
  fread(this_run, sizeof(struct run_param), 1, input_fp);

  int imesh;
  for(imesh=0;imesh<this_run->nmesh_x_local*this_run->nmesh_y_local*this_run->nmesh_z_local;imesh++) {
    struct fluid_mesh_io temp_mesh;
    fread(&temp_mesh, sizeof(struct fluid_mesh_io), 1, input_fp);
    mesh[imesh].dens = temp_mesh.dens;
    mesh[imesh].eneg = temp_mesh.eneg;
    mesh[imesh].momx = temp_mesh.momx;
    mesh[imesh].momy = temp_mesh.momy;
    mesh[imesh].momz = temp_mesh.momz;
    mesh[imesh].uene = temp_mesh.uene;
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

  fclose(input_fp);
}

void input_mesh_div(struct fluid_mesh *mesh, struct run_param *this_run,
		    char *prefix)
{

  static char filename[256];

  sprintf(filename, "%s_%03d_%03d_%03d", prefix, 
	  this_run->rank_x, this_run->rank_y, this_run->rank_z);

  input_mesh_single_div(mesh, this_run, filename);
}


void output_mesh_single_div(struct fluid_mesh *mesh, struct run_param *this_run,
			char *filename)
{

  FILE *output_fp;
  output_fp = fopen(filename,"w");

  fwrite(this_run, sizeof(struct run_param), 1, output_fp);
  int imesh;
  for(imesh=0;imesh<this_run->nmesh_x_local*this_run->nmesh_y_local*this_run->nmesh_z_local;imesh++) {
    struct fluid_mesh_io tmp_mesh;
    tmp_mesh.dens = mesh[imesh].dens;
    tmp_mesh.eneg = mesh[imesh].eneg;
    tmp_mesh.momx = mesh[imesh].momx;
    tmp_mesh.momy = mesh[imesh].momy;
    tmp_mesh.momz = mesh[imesh].momz;
    tmp_mesh.uene = mesh[imesh].uene;
    //    tmp_mesh.pot  = mesh[imesh].pot;
    tmp_mesh.chem = mesh[imesh].chem;
    fwrite(&tmp_mesh, sizeof(struct fluid_mesh_io), 1, output_fp);
  }
  fclose(output_fp);
}

void output_mesh_div(struct fluid_mesh *mesh, struct run_param *this_run,
		 char *prefix)
{

  static char filename[256];

  sprintf(filename,"%s_%03d_%03d_%03d",prefix, 
	  this_run->rank_x, this_run->rank_y, this_run->rank_z);

  output_mesh_single_div(mesh, this_run, filename);
}



int main(int argc,char **argv)
{
  printf("Usage: <input prefix> <output-name> <NX> <NY> <NZ>\n");
  
  char input_prefix[256], output_dir[256], output_label[256], base_file[256];

  struct run_param this_run;
  struct fluid_mesh *mesh, *mesh_local;

  int onx, ony, onz;
  
  sprintf(input_prefix, "%s", argv[1]);
  sprintf(output_dir, "%s-init", argv[2]);
  sprintf(output_label, "%s-init/%s-init", argv[2],argv[2]);

  onx = atoi(argv[3]);
  ony = atoi(argv[4]);
  onz = atoi(argv[5]);
  
  make_directory(output_dir);
  sprintf(base_file,"%s_000_000_000",input_prefix);
  read_header_div(&this_run, base_file);

  int nx_total, ny_total, nz_total;
  int nx_local, ny_local, nz_local;

  nx_total = this_run.nmesh_x_total;
  ny_total = this_run.nmesh_y_total;
  nz_total = this_run.nmesh_z_total;

  nx_local = nx_total/onx;
  ny_local = ny_total/ony;
  nz_local = nz_total/onz;
  
  mesh = (struct fluid_mesh*) malloc (sizeof(struct fluid_mesh)*
				      nx_total*ny_total*nz_total);

  input_mesh_div(mesh, &this_run, input_prefix);
  
  this_run.nmesh_x_local = nx_local;
  this_run.nmesh_y_local = ny_local;
  this_run.nmesh_z_local = nz_local;

  this_run.nnode_x = onx;
  this_run.nnode_y = ony;
  this_run.nnode_z = onz;

  mesh_local = (struct fluid_mesh*) malloc (sizeof(struct fluid_mesh)*
					    nx_local*ny_local*nz_local);
  
  for(int inx=0; inx<onx; inx++) {
    float dx_domain = (this_run.xmax-this_run.xmin)/(float)onx;
    this_run.xmin_local = this_run.xmin + (float)inx*dx_domain;
    this_run.xmax_local = this_run.xmin_local + dx_domain;
    
    for(int iny=0; iny<ony; iny++) {
      float dy_domain = (this_run.ymax-this_run.ymin)/(float)ony;
      this_run.ymin_local = this_run.ymin + (float)iny*dy_domain;
      this_run.ymax_local = this_run.ymin_local + dy_domain;
      
      for(int inz=0; inz<onz; inz++) {
	float dz_domain = (this_run.zmax-this_run.zmin)/(float)onz;
	this_run.zmin_local = this_run.zmin + (float)inz*dz_domain;
	this_run.zmax_local = this_run.zmin_local + dz_domain;

	this_run.rank_x = inx;
	this_run.rank_y = iny;
	this_run.rank_z = inz;

	this_run.mpi_nproc = onx*ony*onz;
	this_run.mpi_rank = mpi_rank_div(inx,iny,inz,&this_run);
	
	for(int ix=0; ix<nx_local; ix++) {
	  for(int iy=0; iy<ny_local; iy++) {
	    for(int iz=0; iz<nz_local; iz++) {

	      int gix, giy, giz;
	      gix = ix + nx_local*inx;
	      giy = iy + ny_local*iny;
	      giz = iz + nz_local*inz;

	      MESHL(ix,iy,iz) = MESH(gix,giy,giz);
	    }
	  }
	}

	output_mesh_div(mesh_local, &this_run, output_label);
      }
    }
  }

  free(mesh);
  free(mesh_local);
  
  return EXIT_SUCCESS;
}
