#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "constants.h"
#include "run_param.h"
#include "fluid.h"
#include "radiation.h"

#include "cpgplot.h"

#define MESH(ix,iy,iz) mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))]

void input_mesh_single(struct fluid_mesh*, struct run_param*, char*);
void input_mesh_header(struct run_param*, char*);
void view_map(float*, float*, float*, int, int, int, int, int, int, 
	      float, float, char*, char*);

void compile_map_x(struct fluid_mesh *mesh, 
		   float *map_frac, float *map_tmpr, float *map_dens,
		   float *pos1, float *pos2, float slice_pos,
		   struct run_param *this_run)
{
  int ix, iy, iz;
  int jx, jy, jz;
#define MAPFRAC(iy, iz) (map_frac[(iz)+NMESH_Z_TOTAL*(iy)])
#define MAPTMPR(iy, iz) (map_tmpr[(iz)+NMESH_Z_TOTAL*(iy)])
#define MAPDENS(iy, iz) (map_dens[(iz)+NMESH_Z_TOTAL*(iy)])
  ix = (int)((slice_pos - this_run->xmin_local)/this_run->delta_x);

  for(iy=0;iy<NMESH_Y_LOCAL;iy++) {
    int jy = iy+this_run->rank_y*NMESH_Y_LOCAL;
    pos1[jy] = this_run->ymin_local + ((float)iy+0.5)*this_run->delta_y;
    for(iz=0;iz<NMESH_Z_LOCAL;iz++) {
      int jz = iz+this_run->rank_z*NMESH_Z_LOCAL;
      pos2[jz] = this_run->zmin_local + ((float)iz+0.5)*this_run->delta_z;

      float wmol = WMOL(MESH(ix,iy,iz).chem);

      MAPFRAC(jy,jz) = MESH(ix,iy,iz).chem.fHI;
      MAPTMPR(jy,jz) = log10(MESH(ix,iy,iz).uene*this_run->uenetok*wmol);
      MAPDENS(jy,jz) = log10(MESH(ix,iy,iz).dens*this_run->denstonh);
    }
  }
#undef MAPFRAC
#undef MAPTMPR
#undef MAPDENS
}

void compile_map_y(struct fluid_mesh *mesh, 
		   float *map_frac, float *map_tmpr, float *map_dens,
		   float *pos1, float *pos2, float slice_pos,
		   struct run_param *this_run)
{
  int ix, iy, iz;
  int jx, jy, jz;
#define MAPFRAC(ix, iz) (map_frac[(iz)+NMESH_Z_TOTAL*(ix)])
#define MAPTMPR(ix, iz) (map_tmpr[(iz)+NMESH_Z_TOTAL*(ix)])
#define MAPDENS(ix, iz) (map_dens[(iz)+NMESH_Z_TOTAL*(ix)])
  iy = (int)((slice_pos - this_run->ymin_local)/this_run->delta_y);

  for(ix=0;ix<NMESH_X_LOCAL;ix++) {
    int jx = ix+this_run->rank_x*NMESH_X_LOCAL;
    pos1[jx] = this_run->xmin_local + ((float)ix+0.5)*this_run->delta_x;
    for(iz=0;iz<NMESH_Z_LOCAL;iz++) {
      int jz = iz+this_run->rank_z*NMESH_Z_LOCAL;
      pos2[jz] = this_run->zmin_local + ((float)iz+0.5)*this_run->delta_z;

      float wmol = WMOL(MESH(ix,iy,iz).chem);

      MAPFRAC(jx,jz) = MESH(ix,iy,iz).chem.fHI;
      MAPTMPR(jx,jz) = log10(MESH(ix,iy,iz).uene*this_run->uenetok*wmol);
      MAPDENS(jx,jz) = log10(MESH(ix,iy,iz).dens*this_run->denstonh);
    }
  }
#undef MAPFRAC
#undef MAPTMPR
#undef MAPDENS
}

void compile_map_z(struct fluid_mesh *mesh, 
		   float *map_frac, float *map_tmpr, float *map_dens,
		   float *pos1, float *pos2, float slice_pos,
		   struct run_param *this_run)
{
  int ix, iy, iz;
  int jx, jy, jz;
#define MAPFRAC(ix, iy) (map_frac[(iy)+NMESH_Y_TOTAL*(ix)])
#define MAPTMPR(ix, iy) (map_tmpr[(iy)+NMESH_Y_TOTAL*(ix)])
#define MAPDENS(ix, iy) (map_dens[(iy)+NMESH_Y_TOTAL*(ix)])
  iz = (int)((slice_pos - this_run->zmin_local)/this_run->delta_z);

  for(ix=0;ix<NMESH_X_LOCAL;ix++) {
    int jx = ix+this_run->rank_x*NMESH_X_LOCAL;
    pos1[jx] = this_run->xmin_local + ((float)ix+0.5)*this_run->delta_x;
    for(iy=0;iy<NMESH_Y_LOCAL;iy++) {
      int jy = iy+this_run->rank_y*NMESH_Y_LOCAL;
      pos2[jy] = this_run->ymin_local + ((float)iy+0.5)*this_run->delta_y;

      float wmol = WMOL(MESH(ix,iy,iz).chem);

      MAPFRAC(jx,jy) = MESH(ix,iy,iz).chem.fHI;
      MAPTMPR(jx,jy) = log10(MESH(ix,iy,iz).uene*this_run->uenetok*wmol);
      MAPDENS(jx,jy) = log10(MESH(ix,iy,iz).dens*this_run->denstonh);
    }
  }
#undef MAPFRAC
#undef MAPTMPR
#undef MAPDENS
}

float get_tick(float min, float max)
{
  float eff, sft;

  eff = max-min;
  sft = 1.0;
  while(eff >= 10.0) {
    eff /= 10.0; sft *= 10.0;
  }
  while(eff < 1.0) {
    eff *= 10.0;
    sft /= 10.0;
  }

  float tick;
  
  if(eff > 5.0) {
    tick = sft;
  }else if(eff > 2.0) {
    tick = sft*0.5;
  }else{
    tick = sft*0.2;
  }

  return tick;
}

void get_minmax(float fmin, float fmax, float *vmin, float *vmax)
{
  float bln;
  float fave;

  bln = 0.5*(fmin+fmax);
  
  if(fmin > bln && fmax > bln) {
    fmin = bln;
  }else if(fmin<bln && fmax<bln) {
    fmax = bln;
  }else if(fmin==fmax) {
    fmax += 0.9;
  }

  float tick = get_tick(fmin, fmax);
  
  int bln_bin = bln/tick;
  bln = bln_bin * tick;

  float list_max, list_min;
  
  list_max = list_min = bln;

  if(fmax != bln) {
    for(int i=1;i<=(fmax-bln)/tick+1;i++) {
      list_min = MIN(list_min, i*tick+bln);
      list_max = MAX(list_max, i*tick+bln);
    }
  }

  if(fmin != bln) {
    for(int i=-1;i>=(fmin-bln)/tick-1;i--) {
      list_min = MIN(list_min, i*tick+bln);
      list_max = MAX(list_max, i*tick+bln);      
    }
  }

  *vmin = list_min;
  *vmax = list_max;
}

int main(int argc, char **argv) 
{

  static struct run_param this_run;

  static struct fluid_mesh mesh[NMESH_LOCAL];
  static struct radiation_src src[NSOURCE_MAX];

  static char filename[256];

  float slice_pos;
  float *map_frac, *map_tmpr, *map_dens, *pos1, *pos2;

  this_run.proc_file=stdout;

  if(argc != 6) {
    fprintf(stderr, 
	    "Usage: %s <input prefix> <X|Y|Z> <slice position> <vmin> <vmax>\n",
	    argv[0]);
    exit(EXIT_FAILURE);
  }

  slice_pos = atof(argv[3]);

  if(strcmp(argv[2],"X") == 0) {
    map_frac = (float *) malloc(sizeof(float)*NMESH_Y_TOTAL*NMESH_Z_TOTAL);
    map_tmpr = (float *) malloc(sizeof(float)*NMESH_Y_TOTAL*NMESH_Z_TOTAL);
    map_dens = (float *) malloc(sizeof(float)*NMESH_Y_TOTAL*NMESH_Z_TOTAL);
    pos1 = (float *) malloc(sizeof(float)*NMESH_Y_TOTAL);
    pos2 = (float *) malloc(sizeof(float)*NMESH_Z_TOTAL);
  }else if(strcmp(argv[2],"Y") == 0) {
    map_frac = (float *) malloc(sizeof(float)*NMESH_Z_TOTAL*NMESH_X_TOTAL);
    map_tmpr = (float *) malloc(sizeof(float)*NMESH_Z_TOTAL*NMESH_X_TOTAL);
    map_dens = (float *) malloc(sizeof(float)*NMESH_Z_TOTAL*NMESH_X_TOTAL);
    pos1 = (float *) malloc(sizeof(float)*NMESH_X_TOTAL);
    pos2 = (float *) malloc(sizeof(float)*NMESH_Z_TOTAL);
  }else if(strcmp(argv[2],"Z") == 0) {
    map_frac = (float *) malloc(sizeof(float)*NMESH_X_TOTAL*NMESH_Y_TOTAL);
    map_tmpr = (float *) malloc(sizeof(float)*NMESH_X_TOTAL*NMESH_Y_TOTAL);
    map_dens = (float *) malloc(sizeof(float)*NMESH_X_TOTAL*NMESH_Y_TOTAL);
    pos1 = (float *) malloc(sizeof(float)*NMESH_X_TOTAL);
    pos2 = (float *) malloc(sizeof(float)*NMESH_Y_TOTAL);
  }else{
    fprintf(stderr, "Invalid slice direction\n");
    exit(EXIT_FAILURE);
  }

  int rank_x, rank_y, rank_z;
  for(rank_x=0;rank_x<NNODE_X;rank_x++) {
    for(rank_y=0;rank_y<NNODE_Y;rank_y++) {
      for(rank_z=0;rank_z<NNODE_Z;rank_z++) {
	FILE *fp;
	
	sprintf(filename, "%s_%03d_%03d_%03d", 
		argv[1], rank_x, rank_y, rank_z);

	fp = fopen(filename,"r");
	input_mesh_header(&this_run, filename);
	fclose(fp);
	
	if(strcmp(argv[2],"X") == 0 &&
	   this_run.xmin_local < slice_pos &&
	   this_run.xmax_local >= slice_pos ) {

	  fp = fopen(filename,"r");
	  input_mesh_single(mesh, &this_run, filename);
	  fclose(fp);
	  
	  compile_map_x(mesh, map_frac, map_tmpr, map_dens, 
			pos1, pos2, slice_pos, &this_run);
	  
	}else if(strcmp(argv[2],"Y") == 0 &&
		 this_run.ymin_local < slice_pos &&
		 this_run.ymax_local >= slice_pos) {

	  fp = fopen(filename,"r");
	  input_mesh_single(mesh, &this_run, filename);
	  fclose(fp);

	  compile_map_y(mesh, map_frac, map_tmpr, map_dens, 
			pos1, pos2, slice_pos, &this_run);

	}else if(strcmp(argv[2],"Z") == 0 &&
		 this_run.zmin_local < slice_pos &&
		 this_run.zmax_local >= slice_pos) {
	  
	  fp = fopen(filename,"r");
	  input_mesh_single(mesh, &this_run, filename);
	  fclose(fp);

	  compile_map_z(mesh, map_frac, map_tmpr, map_dens, 
			pos1, pos2, slice_pos, &this_run);

	}

      }
    }
  }

  float fmin_frac, fmax_frac;
  float fmin_tmpr, fmax_tmpr;
  float fmin_dens, fmax_dens;
  fmin_frac = fmin_tmpr = fmin_dens = FLT_MAX;
  fmax_frac = fmax_tmpr = fmax_dens = -FLT_MAX;
  if(strcmp(argv[2],"X")==0) {
    for(int imesh=0;imesh<NMESH_Z_TOTAL*NMESH_Y_TOTAL;imesh++) {
      fmin_frac = MIN(map_frac[imesh], fmin_frac);
      fmax_frac = MAX(map_frac[imesh], fmax_frac);
      fmin_tmpr = MIN(map_tmpr[imesh], fmin_tmpr);
      fmax_tmpr = MAX(map_tmpr[imesh], fmax_tmpr);
      fmin_dens = MIN(map_dens[imesh], fmin_dens);
      fmax_dens = MAX(map_dens[imesh], fmax_dens);
    }
  }else if(strcmp(argv[2],"Y")==0) {
    for(int imesh=0;imesh<NMESH_Z_TOTAL*NMESH_X_TOTAL;imesh++) {
      fmin_frac = MIN(map_frac[imesh], fmin_frac);
      fmax_frac = MAX(map_frac[imesh], fmax_frac);
      fmin_tmpr = MIN(map_tmpr[imesh], fmin_tmpr);
      fmax_tmpr = MAX(map_tmpr[imesh], fmax_tmpr);
      fmin_dens = MIN(map_dens[imesh], fmin_dens);
      fmax_dens = MAX(map_dens[imesh], fmax_dens);
    }
  }else if(strcmp(argv[2],"Z")==0) {
    for(int imesh=0;imesh<NMESH_X_TOTAL*NMESH_Y_TOTAL;imesh++) {
      fmin_frac = MIN(map_frac[imesh], fmin_frac);
      fmax_frac = MAX(map_frac[imesh], fmax_frac);
      fmin_tmpr = MIN(map_tmpr[imesh], fmin_tmpr);
      fmax_tmpr = MAX(map_tmpr[imesh], fmax_tmpr);
      fmin_dens = MIN(map_dens[imesh], fmin_dens);
      fmax_dens = MAX(map_dens[imesh], fmax_dens);
    }
  }

#if 0
  printf("# max value for fraction : %14.6e \n", fmax_frac);
  printf("# min value for fraction : %14.6e \n", fmin_frac);
  printf("# max value for temperature : %14.6e \n", fmax_tmpr);
  printf("# min value for temperature : %14.6e \n", fmin_tmpr);
  printf("# max value for number density : %14.6e \n", fmax_dens);
  printf("# min value for number density : %14.6e \n", fmin_dens);
#endif

  /* draw the maps */

  float vmin_frac, vmax_frac, vmin_tmpr, vmax_tmpr, vmin_dens, vmax_dens;
  int pgplot_err;

  get_minmax(fmin_frac, fmax_frac, &vmin_frac, &vmax_frac);
  get_minmax(fmin_tmpr, fmax_tmpr, &vmin_tmpr, &vmax_tmpr);
  get_minmax(fmin_dens, fmax_dens, &vmin_dens, &vmax_dens);

#if 1
  printf("# max value for fraction : %14.6e \n", vmax_frac);
  printf("# min value for fraction : %14.6e \n", vmin_frac);
  printf("# max value for temperature : %14.6e \n", vmax_tmpr);
  printf("# min value for temperature : %14.6e \n", vmin_tmpr);
  printf("# max value for number density : %14.6e \n", vmax_dens);
  printf("# min value for number density : %14.6e \n", vmin_dens);
#endif
  

  //  vmin = atof(argv[4]);
  //  vmax = atof(argv[5]);

  pgplot_err = cpgopen("?");

  if(strcmp(argv[2],"X")==0) {

    view_map(map_frac, pos1, pos2, NMESH_Y_TOTAL, NMESH_Z_TOTAL, 
             1, NMESH_Y_TOTAL, 1, NMESH_Z_TOTAL, vmin_frac, vmax_frac, "Z", "Y");
    view_map(map_tmpr, pos1, pos2, NMESH_Y_TOTAL, NMESH_Z_TOTAL, 
             1, NMESH_Y_TOTAL, 1, NMESH_Z_TOTAL, vmin_tmpr, vmax_tmpr, "Z", "Y");

  }else if(strcmp(argv[2],"Y")==0) {

    view_map(map_frac, pos1, pos2, NMESH_X_TOTAL, NMESH_Z_TOTAL, 
             1, NMESH_X_TOTAL, 1, NMESH_Z_TOTAL, vmin_frac, vmax_frac, "Z", "X");
    view_map(map_tmpr, pos1, pos2, NMESH_X_TOTAL, NMESH_Z_TOTAL, 
             1, NMESH_X_TOTAL, 1, NMESH_Z_TOTAL, vmin_tmpr, vmax_tmpr, "Z", "X");

  }else if(strcmp(argv[2],"Z")==0) {

    view_map(map_frac, pos1, pos2, NMESH_X_TOTAL, NMESH_Y_TOTAL, 
             1, NMESH_X_TOTAL, 1, NMESH_Y_TOTAL, vmin_frac, vmax_frac, "Y", "X");
    view_map(map_tmpr, pos1, pos2, NMESH_X_TOTAL, NMESH_Y_TOTAL, 
             1, NMESH_X_TOTAL, 1, NMESH_Y_TOTAL, vmin_tmpr, vmax_tmpr, "Y", "X");

  }

  cpgclos();

  free(pos1);
  free(pos2);
  free(map_frac);
  free(map_tmpr);
  free(map_dens);

}
