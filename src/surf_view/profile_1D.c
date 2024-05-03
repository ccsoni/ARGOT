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

//typedef float REAL;
typedef double REAL;

#define MESH(ix,iy,iz) mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))]

void input_mesh_single(struct fluid_mesh*, struct run_param*, char*);
void input_mesh_header(struct run_param*, char*);

#define NRADBIN (256)

int main(int argc, char **argv) 
{
  static struct run_param this_run;

  static struct fluid_mesh mesh[NMESH_LOCAL];
  static struct radiation_src src[NSOURCE_MAX];
  
  static REAL rad[NRADBIN], frac[NRADBIN], tmpr[NRADBIN];
  static REAL nh[NRADBIN], vrad[NRADBIN], nin[NRADBIN];

  static char filename[256];

  REAL xcent, ycent, zcent, range;

  this_run.proc_file = stdout;

  if(argc != 6) {
    fprintf(stderr, 
	    "Usage: %s <input prefix> <profile center x y z> <range>\n",
	    argv[0]);
    exit(EXIT_FAILURE);
  }

  xcent = atof(argv[2]);
  ycent = atof(argv[3]);
  zcent = atof(argv[4]);
  range = atof(argv[5]);

  REAL radmin = 0.0;
  REAL radmax = range;
  REAL drad = (radmax-radmin)/NRADBIN;
  for(int ibin=0;ibin<NRADBIN;ibin++) {
    rad[ibin] = radmin + (ibin+0.5)*drad;
    frac[ibin] = 0.0;
    tmpr[ibin] = 0.0;
    vrad[ibin] = 0.0;
    nh[ibin] = 0.0;
    nin[ibin] = 0.0;
  }

  int rank_x, rank_y, rank_z;
  for(rank_x=0;rank_x<NNODE_X;rank_x++) {
    for(rank_y=0;rank_y<NNODE_Y;rank_y++) {
      for(rank_z=0;rank_z<NNODE_Z;rank_z++) {
	FILE *fp;
	
	sprintf(filename, "%s_%03d_%03d_%03d", 
		argv[1], rank_x, rank_y, rank_z);

	fp = fopen(filename,"r");
	input_mesh_single(mesh, &this_run, filename);
	fclose(fp);

	for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
	  REAL xpos = this_run.xmin_local + (ix+0.5)*this_run.delta_x;
	  for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
	    REAL ypos = this_run.ymin_local + (iy+0.5)*this_run.delta_y;
	    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	      REAL zpos = this_run.zmin_local + (iz+0.5)*this_run.delta_z;

	      REAL rad = sqrtf(SQR(xpos-xcent) + SQR(ypos-ycent) + SQR(zpos-zcent));
	      REAL wmol = WMOL(MESH(ix,iy,iz).chem);

#if 0
	      if(rad < range) {
		printf("%14.6e %14.6e %14.6e %14.6e\n", 
		       rad, 
		       MESH(ix,iy,iz).chem.fHI, 
		       MESH(ix,iy,iz).uene*this_run.uenetok*wmol,
		       MESH(ix,iy,iz).dens*this_run.denstonh);
	      }
#else
	      int ibin = (rad-radmin)/drad;
	      if(ibin<NRADBIN) {
		frac[ibin] += MESH(ix,iy,iz).chem.fHI;
		tmpr[ibin] += MESH(ix,iy,iz).uene*this_run.uenetok*wmol;
		vrad[ibin] += (MESH(ix,iy,iz).momx/MESH(ix,iy,iz).dens*(xpos-xcent) +
			       MESH(ix,iy,iz).momy/MESH(ix,iy,iz).dens*(ypos-ycent) +
			       MESH(ix,iy,iz).momz/MESH(ix,iy,iz).dens*(zpos-zcent))/rad;
		nh[ibin]   += MESH(ix,iy,iz).dens*this_run.denstonh;
		nin[ibin]  += 1.0;
	      }
#endif
	    }
	  }
	}

      }
    }
  }

  for(int ibin=0;ibin<NRADBIN;ibin++) {
    frac[ibin] /= nin[ibin];
    tmpr[ibin] /= nin[ibin];
    vrad[ibin] /= nin[ibin];
    nh[ibin]   /= nin[ibin];
    if(nin[ibin]!=0) printf("%14.6e %14.6e %14.6e %14.6e %14.6e \n", 
			    rad[ibin], frac[ibin], tmpr[ibin], nh[ibin], vrad[ibin]);
  }
		      
}
