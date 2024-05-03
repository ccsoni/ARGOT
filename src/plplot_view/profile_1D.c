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

REAL gamma_total(struct fluid_mesh *mesh, struct run_param *this_run)
{
  REAL gamma_H2_m1_inv; /* 1/(gamma_H2-1) */

  REAL x;
  REAL tmpr, wmol;
  struct prim_chem *chem;

  wmol = WMOL(mesh->chem);
  tmpr = mesh->uene*this_run->uenetok*wmol;

  chem = &(mesh->chem);

  x = 6100.0/tmpr;
  
  //gamma_H2_m1_inv = 0.5*(5.0+2.0*SQR(x)*expf(x)/SQR(expf(x)-1));
  gamma_H2_m1_inv = 0.5*(5.0+2.0*SQR(x)*exp(-x)/SQR(1.0-exp(-x)));

  REAL sum,denom;

  sum = (chem->fHI+chem->fHII)/GAMM1_MONOATOMIC; /* atomic hydrogen */
  denom = chem->fHI+chem->fHII;

#ifdef __HELIUM__
  sum += HELIUM_FACT/GAMM1_MONOATOMIC;
  denom += HELIUM_FACT;
#endif /* __HELIUM__ */

#ifdef __HYDROGEN_MOL__
  sum += chem->fHM/GAMM1_MONOATOMIC;
  sum += chem->fH2II/GAMM1_DIATOMIC;
  sum += chem->fH2I*gamma_H2_m1_inv;
  denom += (chem->fHM + chem->fH2I + chem->fH2II);
#endif
  
  REAL gamma_m1;
  gamma_m1 = denom/sum;

  return (gamma_m1+1.0);
  //return (1.4);
}

int main(int argc, char **argv) 
{
  static struct run_param this_run;

  static struct fluid_mesh mesh[NMESH_LOCAL];
  static struct radiation_src src[NSOURCE_MAX];
  
  static REAL rad[NRADBIN], felec[NRADBIN], fracHI[NRADBIN], fracHII[NRADBIN], tmpr[NRADBIN];
  static REAL nh[NRADBIN], vrad[NRADBIN], nin[NRADBIN];
  static REAL kene[NRADBIN], pene[NRADBIN], tene[NRADBIN], etrp_func[NRADBIN];
  static REAL fracHeI[NRADBIN], fracHeII[NRADBIN], fracHeIII[NRADBIN];
  static REAL fracH2I[NRADBIN], fracH2II[NRADBIN], fracHM[NRADBIN];
 
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

    felec[ibin] = 0.0;
    fracHI[ibin] = 0.0;
    fracHII[ibin] = 0.0;
    tmpr[ibin] = 0.0;
    vrad[ibin] = 0.0;
    nh[ibin] = 0.0;
    nin[ibin] = 0.0;

    kene[ibin] = 0.0;
    pene[ibin] = 0.0;
    tene[ibin] = 0.0;
    etrp_func[ibin] = 0.0;

    fracHeI[ibin]   = 0.0;
    fracHeII[ibin]  = 0.0;
    fracHeIII[ibin] = 0.0;

    fracH2I[ibin]   = 0.0;
    fracH2II[ibin]  = 0.0;
    fracHM[ibin]    = 0.0;
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

	      REAL radd = sqrt(SQR(xpos-xcent)+SQR(ypos-ycent)+SQR(zpos-zcent));

	      struct fluid_mesh *tgt;
	      tgt = &MESH(ix,iy,iz);

	      REAL wmol = WMOL(tgt->chem);
	      REAL gamma = gamma_total(tgt, &this_run);

	      int ibin = (radd-radmin)/drad;
	      if(ibin<NRADBIN) {
		felec[ibin]  += tgt->chem.felec;
		fracHI[ibin] += tgt->chem.fHI;
                fracHII[ibin] += tgt->chem.fHII;
		tmpr[ibin] += tgt->uene*this_run.uenetok*wmol;
		vrad[ibin] += (tgt->momx/tgt->dens*(xpos-xcent) +
			       tgt->momy/tgt->dens*(ypos-ycent) +
			       tgt->momz/tgt->dens*(zpos-zcent))/radd;
		nh[ibin]   += tgt->dens*this_run.denstonh;

		kene[ibin] += 0.5*NORML2(tgt->momx,tgt->momy,tgt->momz)/tgt->dens;
		pene[ibin] += -0.5*tgt->pot*tgt->dens;
		tene[ibin] += tgt->uene*tgt->dens;
		etrp_func[ibin] += (gamma-1.0)*tgt->uene*tgt->dens/pow(tgt->dens,gamma);
		
#ifdef __HELIUM__
		fracHeI[ibin]   += tgt->chem.fHeI;
		fracHeII[ibin]  += tgt->chem.fHeII;
		fracHeIII[ibin] += tgt->chem.fHeIII;
#endif
#ifdef __HYDROGEN_MOL__
		fracH2I[ibin]  += tgt->chem.fH2I;
		fracH2II[ibin] += tgt->chem.fH2II;
		fracHM[ibin]   += tgt->chem.fHM;
#endif
	
		nin[ibin]  += 1.0;
	      }
	    }
	  }
	}

      }
    }
  }

  printf("# 0:rad 1:felec 2:fracHI 3:fracHII 4:tmpr 5:nH 6:vrad 7:kene 8:pene 9:tene 10:etrp_func 11:fracHeI 12:fracHeII 13:fracHeIII 14:fracH2I 15:fracH2II 16:fracHM\n");
  
  for(int ibin=0;ibin<NRADBIN;ibin++) {
    felec[ibin]   /= nin[ibin];
    fracHI[ibin]  /= nin[ibin];
    fracHII[ibin] /= nin[ibin];
  
    tmpr[ibin] /= nin[ibin];
    vrad[ibin] /= nin[ibin];
    nh[ibin]   /= nin[ibin];
    kene[ibin] /= nin[ibin];
    pene[ibin] /= nin[ibin];
    tene[ibin] /= nin[ibin];
    etrp_func[ibin] /= nin[ibin];
    
#ifdef __HELIUM__
    fracHeI[ibin]   /= nin[ibin];
    fracHeII[ibin]  /= nin[ibin];
    fracHeIII[ibin] /= nin[ibin];
#endif
#ifdef __HYDROGEN_MOL__
    fracH2I[ibin]   /= nin[ibin];
    fracH2II[ibin]  /= nin[ibin];
    fracHM[ibin]    /= nin[ibin];
#endif
    

    if(nin[ibin]!=0) {
      printf("%14.6e %14.6e %14.6e %14.6e %14.6e  %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n", 
	     rad[ibin], felec[ibin], fracHI[ibin], fracHII[ibin],
	     tmpr[ibin], nh[ibin], vrad[ibin],
	     kene[ibin], pene[ibin], tene[ibin], etrp_func[ibin],
	     fracHeI[ibin], fracHeII[ibin], fracHeIII[ibin],
	     fracH2I[ibin], fracH2II[ibin], fracHM[ibin]);
    }
      
  }
		      
}
