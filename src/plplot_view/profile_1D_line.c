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

#define NRADBIN (NMESH_X_TOTAL)

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
}

int main(int argc, char **argv)
{
  struct run_param this_run;
  static struct fluid_mesh mesh[NMESH_LOCAL];

  static REAL rad[NRADBIN], felec[NRADBIN], fracHI[NRADBIN], fracHII[NRADBIN];
  static REAL tmpr[NRADBIN], nh[NRADBIN], nin[NRADBIN];
  static REAL kene[NRADBIN], pene[NRADBIN], tene[NRADBIN], etrp_func[NRADBIN];
  static REAL fracHeI[NRADBIN], fracHeII[NRADBIN], fracHeIII[NRADBIN];
  static REAL fracH2I[NRADBIN], fracH2II[NRADBIN], fracHM[NRADBIN],GammaHM[NRADBIN];
  static REAL cs[NRADBIN],pres[NRADBIN],velc[NRADBIN];
  static REAL velx[NRADBIN],vely[NRADBIN],velz[NRADBIN],gamma[NRADBIN];
  
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
  
  for(int ibin=0;ibin<NRADBIN;ibin++) {
    rad[ibin] = 0.0;
    felec[ibin]   = 0.0;
    fracHI[ibin]  = 0.0;
    fracHII[ibin] = 0.0;
    tmpr[ibin] = 0.0;
    nh[ibin] = 0.0;
    nin[ibin] = 0.0;
     
    kene[ibin] = 0.0;
    pene[ibin] = 0.0;
    tene[ibin] = 0.0;
    etrp_func[ibin] = 0.0;
    cs[ibin] = 0.0;
    pres[ibin] = 0.0;
    velc[ibin] = 0.0;
    gamma[ibin] = 0.0;

#ifdef __HELIUM__
    fracHeI[ibin]   = 0.0;
    fracHeII[ibin]  = 0.0;
    fracHeIII[ibin] = 0.0;
#endif
#ifdef __HYDROGEN_MOL__
    fracH2I[ibin]  = 0.0;
    fracH2II[ibin] = 0.0;
    fracHM[ibin]   = 0.0;
#endif
  }

  for(int rank_x=0;rank_x<NNODE_X;rank_x++) {
    int rank_y = (NNODE_Y/2)-1;
    int rank_z = (NNODE_Z/2)-1;
    
    FILE *fp;
    
    sprintf(filename, "%s_%03d_%03d_%03d",
            argv[1], rank_x, rank_y, rank_z);
    
    fp = fopen(filename,"r");
    input_mesh_single(mesh, &this_run, filename);
    fclose(fp);
    
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      int iy = NMESH_Y_LOCAL-1; //(NMESH_Y_TOTAL/2)-1;
      int iz = NMESH_Z_LOCAL-1; //(NMESH_Z_TOTAL/2)-1;
      
      int gix = ix + rank_x*NMESH_X_LOCAL;
      REAL xpos = (gix+0.5)*this_run.delta_x;

      struct fluid_mesh *tgt;
      tgt = &MESH(ix,iy,iz);
      REAL wmol = WMOL(tgt->chem);
      REAL gamma_l = gamma_total(tgt, &this_run);

      rad[gix] = xpos;

      felec[gix]   = tgt->chem.felec;
      fracHI[gix]  = tgt->chem.fHI;
      fracHII[gix] = tgt->chem.fHII;
      tmpr[gix]    = tgt->uene*this_run.uenetok*wmol;
      nh[gix]      = tgt->dens*this_run.denstonh;

      kene[gix] = 0.5*NORML2(tgt->momx,tgt->momy,tgt->momz)/tgt->dens;
      pene[gix] = -0.5*tgt->pot*tgt->dens;
      tene[gix] = tgt->uene*tgt->dens;
      etrp_func[gix] = (gamma_l-1.0)*tgt->uene*tgt->dens/pow(tgt->dens,gamma_l);
      cs[gix]   = sqrt((gamma_l-1.0)*gamma_l*tgt->uene);
      pres[gix] = tgt->uene*(gamma_l-1.0)*tgt->dens;

      velx[gix] = tgt->momx/tgt->dens;
      vely[gix] = tgt->momy/tgt->dens;
      velz[gix] = tgt->momz/tgt->dens;
      velc[gix] = sqrt(NORML2(tgt->momx,tgt->momy,tgt->momz))/tgt->dens;
      gamma[gix] = gamma_l;

      nin[gix]  = 1.0;
      
#ifdef __HELIUM__
      fracHeI[gix]   = tgt->chem.fHeI;
      fracHeII[gix]  = tgt->chem.fHeII;
      fracHeIII[gix] = tgt->chem.fHeIII;
#endif
#ifdef __HYDROGEN_MOL__
      fracH2I[gix]  = tgt->chem.fH2I;
      fracH2II[gix] = tgt->chem.fH2II;
      fracHM[gix]   = tgt->chem.fHM;
#endif
		
    }
  }
  
  printf("# rad felec fracHI fracHII tmpr nH velc velx vely velz kene pene tene etrp_func fracHeI fracHeII fracHeIII fracH2I fracH2II fracHM cs pres gamma\n");
  
  for(int ibin=0;ibin<NRADBIN;ibin++) {
    
    printf("%14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n",
           rad[ibin], felec[ibin], fracHI[ibin], fracHII[ibin],
           tmpr[ibin], nh[ibin], velc[ibin], velx[ibin], vely[ibin],velz[ibin],
           kene[ibin], pene[ibin], tene[ibin], etrp_func[ibin],
           fracHeI[ibin], fracHeII[ibin], fracHeIII[ibin],
           fracH2I[ibin], fracH2II[ibin], fracHM[ibin],
	   cs[ibin],pres[ibin],gamma[ibin]);

  }
}
