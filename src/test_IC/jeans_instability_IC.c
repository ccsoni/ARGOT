#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "run_param.h"
#include "fluid.h"
#include "source.h"
#include "prototype.h"

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

void make_directory(char *);

int main(int argc, char **argv)
{
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <prefix>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  static struct run_param this_run;
  struct fluid_mesh *mesh;
  struct radiation_src *src;

  static char model_name[256],label[256],dir_name[256];

  double nH, tmpr;
    
  this_run.nmesh_x_total=NMESH_X_TOTAL;
  this_run.nmesh_y_total=NMESH_Y_TOTAL;
  this_run.nmesh_z_total=NMESH_Z_TOTAL;

  this_run.nmesh_x_local=NMESH_X_LOCAL;
  this_run.nmesh_y_local=NMESH_Y_LOCAL;
  this_run.nmesh_z_local=NMESH_Z_LOCAL;

  this_run.xmin=0.0;
  this_run.ymin=0.0;
  this_run.zmin=0.0;
  
  this_run.xmax=1.0;
  this_run.ymax=1.0;
  this_run.zmax=1.0;

  this_run.delta_x=(this_run.xmax-this_run.xmin)/(float)this_run.nmesh_x_total;
  this_run.delta_y=(this_run.ymax-this_run.ymin)/(float)this_run.nmesh_y_total;
  this_run.delta_z=(this_run.zmax-this_run.zmin)/(float)this_run.nmesh_z_total;

  mesh = (struct fluid_mesh *)
    malloc(sizeof(struct fluid_mesh)*NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL);
  
  float dens_0 = 1.5e7; // in g/cm3
  float pres_0 = 1.5e7; // in dyn/cm2
  float delta = 1.0e-3;

  this_run.lunit = 0.5*sqrt(PI*GAMMA_MONOATOMIC*pres_0/(gnewton*SQR(dens_0)));
  this_run.munit = dens_0*CUBE(this_run.lunit);
  this_run.tunit = 1.0/sqrtf(gnewton*dens_0);

  double pres_unit = this_run.munit/(this_run.lunit*SQR(this_run.tunit));

  this_run.denstonh = this_run.munit/CUBE(this_run.lunit)*XHYDROGEN/mproton;
  this_run.uenetok  = GAMM1_MONOATOMIC*mproton/kboltz*
    SQR(this_run.lunit)/SQR(this_run.tunit);
  this_run.anow = 1.0;
  this_run.znow = 0.0;

  this_run.output_indx = -1; // just for initial conditions
  this_run.ngrid_nu = NGRID_NU;
  this_run.nspecies = NSPECIES;
  this_run.nchannel = NCHANNEL;

  /* ionization state at the initial condition */
  struct prim_chem ioneq_chem;
  nH = dens_0*this_run.denstonh;
  tmpr = 1.0e4;
  calc_ioneq(&ioneq_chem, nH, tmpr, 0.0);
  ioneq_chem.GammaHI = 0.0;
#ifdef __HELIUM__
  ioneq_chem.GammaHeI = 0.0;
  ioneq_chem.GammaHeII = 0.0;
#endif

  printf("# initial fHI = %14.6e\n",ioneq_chem.fHI);

  int rank_x, rank_y, rank_z;
  float lambda_x, lambda_y, lambda_z;
  float kwave_x, kwave_y, kwave_z;

  lambda_x = 1.0;
  lambda_y = 1.0e10;
  lambda_z = 1.0e10;

  kwave_x = 2.0*PI/lambda_x;
  kwave_y = 2.0*PI/lambda_y;
  kwave_z = 2.0*PI/lambda_z;

  printf("output parameter in unit scale.\n");
  double c0 = sqrt(GAMMA_MONOATOMIC*pres_0/pres_unit);
  double omega = sqrt(SQR(c0)*SQR(kwave_x) -4.0*PI);

  printf("omega %14.6e\n",omega);
  printf("kinetic   factor T %14.6e\n",SQR(delta)*SQR(omega)/(8.0*SQR(kwave_x)));
  printf("thermal   factor U %14.6e\n",-SQR(delta)*SQR(c0)*0.125);
  printf("potential factor W %14.6e\n",-PI*SQR(delta)/(2.0*SQR(kwave_x)));
  
  this_run.nnode_x = NNODE_X;
  this_run.nnode_y = NNODE_Y;
  this_run.nnode_z = NNODE_Z;

  this_run.step = 0;
  this_run.tnow = 0.0;

  this_run.mpi_rank = 0;

  setup_freq_param(&this_run.freq);

  this_run.nsrc = 1;

  src = (struct radiation_src *) 
    malloc(sizeof(struct radiation_src)*this_run.nsrc);
  src[0].xpos = 0.4999;
  src[0].ypos = 0.4999;
  src[0].zpos = 0.4999;
  src[0].type = 0; /* black body */ 
  src[0].param = 1.0e5; /* T_bb= 100000 K */
  //  src[0].photon_rate = 5.0e48;
  setup_photon_rate(&this_run.freq, &src[0], 5.0e48);

  sprintf(model_name, "%s", argv[1]);
  sprintf(dir_name, "%s-init", argv[1]);
  make_directory(dir_name);
  sprintf(label,"%s-init/%s-init",model_name,model_name);

  output_src(src, &this_run, label);

  for(int rank_x=0;rank_x<NNODE_X;rank_x++) {
    float dx_domain = (this_run.xmax-this_run.xmin)/(float)NNODE_X;
    this_run.xmin_local = this_run.xmin + (float)rank_x*dx_domain;
    this_run.xmax_local = this_run.xmin_local + dx_domain;

    this_run.rank_x = rank_x;

    for(int rank_y=0;rank_y<NNODE_Y;rank_y++) {
      float dy_domain = (this_run.ymax-this_run.ymin)/(float)NNODE_Y;
      this_run.ymin_local = this_run.ymin + (float)rank_y*dy_domain;
      this_run.ymax_local = this_run.ymin_local + dy_domain;

      this_run.rank_y = rank_y;

      for(int rank_z=0;rank_z<NNODE_Z;rank_z++) {
	float dz_domain = (this_run.zmax-this_run.zmin)/(float)NNODE_Z;
	this_run.zmin_local = this_run.zmin + (float)rank_z*dz_domain;
	this_run.zmax_local = this_run.zmin_local + dz_domain;

	this_run.rank_z = rank_z;

	this_run.mpi_nproc = NNODE_X*NNODE_Y*NNODE_Z;
	this_run.mpi_rank = mpi_rank(rank_x,rank_y,rank_z);

	for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
	  float xpos = this_run.xmin_local + ((float)ix+0.5)*this_run.delta_x;

	  for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
	    float ypos = this_run.ymin_local + ((float)iy+0.5)*this_run.delta_y;

	    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	      float zpos = this_run.zmin_local + ((float)iz+0.5)*this_run.delta_z;
	      struct fluid_mesh *tgt;

	      tgt = &MESH(ix,iy,iz);

	      tgt->chem.fHI = 1.0;
	      tgt->chem.fHII = 0.0;
#ifdef __HELIUM__
	      tgt->chem.fHeI = 1.0;
	      tgt->chem.fHeII = 0.0;
	      tgt->chem.fHeIII = 0.0;
#endif

	      float cos_kr,pres,gamma;
	      cos_kr = cosf(kwave_x*xpos + kwave_y*ypos + kwave_z*zpos);
	      gamma = GAMMA_MONOATOMIC;
	      
	      tgt->dens = 1.0+delta*cos_kr;
	      pres = pres_0*(1.0+gamma*delta*cos_kr)/pres_unit;
	      tgt->eneg = pres/(gamma-1.0);
	      tgt->uene = tgt->eneg/tgt->dens;
	      tgt->etrp = (gamma-1.0)*tgt->eneg/powf(tgt->dens, gamma-1.0);

	      tgt->momx = tgt->momy = tgt->momz = 0.0;
	    }
	  }
	}

	output_mesh(mesh, &this_run, label);
	
      }
    }
  }

}
