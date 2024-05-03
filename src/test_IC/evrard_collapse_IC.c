#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "run_param.h"
#include "fluid.h"
#include "source.h"
#include "prototype.h"

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

#define R_LIMIT (1.0)

void make_directory(char *);

int main(int argc, char **argv)
{
  if(argc != 2) {
    fprintf(stderr,"Usage: %s <prefix>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if(argc != 2) {
    fprintf(stderr,"Usage: %s <prefix>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  static struct run_param this_run;
  struct fluid_mesh *mesh;
  struct radiation_src *src;

  static char model_name[256],label[256],dir_name[256];

  this_run.nmesh_x_total=NMESH_X_TOTAL;
  this_run.nmesh_y_total=NMESH_Y_TOTAL;
  this_run.nmesh_z_total=NMESH_Z_TOTAL;

  this_run.nmesh_x_local=NMESH_X_LOCAL;
  this_run.nmesh_y_local=NMESH_Y_LOCAL;
  this_run.nmesh_z_local=NMESH_Z_LOCAL;

  this_run.xmin=-1.5;
  this_run.ymin=-1.5;
  this_run.zmin=-1.5;  

  this_run.xmax=1.5;
  this_run.ymax=1.5;
  this_run.zmax=1.5;

  this_run.delta_x=(this_run.xmax-this_run.xmin)/(float)this_run.nmesh_x_total;
  this_run.delta_y=(this_run.ymax-this_run.ymin)/(float)this_run.nmesh_y_total;
  this_run.delta_z=(this_run.zmax-this_run.zmin)/(float)this_run.nmesh_z_total;  

  this_run.lunit = 1.0e-5*mpc;
  this_run.munit = 1.0e5*sunmass;
  this_run.tunit = sqrt(CUBE(R_LIMIT*this_run.lunit)/(gnewton*this_run.munit));

  printf(" gas radius : %14.6e [Mpc] / %14.6e [cm] / %14.6e [unit length]\n",
	 R_LIMIT*this_run.lunit/mpc, R_LIMIT*this_run.lunit, R_LIMIT);

  printf(" gas mass   : %14.6e [M_solar] / %14.6e [g]\n",
	 this_run.munit/sunmass, this_run.munit);

  printf(" t_dyn      : %14.6e [Myr] / %14.6e [s]\n",
	 this_run.tunit/(1.0e6*year),
	 this_run.tunit);

  this_run.denstonh = this_run.munit/CUBE(this_run.lunit)*XHYDROGEN/mproton;
  this_run.uenetok  = 
    GAMM1_MONOATOMIC*mproton/kboltz*SQR(this_run.lunit/this_run.tunit);
  this_run.anow = 1.0;
  this_run.znow = 0.0;

  this_run.output_indx = -1; // just for a initial condition
  this_run.ngrid_nu = NGRID_NU;
  this_run.nspecies = NSPECIES;
  this_run.nchannel = NCHANNEL;

  float beta1 = 1.0e-2;
  float edge_dens = 1.0e0/(2.0*PI*CUBE(R_LIMIT)+beta1);
  float clump_tmpr = 0.05*this_run.uenetok;

  float ambient_dens = edge_dens*1.0e-3;
  float ambient_tmpr = clump_tmpr*edge_dens/ambient_dens;

  float nH, tmpr;
  
  /* temporary set the nominal number density and temperature */
  nH = 1.0e-3;
  tmpr = 1.0e4;
  
  /* ionization state at the initial condition */
  struct prim_chem ioneq_chem;
  calc_ioneq(&ioneq_chem, nH, tmpr, 0.0);
  ioneq_chem.GammaHI = 0.0;
#ifdef __HELIUM__
  ioneq_chem.GammaHeI = 0.0;
  ioneq_chem.GammaHeII = 0.0;
#endif

  this_run.nnode_x = NNODE_X;
  this_run.nnode_y = NNODE_Y;
  this_run.nnode_z = NNODE_Z;

  this_run.step = 0;
  this_run.tnow = 0.0;

  this_run.nsrc = 1;

  src = 
    (struct radiation_src *) malloc(sizeof(struct radiation_src)*this_run.nsrc);

  for(int isrc=0;isrc<this_run.nsrc;isrc++) {
    src[isrc].xpos = (float)rand()/(float)RAND_MAX;
    src[isrc].ypos = (float)rand()/(float)RAND_MAX;
    src[isrc].zpos = (float)rand()/(float)RAND_MAX;
    src[isrc].type = 0; 
    src[isrc].param = 5.0e3;
    setup_photon_rate(&this_run.freq, &src[isrc], 5.0e48);
  }
  
  sprintf(model_name, "%s", argv[1]);
  sprintf(dir_name, "%s-init", argv[1]);
  make_directory(dir_name);
  sprintf(label,"%s-init/%s-init",model_name,model_name);

  output_src(src, &this_run, label);

  mesh  = (struct fluid_mesh *) malloc(sizeof(struct fluid_mesh)*NMESH_LOCAL);

  float xcent, ycent, zcent;
  
#if 0
  xcent = 0.5*(this_run.xmax+this_run.xmin) - 0.5*this_run.delta_x;
  ycent = 0.5*(this_run.ymax+this_run.ymin) - 0.5*this_run.delta_y;
  zcent = 0.5*(this_run.zmax+this_run.zmin) - 0.5*this_run.delta_z;
#else  
  xcent = 0.5*(this_run.xmax+this_run.xmin);
  ycent = 0.5*(this_run.ymax+this_run.ymin);
  zcent = 0.5*(this_run.zmax+this_run.zmin);
#endif

  printf("center of the sphere : (X,Y,Z) = (%14.6e %14.6e %14.6e)\n",
	 xcent, ycent, zcent);

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
          for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
            for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
              struct fluid_mesh *tgt;
	      float x, y, z, rad;

	      x = this_run.xmin_local  + ((float)ix+0.5)*this_run.delta_x;
	      y = this_run.ymin_local  + ((float)iy+0.5)*this_run.delta_y;
	      z = this_run.zmin_local  + ((float)iz+0.5)*this_run.delta_z;

	      rad = sqrtf(SQR(x-xcent)+SQR(y-ycent)+SQR(z-zcent));

	      tgt = &MESH(ix,iy,iz);

	      if(rad <= R_LIMIT) {
		tgt->dens = 1.0/(2.0*PI*SQR(R_LIMIT)*rad + beta1);
		tgt->uene = 0.05;
	      }else{
		tgt->dens = 1.0/(2.0*PI*CUBE(R_LIMIT) + beta1)*1.0e-3;
		tgt->uene = 0.05*1.0e3;
	      }

	      float gamma = gamma_total(tgt, &this_run);

	      tgt->eneg = tgt->dens*tgt->uene;
	      tgt->etrp = (gamma-1.0)*tgt->eneg/powf(tgt->dens,gamma-1.0);

	      tgt->momx = tgt->momy = tgt->momz = 0.0;
	      tgt->chem.fHI = 1.0;
	      tgt->chem.fHII = 0.0;
	      tgt->chem.felec = 0.0;
	    }
	  }
	}

	output_mesh(mesh, &this_run, label);

      }
    }
  }
  
}
