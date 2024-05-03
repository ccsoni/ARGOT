/*
Cosmological Radiative Transfer Comparison Project
Test 4
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "run_param.h"
#include "fluid.h"
#include "source.h"
#include "prototype.h"

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

#define POINT_SRC_NUM (16)

struct source_mesh{
  int x;
  int y;
  int z;
  double photon;
};


void make_directory(char*);

int main(int argc, char **argv)
{

  if(argc != 4) {
    fprintf(stderr,"Usage: %s <prefix> <density file> <source file>\n", argv[0]);
    fprintf(stderr,"get file at this URL\n");
    fprintf(stderr,"http://www.cita.utoronto.ca/~iliev/rtwiki/lib/exe/fetch.php?id=tests1-4&cache=cache&media=rt:t4:density.bin\n");
    fprintf(stderr,"http://www.cita.utoronto.ca/~iliev/rtwiki/lib/exe/fetch.php?id=tests1-4&cache=cache&media=rt:t4:sources.dat\n");
    exit(EXIT_FAILURE);
  }

  if(NMESH_X_TOTAL!=128 || NMESH_Y_TOTAL!=128 || NMESH_Z_TOTAL!=128) {
    printf("This test require each mesh size = 128. (%d,%d,%d) \n",
	   NMESH_X_TOTAL,NMESH_Y_TOTAL,NMESH_Z_TOTAL);
    exit(EXIT_FAILURE);
  }
  
  if(NSOURCE_MAX < POINT_SRC_NUM) {
    printf("require NSOURCE_MAX < POINT_SRC_NUM . NSOURCE_MAX %d  POINT_SRC_NUM %d\n",
	   NSOURCE_MAX,POINT_SRC_NUM);
    exit(EXIT_FAILURE);
  }

  FILE *fp;

  /* read density binary file */
  float redshift,dummy;
  float *rho;  
  rho = (float*)malloc(sizeof(float)*NMESH_X_TOTAL*NMESH_Y_TOTAL*NMESH_Z_TOTAL);

  fp = fopen(argv[2], "rb");
  if(fp==NULL) {
    printf("Cannot open read file !\n");
    exit(EXIT_FAILURE);
  }

  fread(&redshift, sizeof(float), 1, fp);
  fread(&dummy, sizeof(float), 1, fp);
  fread(rho, sizeof(float), NMESH_X_TOTAL*NMESH_Y_TOTAL*NMESH_Z_TOTAL, fp);
  
  fclose(fp);
  
  /* read sources text file */
  static struct source_mesh source[POINT_SRC_NUM];

  fp = fopen(argv[3], "r");
  if(fp==NULL) {
    printf("Cannot open read file !\n");
    exit(EXIT_FAILURE);
  }
  
  int line=0;
  int ret;
  while(( ret = fscanf( fp , " %d %d %d %lf" , 
			&source[line].x,&source[line].y,&source[line].z,&source[line].photon ))
	!= EOF ) {
   
    source[line].photon *= 1.0e+52;
    line++;
  }
  
  fclose(fp);

  printf("redshift %16.4e\n",redshift);
  for(int pn=0; pn<POINT_SRC_NUM; pn++)
    printf("%3d %3d %3d %16.4e\n",source[pn].x,source[pn].y,source[pn].z,
	   source[pn].photon);

  static struct run_param this_run;

  static struct fluid_mesh mesh[NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL];
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
     
  this_run.cosm.omega_m = 0.27;
  this_run.cosm.omega_v = 0.73;
  this_run.cosm.omega_b = 0.043;
  this_run.cosm.hubble  = 0.7;

  this_run.znow = redshift;
  this_run.anow = 1.0/(1.0+this_run.znow);
  this_run.tnow = ztotime(redshift,this_run.cosm);

  /* density and temperature of the medium */
  nH = 1.0e-3;
  tmpr = 1.0e2;

  this_run.lunit = 0.5/this_run.cosm.hubble*mpc;
  this_run.munit = nH*CUBE(this_run.lunit)*mproton/XHYDROGEN;
  // this_run.tunit = 1.0e8*year/this_run.cosm.hubble; /* 100 Myr */  
  this_run.tunit = 3.085629e+17/this_run.cosm.hubble; /* cosmological unit time */

  this_run.denstonh = this_run.munit/CUBE(this_run.lunit)*XHYDROGEN/mproton;
  this_run.uenetok = GAMM1_MONOATOMIC*mproton/kboltz*
    SQR(this_run.lunit)/SQR(this_run.tunit);

  this_run.output_indx = -1; // just for initial conditions
  this_run.ngrid_nu = NGRID_NU;
  this_run.nspecies = NSPECIES;
  this_run.nchannel = NCHANNEL;

  /* ionization state at the initial condition */
  struct prim_chem ioneq_chem;
  calc_ioneq(&ioneq_chem, nH, tmpr, 0.0);
  ioneq_chem.GammaHI = 0.0;
#ifdef __HELIUM__
  ioneq_chem.GammaHeI = 0.0;
  ioneq_chem.GammaHeII = 0.0;
#endif

  tmpr=1.0e2;

  printf("# initial fHI = %14.6e\n",ioneq_chem.fHI);

  int rank_x, rank_y, rank_z;

  this_run.nnode_x = NNODE_X;
  this_run.nnode_y = NNODE_Y;
  this_run.nnode_z = NNODE_Z;

  this_run.step = 0;

  this_run.mpi_rank = 0;

  setup_freq_param(&this_run.freq);

  this_run.nsrc = POINT_SRC_NUM;
  src = (struct radiation_src *) malloc(sizeof(struct radiation_src)*this_run.nsrc);

  int isrc;
  for(isrc=0;isrc<this_run.nsrc;isrc++) {
    src[isrc].xpos = (float)(source[isrc].x+0.7)/NMESH_X_TOTAL;
    src[isrc].ypos = (float)(source[isrc].y+0.5)/NMESH_Y_TOTAL;
    src[isrc].zpos = (float)(source[isrc].z+0.3)/NMESH_Z_TOTAL;
    src[isrc].type = 0; 
    src[isrc].param = 1.0e5;
    setup_photon_rate(&this_run.freq, &src[isrc], source[isrc].photon);
  }

  sprintf(model_name, "%s", argv[1]);
  sprintf(dir_name, "%s-init", argv[1]);
  make_directory(dir_name);
  sprintf(label,"%s-init/%s-init",model_name,model_name);

  output_src(src, &this_run, label);

  for(rank_x=0;rank_x<NNODE_X;rank_x++) {
    float dx_domain = (this_run.xmax-this_run.xmin)/(float)NNODE_X;
    this_run.xmin_local = this_run.xmin + (float)rank_x*dx_domain;
    this_run.xmax_local = this_run.xmin_local + dx_domain;

    this_run.rank_x = rank_x;
    for(rank_y=0;rank_y<NNODE_Y;rank_y++) {
      float dy_domain = (this_run.ymax-this_run.ymin)/(float)NNODE_Y;
      this_run.ymin_local = this_run.ymin + (float)rank_y*dy_domain;
      this_run.ymax_local = this_run.ymin_local + dy_domain;

    this_run.rank_y = rank_y;
      for(rank_z=0;rank_z<NNODE_Z;rank_z++) {
	float dz_domain = (this_run.zmax-this_run.zmin)/(float)NNODE_Z;
	this_run.zmin_local = this_run.zmin + (float)rank_z*dz_domain;
	this_run.zmax_local = this_run.zmin_local + dz_domain;

	this_run.rank_z = rank_z;

	this_run.mpi_nproc = NNODE_X*NNODE_Y*NNODE_Z;
	this_run.mpi_rank = mpi_rank(rank_x,rank_y,rank_z);

	int ix,iy,iz;
	for(ix=0;ix<NMESH_X_LOCAL;ix++) {
	  for(iy=0;iy<NMESH_Y_LOCAL;iy++) {
	    for(iz=0;iz<NMESH_Z_LOCAL;iz++) {
	      struct fluid_mesh *tgt;

	      int ixg,iyg,izg;
	      ixg = rank_x*NMESH_X_LOCAL + ix;
	      iyg = rank_y*NMESH_Y_LOCAL + iy;
	      izg = rank_z*NMESH_Z_LOCAL + iz;
	      
	      tgt = &MESH(ix,iy,iz);
	      
	      tgt->dens = rho[(izg)+NMESH_Z_TOTAL*((iyg)+NMESH_Y_TOTAL*(ixg))]
		/this_run.denstonh*CUBE(this_run.anow);
	      tgt->momx = 0.0;
	      tgt->momy = 0.0;
	      tgt->momz = 0.0;
	      tgt->uene = tmpr/(this_run.uenetok*WMOL(ioneq_chem));
	      tgt->eneg = tgt->dens*tgt->uene;
	      tgt->chem = ioneq_chem;
	      tgt->prev_chem = ioneq_chem;
	    }
	  }
	}

	output_mesh(mesh, &this_run, label);

      }
    }
  }

  printf("# initial heat capacity ratio : %14.6e\n", gamma_total(&mesh[0], &this_run));

  double time0 = ztotime(this_run.znow,this_run.cosm);

  printf("\n");
  printf("Myr     redshift        diff time     output time\n");
  printf("0(init) %14.6e    NULL \n",timetoz(time0, this_run.cosm));
  printf("0.05    %14.6e  %14.6e  %14.6e\n",
	 timetoz(time0 + 0.05e+6*year/this_run.tunit, this_run.cosm),
	 0.05e+6*year/this_run.tunit, time0 + 0.05e+6*year/this_run.tunit);
  printf("0.1     %14.6e  %14.6e  %14.6e\n",
	 timetoz(time0 + 0.1e+6*year/this_run.tunit , this_run.cosm),
	 0.1e+6*year/this_run.tunit , time0 + 0.1e+6*year/this_run.tunit);
  printf("0.2     %14.6e  %14.6e  %14.6e\n",
	 timetoz(time0 + 0.2e+6*year/this_run.tunit , this_run.cosm),
	 0.2e+6*year/this_run.tunit, time0 + 0.2e+6*year/this_run.tunit);
  printf("0.3     %14.6e  %14.6e  %14.6e\n",
	 timetoz(time0 + 0.3e+6*year/this_run.tunit , this_run.cosm),
	 0.3e+6*year/this_run.tunit, time0 + 0.3e+6*year/this_run.tunit);
  printf("0.4     %14.6e  %14.6e  %14.6e\n",
	 timetoz(time0 + 0.4e+6*year/this_run.tunit , this_run.cosm),
	 0.4e+6*year/this_run.tunit, time0 + 0.4e+6*year/this_run.tunit);
  fflush(stdout);

  free(rho);
}
