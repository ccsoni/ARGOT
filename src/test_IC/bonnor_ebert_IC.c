#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "run_param.h"
#include "fluid.h"
#include "source.h"
#include "prototype.h"

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

#define __MATCHED_AMB__

struct BE_profile
{
  float phi;
  float rho;
};

void make_directory(char*);
int calc_BE_profile(struct BE_profile*, float, float, float);
double f1(double, double, double);
double f2(double, double, double);

int main(int argc, char **argv)
{

  if(argc != 2) {
    fprintf(stderr,"Usage: %s <prefix>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  static struct run_param this_run;

  static struct fluid_mesh mesh[NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL];
  struct radiation_src *src;

  static char model_name[256],label[256],dir_name[256];

  this_run.nmesh_x_total=NMESH_X_TOTAL;
  this_run.nmesh_y_total=NMESH_Y_TOTAL;
  this_run.nmesh_z_total=NMESH_Z_TOTAL;

  this_run.nmesh_x_local=NMESH_X_LOCAL;
  this_run.nmesh_y_local=NMESH_Y_LOCAL;
  this_run.nmesh_z_local=NMESH_Z_LOCAL;

  this_run.xmin = -15.0;
  this_run.ymin = -15.0;
  this_run.zmin = -15.0;
  
  this_run.xmax = 15.0;
  this_run.ymax = 15.0;
  this_run.zmax = 15.0;

  this_run.delta_x=(this_run.xmax-this_run.xmin)/(float)this_run.nmesh_x_total;
  this_run.delta_y=(this_run.ymax-this_run.ymin)/(float)this_run.nmesh_y_total;
  this_run.delta_z=(this_run.zmax-this_run.zmin)/(float)this_run.nmesh_z_total;

  /* density and temperature of the medium */
  double nH, tmpr;
  double nH_in, tmpr_in;
  double nH_amb, tmpr_amb;
  
//  nH = 1.0e-2;
  nH = 1.0e+2;
  tmpr = 20.0;

  nH_in = nH;
  tmpr_in = tmpr;

  nH_amb = 1.0e0;
  tmpr_amb = 20.0;
  
  //double rho_c = 1.0e-17; // g cm^-3
  double rho_c = nH*mproton; 
  double cs = sqrt(kboltz*tmpr/mproton);
  double coeff = 1.0/sqrt(4.0*PI*gnewton*rho_c);

  printf("rho_c=%e cs=%e coeff=%e : cgs scale\n",
	 rho_c,cs,coeff);
  
  this_run.lunit = cs*coeff;
  this_run.munit = CUBE(cs)*coeff/gnewton;
  this_run.tunit = coeff;

  this_run.denstonh = this_run.munit/CUBE(this_run.lunit)*XHYDROGEN/mproton;
  this_run.uenetok = GAMM1_MONOATOMIC*mproton/kboltz*
    SQR(this_run.lunit)/SQR(this_run.tunit);
  this_run.anow = 1.0;
  this_run.znow = 0.0;

  this_run.output_indx = -1; // just for initial conditions
  this_run.ngrid_nu = NGRID_NU;
  this_run.nspecies = NSPECIES;
  this_run.nchannel = NCHANNEL;

  printf("lunit: %e cm %e mpc\n", this_run.lunit, this_run.lunit/mpc);
  printf("munit: %e g  %e sunmass\n", this_run.munit, this_run.munit/sunmass);
  printf("tunit: %e s  %e year\n",this_run.tunit, this_run.tunit/year);

  printf("denstonh %e  uenetok %e\n",this_run.denstonh, this_run.uenetok);
  
  /* ionization state at the initial condition */
  struct prim_chem ioneq_chem;
  calc_ioneq(&ioneq_chem, nH, tmpr, 0.0);
  ioneq_chem.GammaHI = 0.0;
#ifdef __HELIUM__
  ioneq_chem.GammaHeI = 0.0;
  ioneq_chem.GammaHeII = 0.0;
#endif


  struct BE_profile *BEprof;
  float rmin,rmax,dr;
  int prof_size;

  rmin = 1.0e-4; //non-zero
  rmax = 6.5;
  // rmax = 20.0;
  //rmax = 5.0;
  dr   = 0.0001;
  
  prof_size = rmax/dr;

  BEprof = (struct BE_profile *)malloc(sizeof(struct BE_profile)*prof_size);
  
  calc_BE_profile(BEprof, rmin, rmax, dr);
  /*
  printf("rhoc rhoext rate %e %e %e\n",BEprof[0].rho, BEprof[prof_size-1].rho,
	 BEprof[0].rho/BEprof[prof_size-1].rho);
  */
  printf("# initial fHI = %14.6e\n",ioneq_chem.fHI);

  int rank_x, rank_y, rank_z;
  this_run.nnode_x = NNODE_X;
  this_run.nnode_y = NNODE_Y;
  this_run.nnode_z = NNODE_Z;

  this_run.step = 0;
  this_run.tnow = 0.0;

  this_run.mpi_rank = 0;

  setup_freq_param(&this_run.freq);

#if 1
  float xcent = 0.0;
  float ycent = 0.0;
  float zcent = 0.0;
#else
  float xcent = 0.0+this_run.delta_x*0.5;
  float ycent = 0.0+this_run.delta_y*0.5;
  float zcent = 0.0+this_run.delta_z*0.5;
#endif
  
#if 1
  this_run.nsrc = 1;

  src = (struct radiation_src *) malloc(sizeof(struct radiation_src)*this_run.nsrc);

  src[0].xpos = xcent;
  src[0].ypos = ycent;
  src[0].zpos = zcent;

  src[0].type = 0; /* black body */ 
  src[0].param = 1.0e5; /* T_bb= 100000 K */
  //  src[0].photon_rate = 5.0e48;
//  setup_photon_rate(&this_run.freq, &src[0], 1.0e45);
  setup_photon_rate(&this_run.freq, &src[0], 1.0e47);
  for(int inu=0;inu<NGRID_NU;inu++) {
    printf("%14.6e %14.6e\n", this_run.freq.nu[inu], src[0].photon_rate[inu]);
  }
#else
  this_run.nsrc = 16;
  srand(2);

  src = (struct radiation_src *) malloc(sizeof(struct radiation_src)*this_run.nsrc);
  int isrc;
  for(isrc=0;isrc<this_run.nsrc;isrc++) {
    src[isrc].xpos = (float)rand()/(float)RAND_MAX;
    src[isrc].ypos = (float)rand()/(float)RAND_MAX;
    src[isrc].zpos = (float)rand()/(float)RAND_MAX;
    src[isrc].type = 0; 
    src[isrc].param = 5.0e3;
    setup_photon_rate(&this_run.freq, &src[isrc], 5.0e48);
  }
#endif

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
	      tgt = &MESH(ix,iy,iz);
	      
	      float rx,ry,rz,rad,xi; 

	      rx = this_run.xmin_local  + ((float)ix+0.5)*this_run.delta_x;
	      ry = this_run.ymin_local  + ((float)iy+0.5)*this_run.delta_y;
	      rz = this_run.zmin_local  + ((float)iz+0.5)*this_run.delta_z;

	      rad = sqrtf(SQR(rx-xcent)+SQR(ry-ycent)+SQR(rz-zcent));
	      
	      double nH_BE,pres,cs;
	      
	      if(rad<=rmax) {
		int index = (int)(rad/dr);
		nH_BE = nH_in*BEprof[index].rho;
		tmpr = tmpr_in;

		tgt->momx = 0.0;
		tgt->momy = 0.0;
		tgt->momz = 0.0;
		
	      } else {
#ifdef __MATCHED_AMB__
		nH_BE = nH_in*BEprof[prof_size-1].rho;
		tmpr = tmpr_in;
#else
		nH_BE = nH_amb;
		tmpr = tmpr_amb;
#endif
		tgt->momx = 0.0;
		tgt->momy = 0.0;
		tgt->momz = 0.0;
	      }

	      tgt->dens = nH_BE/this_run.denstonh;
	      tgt->uene = tmpr/(this_run.uenetok*WMOL(ioneq_chem));
	      //	      tgt->eneg = tgt->dens*tgt->uene;
	      tgt->eneg = tgt->dens*tgt->uene +
		0.5*NORML2(tgt->momx,tgt->momy,tgt->momz)/tgt->dens;
	      tgt->chem = ioneq_chem;
	      tgt->prev_chem = ioneq_chem;
	 
	    }
	  }
	}

	output_mesh(mesh, &this_run, label);

      }
    }
  }

  free(BEprof);

  printf("# initial heat capacity ratio : %14.6e\n", gamma_total(&mesh[0], &this_run));

}



int calc_BE_profile(struct BE_profile *BEprof, 
		    float ximin, float ximax, float dxi)
{
  double phi,phi_dot;
  double xi0;
  
  double k1[2],k2[2],k3[2],k4[2];
  
  xi0=ximin;
  
  double c0,c1,c2,c3;
  c0 = 1.0/6.0;                                    // 1/6
  c1 = 1.0/(5.0 * 4.0*3.0*2.0);                    // 1/(5*4!)
  c2 = 8.0/(12.0 * 6.0*5.0*4.0*3.0*2.0);           // 8/(12*6!)
  c3 = 122.0/(81.0 * 8.0*7.0*6.0*5.0*4.0*3.0*2.0); // 122/(81*8!)  

  phi = c0 * SQR(xi0)
    - c1 * SQR(xi0)*SQR(xi0)
    + c2 * SQR(xi0)*SQR(xi0)*SQR(xi0)
    - c3 * SQR(xi0)*SQR(xi0)*SQR(xi0)*SQR(xi0);
  
  phi_dot = 2.0*c0 * xi0
    - 4.0*c1 * xi0*SQR(xi0)
    + 6.0*c2 * xi0*SQR(xi0)*SQR(xi0)
    - 8.0*c3 * xi0*SQR(xi0)*SQR(xi0)*SQR(xi0);
     
  int index = 0;
  for(double xi=xi0;xi<ximax;xi+=dxi) {
   
    k1[0]=dxi*f1(xi,phi,phi_dot);
    k1[1]=dxi*f2(xi,phi,phi_dot);
    
    k2[0]=dxi*f1(xi+dxi/2.0,phi+k1[0]/2.0,phi_dot+k1[1]/2.0);
    k2[1]=dxi*f2(xi+dxi/2.0,phi+k1[0]/2.0,phi_dot+k1[1]/2.0);
    
    k3[0]=dxi*f1(xi+dxi/2.0,phi+k2[0]/2.0,phi_dot+k2[1]/2.0);
    k3[1]=dxi*f2(xi+dxi/2.0,phi+k2[0]/2.0,phi_dot+k2[1]/2.0);
    
    k4[0]=dxi*f1(xi+dxi,phi+k3[0],phi_dot+k3[1]);
    k4[1]=dxi*f2(xi+dxi,phi+k3[0],phi_dot+k3[1]);
    
    phi=phi+(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])/6.0;
    phi_dot=phi_dot+(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])/6.0;

    BEprof[index].phi = phi;
    BEprof[index].rho = exp(-phi);
    index++;

    if(phi<0) break;
  }

  return index;
}

double f1(double xi,double phi,double phi_dot)
{
  return phi_dot;
}

double f2(double xi,double phi,double phi_dot)
{
  return exp(-phi)-2.0/xi*phi_dot;
}


