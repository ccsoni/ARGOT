#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "run_param.h"
#include "fluid.h"
#include "source.h"
#include "prototype.h"

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

void make_directory(char*);

#define CLUMP_NUM (1) 
#define SMOOTH_NUM (1) /* -SMOOTH_NUM to SMOOTH_NUM */
#define SMOOTH_WEIGHT (5.0)

struct clump_param {
  float cx, cy, cz;  //clump center  
  float rad;         //clump radius
  float nH;               
  float tmpr;
  struct prim_chem ioneq_chem;
};

void make_clump(struct clump_param *clump)
{
  clump[0].cx = 0.5;  
  clump[0].cy = 0.5;
  clump[0].cz = 0.5;
  clump[0].rad= 0.3857;       //38.57 pc / 100 pc
  clump[0].nH = 14.0;
  clump[0].tmpr=1.0e2;
  
  calc_ioneq(&clump[0].ioneq_chem, clump[0].nH, clump[0].tmpr, 0.0);
  if(clump[0].ioneq_chem.fHI == 1.0e0)    clump[0].ioneq_chem.fHI   = 9.999999e-1;
  if(clump[0].ioneq_chem.fHII == 0.0e0)   clump[0].ioneq_chem.fHII  = 0.000001;
  if(clump[0].ioneq_chem.felec == 0.0e0)  clump[0].ioneq_chem.felec = 0.000001;
}

void overwrite_clump(struct fluid_mesh *tgt, struct clump_param clump, 
		     struct run_param this_run)
{
  tgt->dens = clump.nH / this_run.denstonh;
  tgt->uene = clump.tmpr/(this_run.uenetok*WMOL(clump.ioneq_chem));
  tgt->eneg = tgt->dens*tgt->uene;
  tgt->chem = clump.ioneq_chem;
  tgt->prev_chem = clump.ioneq_chem;
}


#define SMOOTH(ix,iy,iz) (smooth_mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

void smooth_region(struct fluid_mesh *mesh, struct run_param *this_run)
{
  struct fluid_mesh *smooth_mesh;
  smooth_mesh = (struct fluid_mesh*)malloc(sizeof(struct fluid_mesh)*NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL);

  for(int imesh=0;imesh<NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL;imesh++) {
    smooth_mesh[imesh] = mesh[imesh];
  }

#pragma omp parallel for schedule(auto)
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {

	struct fluid_mesh *tgt;
	tgt = &MESH(ix,iy,iz);

	double dens, uene, eneg;
	dens = uene = eneg = 0.0;
	
	for(int ixx=-SMOOTH_NUM; ixx<=SMOOTH_NUM; ixx++) {
	  int sx = ix+ixx;
	  if(sx<0)              sx = -sx-1;
	  if(sx>=NMESH_X_LOCAL) sx = NMESH_X_LOCAL-ixx;
	  
	  for(int iyy=-SMOOTH_NUM; iyy<=SMOOTH_NUM; iyy++) {
	    int sy = iy+iyy;
	    if(sy<0)              sy = -sy-1;
	    if(sy>=NMESH_Y_LOCAL) sy = NMESH_Y_LOCAL-iyy;
	    
	    for(int izz=-SMOOTH_NUM; izz<=SMOOTH_NUM; izz++) {
	      int sz = iz+izz;
	      if(sz<0)              sz = -sz-1;
	      if(sz>=NMESH_Z_LOCAL) sz = NMESH_Z_LOCAL-izz;

	      dens += SMOOTH(sx,sy,sz).dens;
	      uene += SMOOTH(sx,sy,sz).uene;
	      eneg += SMOOTH(sx,sy,sz).eneg;
	      
	    }
	  }
	}

	/* weight */
	float weight, vol;
	weight = SMOOTH_WEIGHT - 1.0;
	dens += weight*SMOOTH(ix,iy,iz).dens;
	uene += weight*SMOOTH(ix,iy,iz).uene;
	eneg += weight*SMOOTH(ix,iy,iz).eneg;

	vol = weight+CUBE(2.0*SMOOTH_NUM+1.0);
	dens /= vol;
	uene /= vol;
	eneg /= vol;

	tgt->dens = dens;
	tgt->uene = uene;
	tgt->eneg = eneg;
	      
      }
    }
  }
  
  free(smooth_mesh);
}


int main(int argc, char **argv)
{

  if(argc != 2) {
    fprintf(stderr,"Usage: %s <prefix>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  
  static struct run_param this_run;

  static struct fluid_mesh mesh[NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL];
  //  static struct radiation_src src[NSOURCE_MAX];
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

  this_run.helium_flag = 0;
  this_run.hydrogen_mol_flag = 0;
  this_run.cosmology_flag = 0;
#ifdef __HELIUM__
  this_run.helium_flag = 1;
#endif
#ifdef __HYDROGEN_MOL__
  this_run.hydrogen_mol_flag = 1;
#endif
#ifdef __COSMOLOGICAL__
  this_run.cosmology_flag = 1;
#endif

  
  /* density and temperature of the medium */
  nH = 1.0e-2;
  tmpr = 1.0e2;

  //this_run.lunit = 1.5e-2*mpc;
  this_run.lunit = 1.0e-4*mpc; // 100pc
  this_run.munit = 8.3e+4*sunmass;
  this_run.tunit = sqrt(CUBE(this_run.lunit)/(gnewton*this_run.munit)); 

  printf("lunit : %e cm,  %e pc\n",this_run.lunit,this_run.lunit/(mpc*1.0e-6));
  printf("munit : %e g,   %e sunmass\n",this_run.munit,this_run.munit/sunmass);
  printf("tunit : %e sec, %e yr\n",this_run.tunit, this_run.tunit/year);
  
  this_run.denstonh = this_run.munit/CUBE(this_run.lunit)*XHYDROGEN/mproton;
  this_run.uenetok = GAMM1_MONOATOMIC*mproton/kboltz*
    SQR(this_run.lunit)/SQR(this_run.tunit);
  this_run.anow = 1.0;
  this_run.znow = 0.0;
  
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
#ifdef __HYDROGEN_MOL__
  ioneq_chem.GammaHM = 0.0;
  ioneq_chem.GammaH2I_I = 0.0;
  ioneq_chem.GammaH2I_II = 0.0;
  ioneq_chem.GammaH2II_I = 0.0;
  ioneq_chem.GammaH2II_II = 0.0;
#endif
  
  tmpr=1.4e4;

  printf("# initial fHI = %14.6e\n",ioneq_chem.fHI);
  
  int rank_x, rank_y, rank_z;
  
  this_run.nnode_x = NNODE_X;
  this_run.nnode_y = NNODE_Y;
  this_run.nnode_z = NNODE_Z;
  
  this_run.step = 0;
  this_run.tnow = 0.0;

  this_run.mpi_rank = 0;
  
  setup_freq_param(&this_run.freq);
  
  this_run.nsrc = 0;
  //this_run.nsrc = 1;

  if(this_run.nsrc==0) {
    /* dummy source */
    src = (struct radiation_src *) malloc(sizeof(struct radiation_src)*1);
  } else {
    src = (struct radiation_src *) malloc(sizeof(struct radiation_src)*this_run.nsrc);
  }
    
  src[0].xpos = 0.5;
  src[0].ypos = 0.5;
  src[0].zpos = 0.5;
    
  src[0].type = 0; /* black body */ 
  src[0].param = 1.0e5; /* T_bb= 100000 K */
  //  src[0].photon_rate = 5.0e48;
  setup_photon_rate(&this_run.freq, &src[0], 5.0e48);

  for(int inu=0;inu<NGRID_NU;inu++) {
    printf("%14.6e %14.6e\n", this_run.freq.nu[inu], src[0].photon_rate[inu]);
  }
    
  sprintf(model_name, "%s", argv[1]);
  sprintf(dir_name, "%s-init", argv[1]);
  make_directory(dir_name);
  sprintf(label,"%s-init/%s-init",model_name,model_name);
  
  output_src(src, &this_run, label);
  
  static struct clump_param clump[CLUMP_NUM]; 
  make_clump(clump);
    
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

	for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
	  for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
	    for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
	      struct fluid_mesh *tgt;

	      tgt = &MESH(ix,iy,iz);

	      tgt->dens = nH/this_run.denstonh;
	      tgt->momx = 0.0;
	      tgt->momy = 0.0;
	      tgt->momz = 0.0;
	      tgt->uene = tmpr/(this_run.uenetok*WMOL(ioneq_chem));
	      tgt->eneg = tgt->dens*tgt->uene;
	      tgt->chem = ioneq_chem;
	      tgt->prev_chem = ioneq_chem;

	      float xpos = this_run.xmin_local + (ix+0.5)*this_run.delta_x; 
	      float ypos = this_run.ymin_local + (iy+0.5)*this_run.delta_y;
	      float zpos = this_run.zmin_local + (iz+0.5)*this_run.delta_z;
	      float rc;
	      
	      int cn;
	      for(cn=0; cn<CLUMP_NUM; cn++) {
		rc = sqrt(SQR(xpos-clump[cn].cx) + 
			  SQR(ypos-clump[cn].cy) + 
			  SQR(zpos-clump[cn].cz) );
		
		if(rc < clump[cn].rad)  overwrite_clump(tgt, clump[cn], this_run);
	      }

	    }
	  }
	}

	smooth_region(mesh, &this_run);
	output_mesh(mesh, &this_run, label);

      }
    }
  }

  printf("# initial heat capacity ratio : %14.6e\n", gamma_total(&mesh[0], &this_run));
  printf("### make dense clump. ###\n");

}
