#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "run_param.h"
#include "fluid.h"
#include "source.h"
#include "prototype.h"

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

#ifndef TINY
#define TINY (1.0e-31)
#endif


#define CORE_NUM_DENS (3.2e+0)
//#define CORE_NUM_DENS (1.0e+4)

#define INIT_TMPR (1.0e+2) 

void make_directory(char*);

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

  /* density and temperature of the medium */
  // nH = 1.0e+4;
  nH   = CORE_NUM_DENS;
  tmpr = INIT_TMPR;

  //this_run.lunit = 1.5e-2*mpc;
  //  this_run.lunit = 6.6e-3*mpc;
  //  this_run.lunit = 4.0e-6*mpc;
 // this_run.lunit = 0.8e-3*mpc;  //corner
  //this_run.lunit = 0.2e-3*mpc;  //corner
  //this_run.lunit = 1.6e-3*mpc;     //center
   this_run.lunit = 0.4e-3*mpc;  //center
  //this_run.lunit = 1.5e-6*mpc; //whalen (3/4)

  this_run.munit = nH*CUBE(this_run.lunit)*mproton/XHYDROGEN;
  //this_run.tunit = 3.86e15; /* recombination timescale = ne*alpha_B */  
  this_run.tunit = 1.0e8*year; /* 100Myr */
  //this_run.tunit = 1.0e5*year; /* 100Kyr */ //whalen

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
    
  ioneq_chem.fHI   = 0.9999;
  ioneq_chem.fHII  = 1.0-0.9999;
  ioneq_chem.felec = 1.0-0.9999;
  
  ioneq_chem.GammaHI = 0.0;
#ifdef __HELIUM__
  ioneq_chem.GammaHeI = 0.0;
  ioneq_chem.GammaHeII = 0.0;
#endif
  
  tmpr=INIT_TMPR;

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
  this_run.nsrc = 1;

  float xpos,ypos,zpos;
  
  xpos = 0.0; 
  ypos = 0.0; 
  zpos = 0.0; 
  /*
  xpos = 0.0; 
  ypos = 0.5; 
  zpos = 0.5; 
  */
  xpos = 0.5;
  ypos = 0.5;
  zpos = 0.5;
  
  /* offset */
  /* xpos = (0.5-0.1)*(1.0/NMESH_X_TOTAL); */
  /* ypos = (0.5)*(1.0/NMESH_Y_TOTAL); */
  /* zpos = (0.5+0.1)*(1.0/NMESH_Z_TOTAL); */
  
  src = (struct radiation_src *) malloc(sizeof(struct radiation_src)*this_run.nsrc);
  src[0].xpos = xpos;
  src[0].ypos = ypos;
  src[0].zpos = zpos;
  src[0].type = 0; /* black body */ 
  src[0].param = 1.0e5; /* T_bb= 100000 K */
  //  src[0].photon_rate = 5.0e48;
  setup_photon_rate(&this_run.freq, &src[0], 1.0e50);
  //setup_photon_rate(&this_run.freq, &src[0], 1.0e48);
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

srand(2);

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
	      struct prim_chem ioneq_chem_tgt;


	      float ix_pos = this_run.xmin_local + (ix+0.5)*this_run.delta_x;
	      float iy_pos = this_run.ymin_local + (iy+0.5)*this_run.delta_y;
	      float iz_pos = this_run.zmin_local + (iz+0.5)*this_run.delta_z;

	      float ix_pos0 = this_run.xmin_local + (ix)*this_run.delta_x;
	      float iy_pos0 = this_run.ymin_local + (iy)*this_run.delta_y;
	      float iz_pos0 = this_run.zmin_local + (iz)*this_run.delta_z;

	      float radius = sqrt(SQR(ix_pos-xpos)+SQR(iy_pos-ypos)+SQR(iz_pos-zpos));

	      float edge_min = sqrt(SQR(ix_pos0-xpos)+SQR(iy_pos0-ypos)+SQR(iz_pos0-zpos));
	      float edge_max = sqrt( SQR((ix_pos0+this_run.delta_x)-xpos) +
				     SQR((iy_pos0+this_run.delta_y)-ypos) +
				     SQR((iz_pos0+this_run.delta_z)-zpos) );

	      float clump_rad; 	      
	      //clump_rad = 0.11;   //corner
	      clump_rad = 0.22;   //corner
	      // clump_rad = 0.44;   //corner
	      //clump_rad = 0.055;  //center
	      //clump_rad = 0.13333;   //whalen (3/4)

	      /* 1.0 - 1.0 : fix amplitude */
	      //float amplitude = 1.0e0;
	      /* 0.99 - 1.01 : one percent random amplitude */
	      //float amplitude = (2.0*(float)rand()/(float)RAND_MAX - 1.0e0)/100.0 + 1.0e0;
	      /* 0.9 - 1.1 : ten percent random amplitude */
	      //float amplitude = (2.0*(float)rand()/(float)RAND_MAX - 1.0e0)/10.0 + 1.0e0;
	      /* 0.5 - 1.5 : fifty percent random amplitude */
	      float amplitude = (2.0*(float)rand()/(float)RAND_MAX - 1.0e0)/2.0 + 1.0e0;
	      /* 1.0 - 2.0 : one hundred percent random amplitude */
	      //float amplitude = (2.0*(float)rand()/(float)RAND_MAX)/2.0 + 1.0e0;
	    

	      tgt = &MESH(ix,iy,iz);

	      if(radius <= clump_rad*0.5) amplitude = 1.0;
	      //	      if(radius <= clump_rad*0.625) amplitude = 1.0; //whalen	      
	      nH = CORE_NUM_DENS * amplitude;
	      
	      if(radius <= clump_rad) {
		tgt->dens = nH/this_run.denstonh;
	      } else {
		nH *= SQR(clump_rad/radius); 
		tgt->dens = nH/this_run.denstonh;
	      }
	      
	      float wmol = WMOL(ioneq_chem);
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
 

  /* tau check */
#if 0
  int imesh;
  
  static float optical_depth_HI[NMESH_LOCAL];
  static float depth_HI[NMESH_LOCAL];
  
  for(imesh=0;imesh<NMESH_LOCAL;imesh++) {
  
    optical_depth_HI[imesh] = 0.0e0;
    depth_HI[imesh] = 0.0e0;
    
    
    float nH, nHI, nHe, nHeI, nHeII;
    
    int x_step, y_step, z_step;
    int x_next, y_next, z_next;
    
    float dx, dy, dz;
    float dl, dxy;
    
    float sin_theta, cos_theta;
    float sin_phi,   cos_phi;
    
    float xovr, yovr, zovr;
    float rmin, rx, ry, rz;
    
    int   ix_end, iy_end, iz_end;
    int   ix_cur, iy_cur, iz_cur;
    
    /* current position on the ray segment in the local coordinate */
    float x_cur, y_cur, z_cur; 
    
    optical_depth_HI[imesh] = 0.0;
    depth_HI[imesh] = 0.0;

    ix_end = (int)imesh/(NMESH_Y_LOCAL*NMESH_Z_LOCAL);
    iy_end = (int)(imesh - ix_end*NMESH_Y_LOCAL*NMESH_Z_LOCAL)/NMESH_X_LOCAL;
    iz_end = (int)(imesh - ix_end*NMESH_Y_LOCAL*NMESH_Z_LOCAL - iy_end*NMESH_X_LOCAL);

    ix_end = MAX(MIN(ix_end, NMESH_X_LOCAL-1),0);
    iy_end = MAX(MIN(iy_end, NMESH_Y_LOCAL-1),0);
    iz_end = MAX(MIN(iz_end, NMESH_Z_LOCAL-1),0);
    
    float ix_pos = (ix_end+0.5)*this_run.delta_x;
    float iy_pos = (iy_end+0.5)*this_run.delta_y;
    float iz_pos = (iz_end+0.5)*this_run.delta_z;
    
    float radius = sqrt(SQR(ix_pos-xpos)+SQR(iy_pos-ypos)+SQR(iz_pos-zpos));
    
    ix_cur = 0;
    iy_cur = 0;
    iz_cur = 0;
    
    ix_cur = MAX(MIN(ix_cur, NMESH_X_LOCAL-1),0);
    iy_cur = MAX(MIN(iy_cur, NMESH_Y_LOCAL-1),0);
    iz_cur = MAX(MIN(iz_cur, NMESH_Z_LOCAL-1),0);
      
    dx = (float)(ix_end+0.5)/NMESH_X_LOCAL;
    dy = (float)(iy_end+0.5)/NMESH_Y_LOCAL;
    dz = (float)(iz_end+0.5)/NMESH_Z_LOCAL;
    
    dl = sqrt(SQR(dx)+SQR(dy)+SQR(dz)+TINY);
    dxy = sqrt(SQR(dx)+SQR(dy));
    
    cos_theta = dz/dl;
    sin_theta = dxy/dl;
    
    //    sin_theta = sqrt(1.0-SQR(cos_theta));
    //    dxy = dl*sin_theta;
    cos_phi = dx/(dxy+TINY);
    sin_phi = dy/(dxy+TINY);
    
    xovr = cos_phi*sin_theta;
    if(fabsf(xovr)<TINY) {
      xovr = (xovr >= 0.0 ? TINY : -TINY);
    }

    yovr = sin_phi*sin_theta;
    if(fabs(yovr)<TINY) {
      yovr = (yovr >= 0.0 ? TINY : -TINY);
    }

    zovr = cos_theta;
    if(fabs(zovr)<TINY) {
      zovr = (zovr >= 0.0 ? TINY : -TINY);
    }

    if(xovr > 0.e0) {
      x_step = 1; x_next = 1;
    }else{
      x_step = 0; x_next = -1;
    }

    if(yovr > 0.e0) {
      y_step = 1; y_next = 1;
    }else {
      y_step = 0; y_next = -1;
    }

    if(zovr > 0.e0) {
      z_step = 1; z_next = 1;
    }else{
      z_step = 0; z_next = -1;
    }

    /* local coodinate with respect to each parallel domain */
    x_cur = xpos;
    y_cur = ypos;
    z_cur = zpos;

    
    while((ix_cur != ix_end || iy_cur != iy_end || iz_cur != iz_end) &&
	  (0<=ix_cur && ix_cur<NMESH_X_LOCAL) &&
	  (0<=iy_cur && iy_cur<NMESH_Y_LOCAL) &&
	  (0<=iz_cur && iz_cur<NMESH_Z_LOCAL) ) {
      
      nH  = MESH(ix_cur, iy_cur, iz_cur).dens;
      nHI = nH*MESH(ix_cur, iy_cur, iz_cur).chem.fHI;
      
      dx = this_run.delta_x*(x_step+ix_cur) - x_cur;
      dy = this_run.delta_y*(y_step+iy_cur) - y_cur;
      dz = this_run.delta_z*(z_step+iz_cur) - z_cur;
      
      rx = dx/xovr;
      ry = dy/yovr;
      rz = dz/zovr;
      
      rmin = fminf(rx, fminf(ry, rz));
      //rmin = MIN(rx, MIN(ry, rz));
      if(rmin == rx) ix_cur += x_next;
      if(rmin == ry) iy_cur += y_next;
      if(rmin == rz) iz_cur += z_next;
      
      x_cur += rmin*xovr;
      y_cur += rmin*yovr;
      z_cur += rmin*zovr;
      
      optical_depth_HI[imesh] += rmin*nHI;
      depth_HI[imesh] += rmin;
    }

    float depth;  // mesh center
    float depth2; // cross mesh 


    /* convert to the global coordinates */
    x_cur += this_run.xmin_local;
    y_cur += this_run.ymin_local;
    z_cur += this_run.zmin_local;
    
    depth = sqrt(SQR((float)(ix_end+0.5)/NMESH_X_LOCAL-x_cur)+
		 SQR((float)(iy_end+0.5)/NMESH_Y_LOCAL-y_cur)+
		 SQR((float)(iz_end+0.5)/NMESH_Z_LOCAL-z_cur));
    
    depth2 = 2.0*depth;


    //    if( fabs(depth-radius) < sqrt(SQR(1.0/NMESH_X_LOCAL)+SQR(1.0/NMESH_Y_LOCAL)+SQR(1.0/NMESH_Z_LOCAL))*0.5 ) {
    if( fabs(depth-radius) < 0.5/NMESH_X_LOCAL ) {
  
    }
    

  
    nH  = MESH(ix_end, iy_end, iz_end).dens;
    nHI = nH*MESH(ix_end, iy_end, iz_end).chem.fHI;

    optical_depth_HI[imesh]  += depth*nHI;
    depth_HI[imesh]  += depth;
  }

  
  FILE *fp;
  
  fp = fopen("tau.txt", "w");

  for(imesh=0;imesh<NMESH_LOCAL;imesh++) {

    int ix = (int)imesh/(NMESH_Y_LOCAL*NMESH_Z_LOCAL);
    int iy = (int)(imesh - ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL)/NMESH_X_LOCAL;
    int iz = (int)(imesh - ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL - iy*NMESH_X_LOCAL);

    float ix_pos = (ix+0.5)*this_run.delta_x;
    float iy_pos = (iy+0.5)*this_run.delta_y;
    float iz_pos = (iz+0.5)*this_run.delta_z;
    
    float radius = sqrt(SQR(ix_pos-xpos)+SQR(iy_pos-ypos)+SQR(iz_pos-zpos));
    
    float nH  = MESH(ix, iy, iz).dens;
    float nHI = nH*MESH(ix, iy, iz).chem.fHI;

    fprintf(fp, "%14.6e %14.6e %14.6e %14.6e %14.6e\n",radius,optical_depth_HI[imesh],depth_HI[imesh],nH,nHI);


  }

  fclose(fp);
  /* tau check */
#endif
}
