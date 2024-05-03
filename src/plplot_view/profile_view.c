#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "plplot.h"

#include "run_param.h"
#include "radiation.h"


#define MESH_L(ix,iy,iz) mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))]
#define NMESH_MAX_TOTAL (MAX(NMESH_X_TOTAL,MIN(NMESH_Y_TOTAL,NMESH_Z_TOTAL)))

#define LN (13) //line number

/// 0:rad, 1:HI, 2:HII, 3:temp, 4:dens, 5:HeI, 6:HeII, 7:HeIII
/// 8:iene, 9:tene, 10:pene, 11:kene, 13:etrp
struct draw_data{
#ifdef __ALL_POINT_PLOT__
  PLFLT draw[LN][NMESH_LOCAL*NNODE];
#else
  PLFLT draw[LN][NMESH_MAX_TOTAL];
#endif
};


char *plot_title[LN] =
  {
    "HI,HII fraction",
    "temperature",
    "internal energy",
    "thermal energy",
    "potential energy",
    "kinetic energy",
    "density",
    "pres/dens^gamma"
  };


void input_mesh_single(struct fluid_mesh*, struct run_param*, char*);
void input_mesh_header(struct run_param*, char*);

float gamma_total(struct fluid_mesh *mesh, struct run_param *this_run)
{
  float gamma_H2_m1_inv; /* 1/(gamma_H2-1) */

  float x;
  float tmpr, wmol;
  struct prim_chem *chem;

  wmol = WMOL(mesh->chem);
  tmpr = mesh->uene*this_run->uenetok*wmol;

  chem = &(mesh->chem);

  x = 6100.0/tmpr;
  
  //gamma_H2_m1_inv = 0.5*(5.0+2.0*SQR(x)*expf(x)/SQR(expf(x)-1));
  gamma_H2_m1_inv = 0.5*(5.0+2.0*SQR(x)*exp(-x)/SQR(1.0-exp(-x)));
  
  float sum,denom;

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
  
  float gamma_m1;
  gamma_m1 = denom/sum;

  return (gamma_m1+1.0);
  //return (1.4);
}

void set_frame(long point_num, struct draw_data ddata, float range, 
	       int type, int pn, int log_set, float ymin, float ymax)
{
  /*
  static PLFLT  xmin = -3.0;
  static PLFLT  xmax =  1.0;
  */
  static PLFLT  xmin = 0.0;
  static PLFLT  xmax = 1.0;

  /* static PLFLT  ymin; */
  /* static PLFLT  ymax; */
  
  if(ymin==0.0 && ymax==0.0) {
    if(type==0) {
      ymin = -5.0; 
      ymax = 0.3;  
    }
    else if(type>=1) {
      ymax = FLT_MIN; 
      ymin = FLT_MAX;
      
      for(int i=0; i<point_num; i++) {
	if(ymax <  ddata.draw[pn][i])
	  ymax = ddata.draw[pn][i];
	if(ymin >  ddata.draw[pn][i])
	  ymin = ddata.draw[pn][i];
      }
      
      printf("%d \n",pn);
    }
  }

  ymin += -0.1;
  ymax += 0.1;
  
  printf("(xmin,xmax) %f %f \n",xmin,xmax);
  printf("(ymin,ymax) %f %f \n",ymin,ymax);

  plcol0(14); 
  plschr(7, 1.0);  
  plwidth(1.2);
  /*
  if(log_set==0)
    plenv(xmin, xmax, ymin, ymax, 2, 10);   // semilogarithm
  else
    plenv(xmin, xmax, ymin, ymax, 2, 30);   // logarithm
  */

  if(log_set==0)
    plenv(xmin, xmax, ymin, ymax, 2, 0);   // y linear
  else
    plenv(xmin, xmax, ymin, ymax, 2, 20);   // y logarithm


  //  pllab( "r/r#dbox", "", "" );
  pllab( "r/r#dclump", plot_title[type], "" );
}


void draw_line(long point_num, struct draw_data *ddata, int type, int pn)
{
#if 0
  if(type==0) {
    plline(point_num, ddata->draw[0], ddata->draw[1]);
    plline(point_num, ddata->draw[0], ddata->draw[2]);
  }
  else if(type==1) {
    plline(point_num, ddata->draw[0], ddata->draw[3]);
  }
  else if(type==2) {
    plline(point_num, ddata->draw[0], ddata->draw[4]);
  }
#ifdef __HELIUM__
  else if(type==3) {
    plline(point_num, ddata->draw[0], ddata->draw[5]);
    plline(point_num, ddata->draw[0], ddata->draw[6]);
    plline(point_num, ddata->draw[0], ddata->draw[7]);
  }
#endif //__HELIUM__
#endif //if
  
  if(type==0) {
    plline(point_num, ddata->draw[0], ddata->draw[1]);
    plline(point_num, ddata->draw[0], ddata->draw[2]);
  }
  else if(type>=1) {
    plline(point_num, ddata->draw[0], ddata->draw[pn]);
  }

}


int main(int argc, char **argv) 
{
  char pl_version[80];
  plgver(pl_version);
  fprintf(stdout, "PLplot library version: %s\n", pl_version);
  plparseopts(&argc, (const char**)argv, PL_PARSE_FULL | PL_PARSE_SKIP);
  
  static struct run_param this_run;

  static struct fluid_mesh mesh[NMESH_LOCAL];
  static struct radiation_src src[NSOURCE_MAX];

  static struct draw_data ddata;

  static char filename[256];

  float xcent, ycent, zcent, range;

  this_run.proc_file = stdout;

  /* if(argc != 6) { */
  /*   fprintf(stderr, */
  /*   	    "Usage: %s <input prefix> <profile center x y z> <range>\n", */
  /*   	    argv[0]); */
  /*   exit(EXIT_FAILURE); */
  /* } */

  if(argc != 8) {
    fprintf(stderr,
    	    "Usage: %s <input prefix> <profile center x y z> <ymin> <ymax> <unit clump radius>\n",
    	    argv[0]);
    fprintf(stderr,
    	    "<ymin> and <ymax> = 0 is auto scale. \n");
    exit(EXIT_FAILURE);
  }

  xcent = atof(argv[2]);
  ycent = atof(argv[3]);
  zcent = atof(argv[4]);
  range = atof(argv[7]);
  
  float ymin = atof(argv[5]);
  float ymax = atof(argv[6]);

  int rank_x, rank_y, rank_z;

  for(rank_x=0;rank_x<NNODE_X;rank_x++) {
    rank_y = rank_z = rank_x;
    
    FILE *fp;
    
    sprintf(filename, "%s_%03d_%03d_%03d", 
	    argv[1], rank_x, rank_y, rank_z);
    
    fp = fopen(filename,"r");
    input_mesh_single(mesh, &this_run, filename);
    fclose(fp);
    
    for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
      float xpos = this_run.xmin_local + (ix+0.5)*this_run.delta_x;
      int iy = ix;
      int iz = ix;
      
      float ypos = this_run.ymin_local + (iy+0.5)*this_run.delta_y;
      float zpos = this_run.zmin_local + (iz+0.5)*this_run.delta_z;
      
      float rad = sqrtf(SQR(xpos-xcent) + SQR(ypos-ycent) + SQR(zpos-zcent));

      struct fluid_mesh *tgt;
      tgt = &MESH_L(ix,iy,iz);
      float gamma;
      gamma = gamma_total(tgt, &this_run);

      float wmol = WMOL(tgt->chem);

      int mesh_id = ix + rank_x*NMESH_X_LOCAL;

      float kene = 0.5*NORML2(tgt->momx,tgt->momy,tgt->momz)/tgt->dens; 
      float pene = -0.5*tgt->pot*tgt->dens;
      float etrp_func = (gamma-1.0)*tgt->uene*tgt->dens/pow(tgt->dens,gamma);

      if(kene == 0.0) kene = 1.0e-30; 
      if(pene == 0.0) pene = 1.0e-30; 
      
      ddata.draw[0][mesh_id] = rad/range;
      // ddata.draw[0][mesh_id] = log10(rad/range);
      ddata.draw[1][mesh_id] = log10( tgt->chem.fHI );
      ddata.draw[2][mesh_id] = log10( tgt->chem.fHII );
      ddata.draw[3][mesh_id] = log10( tgt->uene*this_run.uenetok*wmol );
      ddata.draw[4][mesh_id] = log10( tgt->dens*this_run.denstonh );
      //ddata.draw[4][mesh_id] = tgt->dens*this_run.denstonh;
#ifdef __HELIUM__
      ddata.draw[5][mesh_id] = log10( tgt->chem.fHeI );
      ddata.draw[6][mesh_id] = log10( tgt->chem.fHeII );
      ddata.draw[7][mesh_id] = log10( 1.0e0 - (tgt->chem.fHeI + tgt->chem.fHeII) );
#endif //__HELIUM__
      
      ddata.draw[8][mesh_id]  = log10( tgt->uene);
      ddata.draw[9][mesh_id]  = log10( tgt->uene*tgt->dens);
      ddata.draw[10][mesh_id] = log10( pene );
      ddata.draw[11][mesh_id] = log10( kene );

      ddata.draw[12][mesh_id] = etrp_func;
    }
  }


  ///background color , default line color red
  plscolbg(255, 255, 255);
  //change col0 color
  plscol0(14, 0, 0, 0);
  plinit();
  

  int plot_num = 8; 
  int plot_value[8] = {0,3,8,9,10,11,4,12};
  int log_set[8] = {1,1,1,1,1,1,1,0};
  //int log_set[8] = {1,0,0,0,0,0,0,0};

  long point_num;
  point_num = NMESH_MAX_TOTAL;

  int pn;
  for(pn=0; pn<plot_num; pn++) {

    set_frame(point_num,ddata,range,pn,plot_value[pn],log_set[pn],ymin,ymax);
    
    draw_line(point_num, &ddata, pn, plot_value[pn]);
  }

  plend();
  return EXIT_SUCCESS;
}

