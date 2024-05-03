#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdbool.h>

#define __CONTOUR__ 
#define CN (5)  // CONTOUR NUMBER
#define MN (8)  // map number
#define VS (4)  // vec spacing

#include "plplot.h"
#include "run_param.h"

#define MESH_L(ix,iy,iz) mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))]

void input_mesh_single(struct fluid_mesh*, struct run_param*, char*);
void input_mesh_header(struct run_param*, char*);

char *plot_title[MN] =
  {
    "HI fraction",
    "temperature",
    "internal energy",
    "thermal energy",
    "potential energy",
    "kinetic energy",
    "density",
    "pres/dens^gamma"
  };


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

void compile_map_x(struct fluid_mesh *mesh,
		   PLFLT **map[MN],
		   PLFLT **vec[2],
		   PLFLT *pos1, PLFLT *pos2, float slice_pos,
		   struct run_param *this_run)
{
  int ix, iy, iz;
  int jx, jy, jz;

  ix = (int)((slice_pos - this_run->xmin_local)/this_run->delta_x);

  for(iy=0;iy<NMESH_Y_LOCAL;iy++) {
    int jy = iy+this_run->rank_y*NMESH_Y_LOCAL;
    pos1[jy] = this_run->ymin_local + ((float)iy+0.5)*this_run->delta_y;
    for(iz=0;iz<NMESH_Z_LOCAL;iz++) {
      int jz = iz+this_run->rank_z*NMESH_Z_LOCAL;
      pos2[jz] = this_run->zmin_local + ((float)iz+0.5)*this_run->delta_z;
      
      float wmol = WMOL(MESH_L(ix,iy,iz).chem);
      
      //unit scale
      float kene = 0.5*NORML2(MESH_L(ix,iy,iz).momx,MESH_L(ix,iy,iz).momy,MESH_L(ix,iy,iz).momz)/MESH_L(ix,iy,iz).dens; 
      float pene = -0.5*MESH_L(ix,iy,iz).pot*MESH_L(ix,iy,iz).dens;
      float tene = MESH_L(ix,iy,iz).uene*MESH_L(ix,iy,iz).dens;

      if(kene == 0.0) kene = 1.0e-20; 
      if(pene == 0.0) pene = 1.0e-20; 

      map[0][jy][jz] = log10(MESH_L(ix,iy,iz).chem.fHI);
      map[1][jy][jz] = log10(MESH_L(ix,iy,iz).uene*this_run->uenetok*wmol);  //tmpr
      map[2][jy][jz] = log10(MESH_L(ix,iy,iz).uene);                         //uene //unit scale
      map[3][jy][jz] = log10(tene); 
      map[4][jy][jz] = log10(pene);
      map[5][jy][jz] = log10(kene);
      map[6][jy][jz] = log10(MESH_L(ix,iy,iz).dens*this_run->denstonh); //cgs
      
    }
  }

  for(iy=0;iy<NMESH_Y_LOCAL;iy+=VS) {
    int jy = (iy+this_run->rank_y*NMESH_Y_LOCAL)/VS;
    for(iz=0;iz<NMESH_Z_LOCAL;iz+=VS) {
      int jz = (iz+this_run->rank_z*NMESH_Z_LOCAL)/VS;

      vec[0][jy][jz] = MESH_L(ix,iy,iz).momy/MESH_L(ix,iy,iz).dens;
      vec[1][jy][jz] = MESH_L(ix,iy,iz).momz/MESH_L(ix,iy,iz).dens;

    }
  }

}

void compile_map_y(struct fluid_mesh *mesh, 
		   PLFLT **map[MN],
		   PLFLT **vec[2],
		   PLFLT *pos1, PLFLT *pos2, float slice_pos,
		   struct run_param *this_run)
{
  int ix, iy, iz;
  int jx, jy, jz;
  
  iy = (int)((slice_pos - this_run->ymin_local)/this_run->delta_y);
  
  for(ix=0;ix<NMESH_X_LOCAL;ix++) {
    int jx = ix+this_run->rank_x*NMESH_X_LOCAL;
    pos1[jx] = this_run->xmin_local + ((float)ix+0.5)*this_run->delta_x;
    for(iz=0;iz<NMESH_Z_LOCAL;iz++) {
      int jz = iz+this_run->rank_z*NMESH_Z_LOCAL;
      pos2[jz] = this_run->zmin_local + ((float)iz+0.5)*this_run->delta_z;
      
      float wmol = WMOL(MESH_L(ix,iy,iz).chem);
      
      //unit scale
      float kene = 0.5*NORML2(MESH_L(ix,iy,iz).momx,MESH_L(ix,iy,iz).momy,MESH_L(ix,iy,iz).momz)/MESH_L(ix,iy,iz).dens; 
      float pene = -0.5*MESH_L(ix,iy,iz).pot*MESH_L(ix,iy,iz).dens;
      float tene = MESH_L(ix,iy,iz).uene*MESH_L(ix,iy,iz).dens;

      if(kene == 0.0) kene = 1.0e-20; 
      if(pene == 0.0) pene = 1.0e-20; 

      map[0][jx][jz] = log10(MESH_L(ix,iy,iz).chem.fHI);
      map[1][jx][jz] = log10(MESH_L(ix,iy,iz).uene*this_run->uenetok*wmol);  //tmpr
      map[2][jx][jz] = log10(MESH_L(ix,iy,iz).uene);                         //uene //unit scale
      map[3][jx][jz] = log10(tene); 
      map[4][jx][jz] = log10(pene);
      map[5][jx][jz] = log10(kene);
      map[6][jx][jz] = log10(MESH_L(ix,iy,iz).dens*this_run->denstonh); //cgs
      
    }
  }

  for(ix=0;ix<NMESH_X_LOCAL;ix+=VS) {
    int jx = (ix+this_run->rank_x*NMESH_X_LOCAL)/VS;
    for(iz=0;iz<NMESH_Z_LOCAL;iz+=VS) {
      int jz = (iz+this_run->rank_z*NMESH_Z_LOCAL)/VS;
      
      vec[0][jx][jz] = MESH_L(ix,iy,iz).momx/MESH_L(ix,iy,iz).dens;
      vec[1][jx][jz] = MESH_L(ix,iy,iz).momz/MESH_L(ix,iy,iz).dens;
    }
  }

}

void compile_map_z(struct fluid_mesh *mesh, 
		   PLFLT **map[MN],
		   PLFLT **vec[2],
		   PLFLT *pos1, PLFLT *pos2, float slice_pos,
		   struct run_param *this_run)
{
  int ix, iy, iz;
  int jx, jy, jz;
  
  iz = (int)((slice_pos - this_run->zmin_local)/this_run->delta_z);
  
  for(ix=0;ix<NMESH_X_LOCAL;ix++) {
    int jx = ix+this_run->rank_x*NMESH_X_LOCAL;
    pos1[jx] = this_run->xmin_local + ((float)ix+0.5)*this_run->delta_x;
    for(iy=0;iy<NMESH_Y_LOCAL;iy++) {
      int jy = iy+this_run->rank_y*NMESH_Y_LOCAL;
      pos2[jy] = this_run->ymin_local + ((float)iy+0.5)*this_run->delta_y;
      
      struct fluid_mesh *tgt;
      tgt = &MESH_L(ix,iy,iz);

      float wmol = WMOL(tgt->chem);
      float gamma;

      gamma = gamma_total(tgt, this_run);

      //unit scale
      float kene = 0.5*NORML2(tgt->momx,tgt->momy,tgt->momz)/tgt->dens; 
      float pene = -0.5*tgt->pot*tgt->dens;
      float tene = tgt->uene*tgt->dens;
      float etrp_func = (gamma-1.0)*tgt->uene*tgt->dens/pow(tgt->dens,gamma);

      if(kene <= 0.0) kene = 1.0e-20; 
      if(pene <= 0.0) pene = 1.0e-20; 
      if(tene <= 0.0) tene = 1.0e-20;
      if(tgt->uene <= 0.0) tgt->uene = 1.0e-20; 
      if(tgt->dens <= 0.0) tgt->dens = 1.0e-20;
      if(etrp_func <= 0.0) etrp_func = 1.0e-20; 
      
      map[0][jx][jy] = log10(tgt->chem.fHI);
      map[1][jx][jy] = log10(tgt->uene*this_run->uenetok*wmol);  //tmpr
      map[2][jx][jy] = log10(tgt->uene);                         //uene //unit scale
      map[3][jx][jy] = log10(tene); 
      map[4][jx][jy] = log10(pene);
      //map[4][jx][jy] = log10(tgt->eneg);
      map[5][jx][jy] = log10(kene);
      map[6][jx][jy] = log10(tgt->dens*this_run->denstonh); //cgs
      map[7][jx][jy] = log10(etrp_func); //unit

    }
  }

  for(ix=0;ix<NMESH_X_LOCAL;ix+=VS) {
    int jx = (ix+this_run->rank_x*NMESH_X_LOCAL)/VS;
    for(iy=0;iy<NMESH_Y_LOCAL;iy+=VS) {
      int jy = (iy+this_run->rank_y*NMESH_Y_LOCAL)/VS;

      struct fluid_mesh *tgt;
      tgt = &MESH_L(ix,iy,iz);
      
      vec[0][jx][jy] = tgt->momx/tgt->dens;
      vec[1][jx][jy] = tgt->momy/tgt->dens;
    }
  }

}


float get_tick(float min, float max)
{
  float eff, sft;
  
  eff = max-min;
  sft = 1.0;
  while(eff >= 10.0) {
    eff /= 10.0; sft *= 10.0;
  }
  while(eff < 1.0) {
    eff *= 10.0;
    sft /= 10.0;
  }
  
  float tick;
  
  if(eff > 5.0) {
    tick = sft;
  }else if(eff > 2.0) {
    tick = sft*0.5;
  }else{
    tick = sft*0.2;
  }
  
  return tick;
}

void get_minmax(float fmin, float fmax, float *vmin, float *vmax)
{
  float bln;
  float fave;
  
  bln = 0.5*(fmin+fmax);
  
  if(fmin > bln && fmax > bln) {
    fmin = bln;
  }else if(fmin<bln && fmax<bln) {
    fmax = bln;
  }else if(fmin==fmax) {
    fmax += 0.9;
  }

  float tick = get_tick(fmin, fmax);
  
  int bln_bin = bln/tick;
  bln = bln_bin * tick;
  
  float list_max, list_min;
  
  list_max = list_min = bln;
  int i;
  if(fmax != bln) {
    for( i=1;i<=(fmax-bln)/tick+1;i++) {
      list_min = MIN(list_min, i*tick+bln);
      list_max = MAX(list_max, i*tick+bln);
    }
  }
  
  if(fmin != bln) {
    for( i=-1;i>=(fmin-bln)/tick-1;i--) {
      list_min = MIN(list_min, i*tick+bln);
      list_max = MAX(list_max, i*tick+bln);      
    }
  }
  
  *vmin = list_min;
  *vmax = list_max;
}


void set_contour(const PLFLT** map, int nbinx, int nbiny, float vmin, float vmax)
{
  int i,j;
  static PLFLT cont_level[CN];
  for(i=0; i<CN; i++) cont_level[i] = vmin + (vmax-vmin)/(CN-1)*i;

  PLcGrid2  cgrid2;//PLcGrid2  xg,yg,nx,ny
  plAlloc2dGrid(&cgrid2.xg,nbinx,nbiny);
  plAlloc2dGrid(&cgrid2.yg,nbinx,nbiny);
  cgrid2.nx = nbinx;
  cgrid2.ny = nbiny;
  //Set Grid
  for(i=0;i<nbinx;i++){
    for(j=0;j<nbiny;j++){
      cgrid2.xg[i][j]=(double)i+1;///nbinx;
      cgrid2.yg[i][j]=(double)j+1;///nbiny;
    }
  }

  plcont(map, nbinx, nbiny, 1, nbinx, 1, nbiny, cont_level, CN, pltr2, (void*) &cgrid2);

  plFree2dGrid(cgrid2.xg,nbinx,nbiny);
  plFree2dGrid(cgrid2.yg,nbinx,nbiny);
} 



void set_colorbar(float vmin, float vmax, char *bar_label)
{
  PLFLT colorbar_width,colorbar_height;
  PLINT opt = PL_COLORBAR_GRADIENT;
  PLINT position = PL_POSITION_RIGHT|PL_POSITION_OUTSIDE;
  PLFLT cb_x,cb_y;
  cb_x = 0.05;
  cb_y = 0.0;

  PLFLT cb_x_length,cb_y_length;
  cb_x_length = 0.05; 
  cb_y_length = 1.0;
  
  PLINT cb_bg_color, cb_bb_color, cb_bb_style;
  cb_bg_color = 15;
  cb_bb_color = 1;
  cb_bb_style = 1;

  PLFLT low_cap_color, high_cap_color;
  low_cap_color  = 0.0;  
  high_cap_color = 1.0;
  
  PLINT cont_color, cont_width;
  cont_color = cont_width = 0;

  PLFLT ticks[1] = { 0.0e0 };
  PLINT sub_ticks[1] = { 0 };

  PLINT n_axes = 1;
  const char* axis_opts[1];
  plsmaj(0,2);  //tics length (default,scale)
  axis_opts[0] = "tvmxf";
  /// u : left tics
  /// w : right tics
  /// g : right tics from left bar
  /// t : value plot
  /// i : mark type , if i mark is outword ,else inward 
  /// v : label vertical
  /// n : label left
  /// m : label right
  /// l : log scale
  /// s : small tics
  /// x : non plot tics
  /// f : Always use fixed point numeric labels
  
  
  
  const char* labels[1];
  labels[0] = bar_label;
  PLINT n_labels = 1;
  PLINT label_opt[1] = {1};


  PLINT n_values_array[1];
  const PLFLT *values_array[1];
  PLFLT clev[2] = {vmin, vmax};
  n_values_array[0] = 2;
  values_array[0] = clev;
  

  plcolorbar( &colorbar_width, &colorbar_height,
	      opt , position,
	      cb_x, cb_y, cb_x_length, cb_y_length,
	      cb_bg_color, cb_bb_color, cb_bb_style,
	      low_cap_color, high_cap_color,
	      cont_color, cont_width,
	      n_labels, label_opt, labels,
	      n_axes, axis_opts, 
	      ticks, sub_ticks,
	      n_values_array, values_array );
}




void view_map(PLFLT **map, PLFLT *xscale, PLFLT *yscale, 
	      int nbinx, int nbiny, 
	      int x_start, int x_end, int y_start, int y_end,
	      float vmin, float vmax, 
	      char *title_label, char *xlabel, char *ylabel, char *bar_label)
{
  plcol0(14);
  plenv( x_start, x_end, y_start, y_end, 2, -1 );

  pllab( xlabel, ylabel, title_label);

  //  plimage((const PLFLT**)map, nbinx, nbiny, x_start, x_end, y_start, y_end, 
  //	  0 , 0 , x_start, x_end, y_start, y_end);
  plimagefr ((const PLFLT**)map, nbinx, nbiny, x_start, x_end, y_start, y_end,  
	     0.0 , 0.0 , vmin, vmax, NULL, NULL);

#ifdef __CONTOUR__
  set_contour((const PLFLT**)map, nbinx, nbiny, vmin, vmax);
#endif

  set_colorbar(vmin, vmax, bar_label);
}


void view_vec(PLFLT **vec[2], int nbinx, int nbiny, 
	      int x_start, int x_end, int y_start, int y_end,
	      char *title_label, char *xlabel, char *ylabel)
{
  plcol0(14);
  plenv( x_start, x_end, y_start, y_end, 2, -1 );

  pllab( xlabel, ylabel, title_label);

  int i,j;
  PLcGrid2  cgrid2;//PLcGrid2  xg,yg,nx,ny
  plAlloc2dGrid(&cgrid2.xg,nbinx,nbiny);
  plAlloc2dGrid(&cgrid2.yg,nbinx,nbiny);
  cgrid2.nx = nbinx;
  cgrid2.ny = nbiny;
  //Set Grid
  for(i=0;i<nbinx;i++){
    for(j=0;j<nbiny;j++){
      cgrid2.xg[i][j]=i+0.5;///nbinx;
      cgrid2.yg[i][j]=j+0.5;///nbiny;
    }
  }

#if 0
  PLFLT x[6]={-0.5,0.04,0.0 ,0.5,0.0  ,0.04};
  PLFLT y[6]={0.0 ,0.0 ,0.12,0.0,-0.12,0.0 };
  plsvect(x,y,6,true);
#endif

  plvect((const PLFLT**)vec[0], (const PLFLT**)vec[1], nbinx, nbiny, 0.0, 
	 pltr2, (void *) &cgrid2 );

  plFree2dGrid(cgrid2.xg,nbinx,nbiny);
  plFree2dGrid(cgrid2.yg,nbinx,nbiny);
}



int main(int argc, char **argv) 
{
  char pl_version[80];
  plgver(pl_version);
  fprintf(stdout, "PLplot library version: %s\n", pl_version);
  fprintf(stdout, "If arvg[4]=argv[5]=0.0 , color scale is auto.\n");
  plparseopts(&argc, argv, PL_PARSE_FULL | PL_PARSE_SKIP);
    

  static struct run_param this_run;

  static struct fluid_mesh mesh[NMESH_LOCAL];
  static struct radiation_src src[NSOURCE_MAX];

  static char filename[256];

  float slice_pos;
  PLFLT **map[MN];
  PLFLT **vec[2]; 
  PLFLT *pos1, *pos2;

  this_run.proc_file=stdout;

  if(argc != 7) {
    fprintf(stderr, 
	    "Usage: %s <input prefix> <X|Y|Z> <slice position> <vmin> <vmax> <A:all|F:frac|T:tmpr|D:dens>\n",
	    argv[0]);
    exit(EXIT_FAILURE);
  }
  

  slice_pos = atof(argv[3]);

  for(int mn=0; mn<MN; mn++) {
    if(strcmp(argv[2],"X") == 0) {
      plAlloc2dGrid(&map[mn],NMESH_Y_TOTAL,NMESH_Z_TOTAL);
    }else if(strcmp(argv[2],"Y") == 0) {
      plAlloc2dGrid(&map[mn],NMESH_X_TOTAL,NMESH_Z_TOTAL);
    }else if(strcmp(argv[2],"Z") == 0) {
      plAlloc2dGrid(&map[mn],NMESH_X_TOTAL,NMESH_Y_TOTAL);
    }else{
      fprintf(stderr, "Invalid slice direction\n");
      exit(EXIT_FAILURE);
    }
  }    
  
  if(strcmp(argv[2],"X") == 0) {
    pos1 = (PLFLT *) malloc(sizeof(PLFLT)*NMESH_Y_TOTAL);
    pos2 = (PLFLT *) malloc(sizeof(PLFLT)*NMESH_Z_TOTAL);
    plAlloc2dGrid(&vec[0],NMESH_Y_TOTAL,NMESH_Z_TOTAL);
    plAlloc2dGrid(&vec[1],NMESH_Y_TOTAL,NMESH_Z_TOTAL);
  }else if(strcmp(argv[2],"Y") == 0) {
    pos1 = (PLFLT *) malloc(sizeof(PLFLT)*NMESH_X_TOTAL);
    pos2 = (PLFLT *) malloc(sizeof(PLFLT)*NMESH_Z_TOTAL);
    plAlloc2dGrid(&vec[0],NMESH_X_TOTAL,NMESH_Z_TOTAL);
    plAlloc2dGrid(&vec[1],NMESH_X_TOTAL,NMESH_Z_TOTAL);
  }else if(strcmp(argv[2],"Z") == 0) {
    pos1 = (PLFLT *) malloc(sizeof(PLFLT)*NMESH_X_TOTAL);
    pos2 = (PLFLT *) malloc(sizeof(PLFLT)*NMESH_Y_TOTAL);
    plAlloc2dGrid(&vec[0],NMESH_X_TOTAL,NMESH_Y_TOTAL);
    plAlloc2dGrid(&vec[1],NMESH_X_TOTAL,NMESH_Y_TOTAL);
  }else{
    fprintf(stderr, "Invalid slice direction\n");
    exit(EXIT_FAILURE);
  }
    
  char *type;
  type = argv[6];
  
  if(strcmp(type,"A")!=0&&strcmp(type,"F")!=0&&
     strcmp(type,"T")!=0&&strcmp(type,"D")!=0) {
    fprintf(stderr, "Invalid target map\n");
    exit(EXIT_FAILURE);
  }


  int rank_x, rank_y, rank_z;
  for(rank_x=0;rank_x<NNODE_X;rank_x++) {
    for(rank_y=0;rank_y<NNODE_Y;rank_y++) {
      for(rank_z=0;rank_z<NNODE_Z;rank_z++) {
	FILE *fp;
      
	sprintf(filename, "%s_%03d_%03d_%03d", 
		argv[1], rank_x, rank_y, rank_z);

	fp = fopen(filename,"r");
	input_mesh_header(&this_run, filename);
	fclose(fp);
	
	if(strcmp(argv[2],"X") == 0 &&
	   this_run.xmin_local < slice_pos &&
	   this_run.xmax_local >= slice_pos ) {

	  fp = fopen(filename,"r");
	  input_mesh_single(mesh, &this_run, filename);
	  fclose(fp);
	  
	  compile_map_x(mesh, map, vec,
			pos1, pos2, slice_pos, &this_run);
	 
	}else if(strcmp(argv[2],"Y") == 0 &&
		 this_run.ymin_local < slice_pos &&
		 this_run.ymax_local >= slice_pos) {

	  fp = fopen(filename,"r");
	  input_mesh_single(mesh, &this_run, filename);
	  fclose(fp);

	  compile_map_y(mesh, map, vec, 
			pos1, pos2, slice_pos, &this_run);

	}else if(strcmp(argv[2],"Z") == 0 &&
		 this_run.zmin_local < slice_pos &&
		 this_run.zmax_local >= slice_pos) {

	  fp = fopen(filename,"r");
	  input_mesh_single(mesh, &this_run, filename);
	  fclose(fp);
	  compile_map_z(mesh, map, vec,
			pos1, pos2, slice_pos, &this_run);
	  
	}

      }
    }
  }


  float fmin[MN], fmax[MN];

  for(int mn=0; mn<MN; mn++) {
    
    fmin[mn] = FLT_MAX;
    fmax[mn] = -FLT_MAX;
  
    int imesh,jmesh;
    if(strcmp(argv[2],"X")==0) {
      for(imesh=0;imesh<NMESH_Y_TOTAL;imesh++) {
	for(jmesh=0;jmesh<NMESH_Z_TOTAL;jmesh++) {
	  fmin[mn] = MIN(map[mn][imesh][jmesh], fmin[mn]);
	  fmax[mn] = MAX(map[mn][imesh][jmesh], fmax[mn]);
	}
      }
    }else if(strcmp(argv[2],"Y")==0) {
      for(imesh=0;imesh<NMESH_X_TOTAL;imesh++) {
	for(jmesh=0;jmesh<NMESH_Z_TOTAL;jmesh++) {
	  fmin[mn] = MIN(map[mn][imesh][jmesh], fmin[mn]);
	  fmax[mn] = MAX(map[mn][imesh][jmesh], fmax[mn]);
	}
      }
    }else if(strcmp(argv[2],"Z")==0) {
      for(imesh=0;imesh<NMESH_X_TOTAL;imesh++) {
	for(jmesh=0;jmesh<NMESH_Y_TOTAL;jmesh++) {
	  fmin[mn] = MIN(map[mn][imesh][jmesh], fmin[mn]);
	  fmax[mn] = MAX(map[mn][imesh][jmesh], fmax[mn]);
	}
      }
    }
  }

#if 0
  printf("# max value for fraction : %14.6e \n", fmax_frac);
  printf("# min value for fraction : %14.6e \n", fmin_frac);
  printf("# max value for temperature : %14.6e \n", fmax_tmpr);
  printf("# min value for temperature : %14.6e \n", fmin_tmpr);
  printf("# max value for number density : %14.6e \n", fmax_dens);
  printf("# min value for number density : %14.6e \n", fmin_dens);
#endif
 
  /* draw the maps */

  float vmin[MN], vmax[MN];
  
  for(int mn=0; mn<MN; mn++) {
    
    if(atof(argv[4])==0.0 && atof(argv[5])==0.0 ) {
      get_minmax(fmin[mn], fmax[mn], &vmin[mn], &vmax[mn]);
      
    }else{
      fmin[mn] = vmin[mn] = atof(argv[4]);
      fmax[mn] = vmax[mn] = atof(argv[5]);
    }
  }
    


#if 0
  printf("# max value for fraction : %14.6e \n", vmax[0]);
  printf("# min value for fraction : %14.6e \n", vmin[0]);
  printf("# max value for temperature : %14.6e \n", vmax[1]);
  printf("# min value for temperature : %14.6e \n", vmin[1]);
  printf("# max value for number density : %14.6e \n", vmax[2]);
  printf("# min value for number density : %14.6e \n", vmin[2]);
#endif


  ///background color , default line color red
  plscolbg(255, 255, 255);
  //change col0 color
  plscol0(14, 0, 0, 0);
  plinit();

  PLFLT r[3]={0.0,1.0,1.0};
  PLFLT g[3]={0.0,0.0,1.0};
  PLFLT b[3]={1.0,0.0,1.0};
  PLFLT pos[3]={0.0,0.9,1.0};
  plscmap1l(true, 3, pos, r, g, b, NULL);
  
  
  for(int mn=0; mn<MN; mn++) {

    if(strcmp(type,"F")==0) mn=0;
    if(strcmp(type,"T")==0) mn=1;
    if(strcmp(type,"D")==0) mn=6;
    
    printf("# max value  : %14.6e \n", vmax[mn]);
    printf("# min value  : %14.6e \n", vmin[mn]);
    
    if(strcmp(argv[2],"X")==0) {
      view_map(map[mn], pos1, pos2, NMESH_Y_TOTAL, NMESH_Z_TOTAL, 
	       1, NMESH_Y_TOTAL, 1, NMESH_Z_TOTAL, vmin[mn], vmax[mn], 
	       plot_title[mn], "Y", "Z", "log");
      
      plFree2dGrid(map[mn],NMESH_Y_TOTAL,NMESH_Z_TOTAL);
      
    }else if(strcmp(argv[2],"Y")==0) {
      view_map(map[mn], pos1, pos2, NMESH_X_TOTAL, NMESH_Z_TOTAL, 
	       1, NMESH_X_TOTAL, 1, NMESH_Z_TOTAL, vmin[mn], vmax[mn],
	       plot_title[mn], "X", "Z", "log");
      
      plFree2dGrid(map[mn],NMESH_X_TOTAL,NMESH_Z_TOTAL);
      
    }else if(strcmp(argv[2],"Z")==0) {
      view_map(map[mn], pos1, pos2, NMESH_X_TOTAL, NMESH_Y_TOTAL, 
	       1, NMESH_X_TOTAL, 1, NMESH_Y_TOTAL, vmin[mn], vmax[mn], 
	       plot_title[mn], "X", "Y", "log");
      
      plFree2dGrid(map[mn],NMESH_X_TOTAL,NMESH_Y_TOTAL);
      
    }

    if(strcmp(type,"F")==0) mn=MN;
    if(strcmp(type,"T")==0) mn=MN;
    if(strcmp(type,"D")==0) mn=MN;
  }
 
  /* plot vector */
  if(strcmp(argv[2],"X")==0) {
    view_vec(vec, NMESH_Y_TOTAL/VS, NMESH_Z_TOTAL/VS, 1, NMESH_Y_TOTAL/VS, 1, NMESH_Z_TOTAL/VS,
	     "vec", "Y", "Z");
    plFree2dGrid(vec[0],NMESH_Y_TOTAL,NMESH_Z_TOTAL);
    plFree2dGrid(vec[1],NMESH_Y_TOTAL,NMESH_Z_TOTAL);
  }else if(strcmp(argv[2],"Y")==0) {
    view_vec(vec, NMESH_X_TOTAL/VS, NMESH_Z_TOTAL/VS, 1, NMESH_X_TOTAL/VS, 1, NMESH_Z_TOTAL/VS,
	     "vec", "X", "Z");
    plFree2dGrid(vec[0],NMESH_X_TOTAL,NMESH_Z_TOTAL);
    plFree2dGrid(vec[1],NMESH_X_TOTAL,NMESH_Z_TOTAL);
  }else if(strcmp(argv[2],"Z")==0) {
    view_vec(vec, NMESH_X_TOTAL/VS, NMESH_Y_TOTAL/VS, 1, NMESH_X_TOTAL/VS, 1, NMESH_Y_TOTAL/VS,
	     "vec", "X", "Y");
    plFree2dGrid(vec[0],NMESH_X_TOTAL,NMESH_Y_TOTAL);
    plFree2dGrid(vec[1],NMESH_X_TOTAL,NMESH_Y_TOTAL);
  }
  
 
  free(pos1);
  free(pos2);

  plend();
  return EXIT_SUCCESS;
    
}
