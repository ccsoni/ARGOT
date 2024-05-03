#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "cpgplot.h"

void palett(int type, float contr, float bright)
{

  static float GL[2], GR[2], GG[2], GB[2];
  static float RL[9], RR[9], RG[9], RB[9];
  static float HL[5], HR[5], HG[5], HB[5];
  static float WL[10],WR[10],WG[10],WB[10];
  static float AL[20],AR[20],AG[20],AB[20];

  GL[0]=0.0; 
  GL[1]=1.0;
  GR[0]=0.0; 
  GR[1]=1.0;
  GG[0]=0.0; 
  GG[1]=1.0;
  GB[0]=0.0; 
  GB[1]=1.0;

  RL[0]=-0.5;
  RL[1]=0.0;
  RL[2]=0.17;
  RL[3]=0.33;
  RL[4]=0.5;
  RL[5]=0.67;
  RL[6]=0.83;
  RL[7]=1.0;
  RL[8]=1.7;

  RR[0]=0.0;
  RR[1]=0.0;
  RR[2]=0.0;
  RR[3]=0.0;
  RR[4]=0.6;
  RR[5]=1.0;
  RR[6]=1.0;
  RR[7]=1.0;
  RR[8]=1.0;

  RG[0]=0.0;
  RG[1]=0.0;
  RG[2]=0.0;
  RG[3]=1.0;
  RG[4]=1.0;
  RG[5]=1.0;
  RG[6]=0.6;
  RG[7]=0.0;
  RG[8]=1.0;

  RB[0]=0.0;
  RB[1]=0.3;
  RB[2]=0.8;
  RB[3]=1.0;
  RB[4]=0.3;
  RB[5]=0.0;
  RB[6]=0.0;
  RB[7]=0.0;
  RB[8]=1.0;

  HL[0]=0.0;
  HL[1]=0.2;
  HL[2]=0.4;
  HL[3]=0.6;
  HL[4]=1.0;

  HR[0]=0.0;
  HR[1]=0.5;
  HR[2]=1.0;
  HR[3]=1.0;
  HR[4]=1.0;

  HG[0]=0.0;
  HG[1]=0.0;
  HG[2]=0.5;
  HG[3]=1.0;
  HG[4]=1.0;

  HB[0]=0.0;
  HB[1]=0.0;
  HB[2]=0.0;
  HB[3]=0.3;
  HB[4]=1.0;

  WL[0]=0.0;
  WL[1]=0.5;
  WL[2]=0.5;
  WL[3]=0.7;
  WL[4]=0.7;
  WL[5]=0.85;
  WL[6]=0.85;
  WL[7]=0.95;
  WL[8]=0.95;
  WL[9]=1.0;

  WR[0]=0.0;
  WR[1]=1.0;
  WR[2]=0.0;
  WR[3]=0.0;
  WR[4]=0.3;
  WR[5]=0.8;
  WR[6]=0.3;
  WR[7]=1.0;
  WR[8]=1.0;
  WR[9]=1.0;

  WG[0]=0.0;
  WG[1]=0.5;
  WG[2]=0.4;
  WG[3]=1.0;
  WG[4]=0.0;
  WG[5]=0.0;
  WG[6]=0.2;
  WG[7]=0.7;
  WG[8]=1.0;
  WG[9]=1.0;

  WB[0]=0.0;
  WB[1]=0.0;
  WB[2]=0.0;
  WB[3]=0.0;
  WB[4]=0.4;
  WB[5]=1.0;
  WB[6]=0.0;
  WB[7]=0.0;
  WB[8]=0.95;
  WB[9]=1.0;

  AL[0]=0.0;
  AL[1]=0.1;
  AL[2]=0.1;
  AL[3]=0.2;
  AL[4]=0.2;
  AL[5]=0.3;
  AL[6]=0.3;
  AL[7]=0.4;
  AL[8]=0.4;
  AL[9]=0.5;
  AL[10]=0.5;
  AL[11]=0.6;
  AL[12]=0.6;
  AL[13]=0.7;
  AL[14]=0.7;
  AL[15]=0.8;
  AL[16]=0.8;
  AL[17]=0.9;
  AL[18]=0.9;
  AL[19]=1.0;

  AR[0]=0.0;
  AR[1]=0.0;
  AR[2]=0.3;
  AR[3]=0.3;
  AR[4]=0.5;
  AR[5]=0.5;
  AR[6]=0.0;
  AR[7]=0.0;
  AR[8]=0.0;
  AR[9]=0.0;
  AR[10]=0.0;
  AR[11]=0.0;
  AR[12]=0.0;
  AR[13]=0.0;
  AR[14]=1.0;
  AR[15]=1.0;
  AR[16]=1.0;
  AR[17]=1.0;
  AR[18]=1.0;
  AR[19]=1.0;

  AG[0]=0.0;
  AG[1]=0.0;
  AG[2]=0.3;
  AG[3]=0.3;
  AG[4]=0.0;
  AG[5]=0.0;
  AG[6]=0.0;
  AG[7]=0.0;
  AG[8]=0.8;
  AG[9]=0.8;
  AG[10]=0.6;
  AG[11]=0.6;
  AG[12]=1.0;
  AG[13]=1.0;
  AG[14]=1.0;
  AG[15]=1.0;
  AG[16]=0.8;
  AG[17]=0.8;
  AG[18]=0.0;
  AG[19]=0.0;

  AB[0]=0.0;
  AB[1]=0.0;
  AB[2]=0.3;
  AB[3]=0.3;
  AB[4]=0.7;
  AB[5]=0.7;
  AB[6]=0.7;
  AB[7]=0.7;
  AB[8]=0.9;
  AB[9]=0.9;
  AB[10]=0.0;
  AB[11]=0.0;
  AB[12]=0.0;
  AB[13]=0.0;
  AB[14]=0.0;
  AB[15]=0.0;
  AB[16]=0.0;
  AB[17]=0.0;
  AB[18]=0.0;
  AB[19]=0.0;

  if(type == 1) {
    cpgctab(GL, GR, GG, GB, 2, contr, bright);
  }else if(type == 2) {
    cpgctab(RL, RR, RG, RB, 9, contr, bright);
  }else if(type == 3) {
    cpgctab(HL, HR, HG, HB, 5, contr, bright);
  }else if(type == 4) {
    cpgctab(WL, WR, WG, WB, 10,contr, bright);
  }else if(type == 5) {
    cpgctab(AL, AR, AG, AB, 20,contr, bright);
  }
    
}

#define NCOLOR (256)
#define NCONT  (10)

void view_map(float *map, float *xscale, float *yscale, 
	      int nbinx, int nbiny, 
	      int x_start, int x_end, int y_start, int y_end,
	      float vmin, float vmax, 
	      char *xlabel, char *ylabel)
{
  static float trans[6];
  static float contour[NCONT];
  static float cscale[NCOLOR];
  
  float aspect;

  int ix,iy;

  if(vmin == vmax) {
    vmax = 0.0;
    vmin = 0.0;
    for(ix=0;ix<nbinx*nbiny;ix++) {
      if(vmax < map[ix]) vmax = map[ix];
    }
    vmax = ceil(vmax);
  }

  aspect = 1.0;

  cpgpap(5.0,1.05);
  cpgsubp(1,1);
  cpgsch(1.0);
  cpgsvp(0.1,0.9,0.1,0.9);
  cpgswin(0.0, 1.0, 0.0, 1.05);

  cpgslw(1);
  cpgscf(2);

  cpgscir(1,NCOLOR);
  palett(2, 1.0, 0.5);

  trans[0]=0.1-0.5*0.8/nbinx;
  trans[1]=0.8/nbinx;
  trans[2]=0.0;
  trans[3]=0.1-0.5*0.8/nbiny;
  trans[4]=0.0;
  trans[5]=0.8/nbiny*aspect;

  cpgimag(map, nbinx, nbiny, 
  	  x_start, x_end, y_start, y_end, vmin, vmax, trans);

  float cont_vmax, cont_vmin;
  int ncont,icont;

  ncont = 10;
  cont_vmin = vmin;
  cont_vmax = vmax;

  assert(ncont <= NCONT);

  for(icont=0;icont<ncont;icont++)
    contour[icont] = cont_vmin + 
      (float)icont*(cont_vmax-cont_vmin)/(float)(ncont-1);

  cpgcont(map, nbinx, nbiny, 
  	  x_start, x_end, y_start, y_end, 
  	  contour, ncont, trans);

  cpgscr(0, 0.0, 0.0, 0.0);
  cpgsci(0);

  cpgaxis("N",0.1,0.1,0.9,0.1,xscale[x_start],xscale[x_end-1],
          0.0,5,0.5,0.0,0.5,0.5,0.0);

  cpgaxis("N",0.1,0.1,0.1,0.1+0.8*aspect,yscale[y_start],yscale[y_end-1],
          0.0,5,0.0,0.5,0.5,-0.5,90.0);

  cpgaxis("N",0.1,0.1+0.8*aspect,0.9,0.1+0.8*aspect,
	  xscale[x_start],xscale[x_end-1],
          0.0,5,0.0,0.5,0.5,-10.5,0.0);

  cpgaxis("N",0.9,0.1,0.9,0.1+0.8*aspect,yscale[y_start],yscale[y_end-1],
          0.0,5,0.5,0.0,0.5,-100.5,0.0);

  cpgmtxt("B",0.5,0.5,0.5,xlabel);
  cpgmtxt("L",0.5,0.5,0.5,ylabel);

  trans[0]=0.92-0.025;
  trans[1]=0.05;
  trans[2]=0.0;
  trans[3]=0.1;
  trans[4]=0.0;
  trans[5]=0.8*aspect/NCOLOR;

  int icol;
  for(icol=0;icol<NCOLOR;icol++) {
    cscale[icol]=vmin+(float)icol*(vmax-vmin)/(float)(NCOLOR-1);
  }

  cpgimag(cscale,1,NCOLOR,1,1,1,NCOLOR,vmin,vmax,trans);

  cpgaxis("N",0.97,0.1,0.97,0.1+0.8*aspect,vmin,vmax,
          0.0,5,0.5,0.0,0.5,0.5,0.0);

  cpgaxis("N",0.92,0.1,0.92,0.1+0.8*aspect,vmin,vmax,
          0.0,5,0.0,0.5,0.5,100.5,0.0);

  cpgaxis("N",0.92,0.1+0.8*aspect,0.97,0.1+0.8*aspect,0.1,1.0,
          1.0,1,0.0,0.5,0.5,100.5,0.0);

  cpgaxis("N",0.92,0.1,0.97,0.1,0.1,1.0,
          1.0,1,0.0,0.5,0.5,100.5,0.0);


}

#undef NCOLOR

