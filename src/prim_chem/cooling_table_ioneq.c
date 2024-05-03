#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "constants.h"
#include "chemistry.h"

void calc_ioneq(struct prim_chem*, double, double, float);
double calc_heatcool_rate(struct prim_chem*, float, double, double);
double calc_heating_rate(struct prim_chem*, float, double, double);
double calc_cooling_rate(struct prim_chem*, float, double, double);

int main(int argc, char **argv) 
{
  struct prim_chem chem;
  double T,nH;
  double heatcool, heat, cool, wmol;
  float zred;

  if(argc==1) {
    fprintf(stderr, "Usage:: %s <zred>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  
  zred = atof(argv[1]);

  for(nH=1.e-8;nH<1.0;nH*=1.1){
    for(T=1.e2;T<1.e8;T*=1.1){
      calc_ioneq(&chem, nH, T, zred);
      heatcool = calc_heatcool_rate(&chem, zred, nH, T);
      heat = calc_heating_rate(&chem, zred, nH, T)/(nH*nH);
      cool = calc_cooling_rate(&chem, zred, nH, T)/(nH*nH);
      wmol = WMOL(chem);
      printf("%14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n", 
	     log10(nH), log10(T), 
	     //     log10(chem.fHI+1.0e-33), log10(chem.fHeI+1.0e-33), log10(chem.fHeII+1.0e-33), wmol);
	     log10(heat+1.0e-40), log10(cool+1.0e-40), log10(fabs(heat-cool)+1.0e-30), wmol);
    }
    printf("\n");
  }
  exit(0);
}
