/*
$ gcc -std=c11 check.c -lm
$ ./a.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SQR(x) ((x)*(x))

int main(int argc,char **argv)
{
  for(int ix=0; ix<100; ix++) {
    float x=6000/(double)(ix+0.1);
    
    float exp1f = 0.5*(5.0+2.0*SQR(x)*expf(-x)/SQR(1.0-expf(-x)));
    float exp2f = 0.5*(5.0+2.0*SQR(x)*expf(x)/SQR(expf(x)-1.0));
    float exp1d = 0.5*(5.0+2.0*SQR(x)*exp(-x)/SQR(1.0-exp(-x)));
    float exp2d = 0.5*(5.0+2.0*SQR(x)*exp(x)/SQR(exp(x)-1.0));

    printf("%d %e %e %e %e %e\n",ix,x,exp1f,exp2f,exp1d,exp2d);
  }
  
  return EXIT_SUCCESS;
}
