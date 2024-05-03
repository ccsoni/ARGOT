#include <stdio.h>

#include "chemistry.h"

int main(int argc, char **argv) 
{
  double nu;

  printf("# freq(1) HI(2) HeI(3) HeII(4) HM(5) H2II_I(6) H2II_II(7) H2I_I(8) H2I_II(9)\n");
  for(nu=0.05; nu < 10.0; nu*= 1.05) {
    printf("%14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n",
           nu, 
           csectHI(nu), csectHeI(nu),    csectHeII(nu), 
           csectHM(nu), csectH2II_I(nu), csectH2II_II(nu),
           csectH2I_I(nu), csectH2I_II(nu));
  }
}
