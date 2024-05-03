#include <math.h>

#include "chemistry.h"
#include "run_param.h"
#include "constants.h"

double photofuncHI(double nu, double tau, double T_bb)
// nu : freq. in units of nuL (3.28e+15 Hz)
// tau : optical depth at nuL
{
  csectHI(nu)*exp(-tau*csect(nu)/csect(nuL))*blackbody(nu*nuL,T_bb)
}

double calc_gamma_of_tau(double tau_nuL, double T_bb)
{
  
}
