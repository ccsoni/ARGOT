#include "run_param.h"
#include "cross_section.h"
#include "chemistry.h"

void setup_cross_section(struct cross_section *csect, struct freq_param *freq)
{
  for(int inu=0;inu<NGRID_NU;inu++) {
    csect[inu].csect_HI   = csectHI(freq->nu[inu]);
#ifdef __HELIUM__
    csect[inu].csect_HeI  = csectHeI(freq->nu[inu]);
    csect[inu].csect_HeII = csectHeII(freq->nu[inu]);
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
    csect[inu].csect_HM      = csectHM(freq->nu[inu]);
    csect[inu].csect_H2I_I   = csectH2I_I(freq->nu[inu]);
    csect[inu].csect_H2II_I  = csectH2II_I(freq->nu[inu]);
    csect[inu].csect_H2II_II = csectH2II_II(freq->nu[inu]);
#endif /* __HYDROGEN_MOL__ */


  }
}
