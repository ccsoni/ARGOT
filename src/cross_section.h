#ifndef __ARGOT_CROSS_SECTION__
#define __ARGOT_CROSS_SECTION__

#include "run_param.h"
#include "source.h"

struct cross_section {
  float csect_HI;
#ifdef __HELIUM__
  float csect_HeI;
  float csect_HeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
  float csect_HM;
  float csect_H2I_I;
  //  float csect_H2I_II;
  float csect_H2II_I;
  float csect_H2II_II;
#endif /* __HYDROGEN_MOL__ */
};

void setup_cross_section(struct cross_section*, struct freq_param*);

#endif /* __ARGOT_CROSS_SECTION__ */
