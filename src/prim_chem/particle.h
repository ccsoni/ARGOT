#ifndef __PARTICLE__
#define __PARTICLE__

#include "sph.h"

struct Particle { 
  int   isph; /* negative for dark matter particle */ 
              /* and positive for indices of SPH particles */ 
  int   indx; 
  int   ikick; 
  int   step_lvl; 
  float mass; 
  float xpos; 
  float ypos; 
  float zpos; 
  float xvel; 
  float yvel; 
  float zvel; 
  
  float xacc; 
  float yacc; 
  float zacc; 
  
  float dtime; 
  float xvel_pred; 
  float yvel_pred; 
  float zvel_pred;
};

#endif /* __PARTICLE__ */
