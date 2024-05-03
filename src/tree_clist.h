#ifndef __ARGOT_TREE_CLIST__
#define __ARGOT_TREE_CLIST__

#include "run_param.h"
#include "radiation.h"

struct clist_t {
  uint64_t key;
  float cm[3];
  double L;
  double spec[NGRID_NU];
  //  float l_theta;
  float length;
  int next;
  int key_level;
  int ifirst;
  int n;
  int gflag; /* 1: group cell 0: upper cell 2: lower cell */
  int zeroflag;
};

void setup_light_ray_range(int, int, struct light_ray*, uint64_t, struct clist_t*, 
			   struct radiation_src*, int*, struct run_param*);
void construct_tree(struct clist_t*, uint64_t*, int*, 
		    struct radiation_src*, struct run_param*);
uint64_t count_ray_to(int, int, int, struct clist_t*, int*, struct run_param*);
void init_tree_param(int, struct run_param*, int*);
#endif /*__ARGOT_TREE_CLIST__ */
