#pragma once
#ifndef __ARGOT_OMP_PARAM__
#define __ARGOT_OMP_PARAM__

#include <omp.h>

struct omp_param {
  omp_nest_lock_t omp_lock;
};

/* sizeof(omp_nest_lock_t) = 16 byte */

#endif /* __ARGOT_OMP_PARAM__ */
