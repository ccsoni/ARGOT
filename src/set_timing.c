#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/times.h>

#include "run_param.h"

float timing(struct tms, struct tms);
float wallclock_timing(struct timeval, struct timeval);

int start_timing(struct timeval *start_tv, struct tms *start_tms)
{
#ifdef __ARGOT_PROFILE__
  times(start_tms);
  gettimeofday(start_tv, NULL);
#endif
  return 0;
}

int end_timing(struct timeval *start_tv, struct timeval *end_tv,
		struct tms *start_tms, struct tms *end_tms,
		char *label, struct run_param *this_run)
{
#ifdef __ARGOT_PROFILE__
  times(end_tms);
  gettimeofday(end_tv, NULL);

  fprintf(this_run->proc_file,
	  "# %s : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",label,
	  timing(*start_tms, *end_tms), wallclock_timing(*start_tv, *end_tv));
  fflush(this_run->proc_file);
#endif
  return 0;
}
