#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "constants.h"
#include "run_param.h"
#include "source.h"

#include "prototype.h"

int main(int argc, char **argv) 
{
  struct run_param this_run;
  struct radiation_src *src;

  if(argc != 2) {
    fprintf(stderr, "Usage :: %s <source file>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  src = (struct radiation_src*) malloc(sizeof(struct radiation_src)*NSOURCE_MAX);

  input_src_file(src, &this_run, argv[1]);

  printf("# of sources : %llu\n", this_run.nsrc);
  printf("# X            Y            Z            photon rate  type         param\n");

  for(int isrc=0;isrc<this_run.nsrc;isrc++) {
    double total_count;
    total_count = 0.0;
    for(int inu=0;inu<NGRID_NU;inu++) total_count += src[isrc].photon_rate[inu];

    printf("%12.4e %12.4e %12.4e %12.4e %12d %12.4e\n", 
	   src[isrc].xpos,src[isrc].ypos,src[isrc].zpos,
	   total_count, src[isrc].type, src[isrc].param);
  }

  free(src);
}
