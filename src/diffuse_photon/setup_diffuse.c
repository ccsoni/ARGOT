#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "diffuse_photon.h"
#include "diffuse_prototype.h"

void setup_diffuse_photon(struct host_diffuse_param *hd_param, struct run_param *this_run)
{
  set_angle_info(hd_param->angle,hd_param->corner_id_num,this_run);
  sort_angle_id(hd_param->angle);

  set_step_func_factor(hd_param->step_fact);
}


#if 0
  if(this_run->mpi_rank==0) {
    int c0,c1,c2,c3,c4,c5,c6,c7;
    int b0,b1,b2,b3,b4,b5;
    c0=c1=c2=c3=c4=c5=c6=c7=0;
    b0=b1=b2=b3=b4=b5=0;
    
    for(int ipix=0; ipix<N_ANG ;ipix++){
      printf("%d %d\n", hd_param->angle[ipix].corner_id,hd_param->angle[ipix].base_id);
      
      if(hd_param->angle[ipix].corner_id==0) c0++;
      if(hd_param->angle[ipix].corner_id==1) c1++;
      if(hd_param->angle[ipix].corner_id==2) c2++;
      if(hd_param->angle[ipix].corner_id==3) c3++;
      if(hd_param->angle[ipix].corner_id==4) c4++;
      if(hd_param->angle[ipix].corner_id==5) c5++;
      if(hd_param->angle[ipix].corner_id==6) c6++;
      if(hd_param->angle[ipix].corner_id==7) c7++;
      
      if(hd_param->angle[ipix].base_id==0) b0++;
      if(hd_param->angle[ipix].base_id==1) b1++;
      if(hd_param->angle[ipix].base_id==2) b2++;
      if(hd_param->angle[ipix].base_id==3) b3++;
      if(hd_param->angle[ipix].base_id==4) b4++;
      if(hd_param->angle[ipix].base_id==5) b5++;
    }
    
    printf("cid number (%d,%d,%d,%d,%d,%d,%d,%d)\n",c0,c1,c2,c3,c4,c5,c6,c7);
    printf("bid number (%d,%d,%d,%d,%d,%d)\n",b0,b1,b2,b3,b4,b5);
  }
  fflush(stdout);
  exit(1);
#endif

