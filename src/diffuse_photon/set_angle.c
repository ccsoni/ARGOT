#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chealpix.h>

#include "diffuse_photon.h"

inline int set_base_id(double, double, double, double);
inline int set_corner_id(double, double, double, int*);


void set_angle_info(struct angle_info *angle, int *corner_id_num, struct run_param *this_run)
{
  long   ipix, nside, npix;

  double theta,phi;
  double cos_theta, cos_phi, sin_theta, sin_phi, tan_theta;

  double xovr, yovr, zovr;

  double cos_pi_4 = cos(M_PI_4);


  if(N_ANG < 12) {
    fprintf(stderr, "Angle level is too low.\n");
    exit(1);
  }

  //  fprintf(this_run->proc_file,
  //	  "# Angle Level        :  %d \n", ANG_LEVEL);
  fprintf(this_run->proc_file,
	  "# N_SIDE        :  %d \n", N_SIDE);
  fprintf(this_run->proc_file,
	  "# Total angle number :  %d  \n", N_ANG);

  nside = N_SIDE;
  npix  = N_ANG;

  int i;
  for(i=0; i<8; i++) corner_id_num[i] = 0;

  for(ipix=0; ipix<npix ;ipix++){
    pix2ang_ring(nside,ipix,&theta,&phi);
    
    cos_theta = cos(theta); sin_theta = sin(theta);
    cos_phi   = cos(phi)  ; sin_phi   = sin(phi);
    tan_theta = tan(theta);
    
    xovr = sin_theta * cos_phi;
    if(fabs(xovr)<TINY)  xovr = ((xovr >= 0.0e0) ? TINY : -TINY); 
    angle[ipix].xovr = xovr;
    
    yovr = sin_theta * sin_phi;
    if(fabs(yovr)<TINY)  yovr = ((yovr >= 0.0e0) ? TINY : -TINY); 
    angle[ipix].yovr = yovr;
    
    zovr = cos_theta;
    if(fabs(zovr)<TINY)  zovr = ((zovr >= 0.0e0) ? TINY : -TINY); 
    angle[ipix].zovr = zovr;

    
    angle[ipix].base_id   = set_base_id(cos_phi, sin_phi, tan_theta, cos_pi_4);
    angle[ipix].corner_id = set_corner_id(cos_phi, sin_phi, cos_theta, corner_id_num);

  }

}



inline int set_base_id(double cos_phi, double sin_phi, double tan_theta, double cos_pi_4)
{
  double rev_cos_phi = 1.0e0/fabs(cos_phi);
  double rev_sin_phi = 1.0e0/fabs(sin_phi);

  if(fabs(cos_phi) >= cos_pi_4){
    
    if(0.0e0 <= tan_theta && tan_theta <= rev_cos_phi)  return 0;
    if(-rev_cos_phi <= tan_theta && tan_theta < 0.0e0)  return 1;

    if(cos_phi >= 0.0e0)  return 2;
    else                  return 3;
 
  }else{
    
    if(0.0e0 <= tan_theta && tan_theta <= rev_sin_phi)  return 0;
    if(-rev_sin_phi <= tan_theta && tan_theta < 0.0e0)  return 1;

    if(sin_phi >= 0.0e0)  return 4;
    else                  return 5;
  }
}



inline int set_corner_id(double cos_phi, double sin_phi, double cos_theta, int *corner_id_num)
{
  double cp,sp,ct;
  cp = cos_phi;
  sp = sin_phi;
  ct = cos_theta;

  if(cp > 0.0e0){
    if(sp > 0.0e0){
      if(ct > 0.0e0){
	corner_id_num[0]++;
	return 0;
      }else{
	corner_id_num[1]++;
	return 1;
      }     

    }else{
      if(ct > 0.0e0){
	corner_id_num[2]++;
	return 2;
      }else{
	corner_id_num[3]++;
	return 3;
      }
    } 
    
  }else{
    if(sp > 0.0e0){
      if(ct > 0.0e0){
	corner_id_num[4]++;
	return 4;
      }else{
	corner_id_num[5]++;
	return 5;
      }
      
    }else{
      if(ct > 0.0e0){
	corner_id_num[6]++;
	return 6;
      }else{
	corner_id_num[7]++;
	return 7;
      }
      
    }
  }
}



/*** Debug func ***/
#if 0
void set_single_angle_info(struct angle_info *angle, double theta, double phi, int *corner_id_num)
{
  double cos_theta, cos_phi, sin_theta, sin_phi, tan_theta;

  double xovr, yovr, zovr;

  double cos_pi_4 = cos(M_PI_4);
  
  cos_theta = cos(theta); sin_theta = sin(theta);
  cos_phi   = cos(phi)  ; sin_phi   = sin(phi);
  tan_theta = tan(theta);
    
  xovr = sin_theta * cos_phi;
  if(fabs(xovr)<TINY)  xovr = ((xovr >= 0.0e0) ? TINY : -TINY); 
  angle->xovr = xovr;
    
  yovr = sin_theta * sin_phi;
  if(fabs(yovr)<TINY)  yovr = ((yovr >= 0.0e0) ? TINY : -TINY); 
  angle->yovr = yovr;
    
  zovr = cos_theta;
  if(fabs(zovr)<TINY)  zovr = ((zovr >= 0.0e0) ? TINY : -TINY); 
  angle->zovr = zovr;

    
  angle->base_id   = set_base_id(cos_phi, sin_phi, tan_theta, cos_pi_4);
  angle->corner_id = set_corner_id(cos_phi, sin_phi, cos_theta, corner_id_num);
}
#endif




void sort_angle_id(struct angle_info *angle)
{
  int i,j;
  
  struct angle_info angle_temp;
  
  for(i=0; i<N_ANG-1; i++){
    for(j=N_ANG-1; j>i; j--){
      
      if(angle[j-1].corner_id > angle[j].corner_id){
	
	angle_temp.corner_id = angle[j].corner_id;
	angle[j].corner_id   = angle[j-1].corner_id;
	angle[j-1].corner_id = angle_temp.corner_id;

	angle_temp.xovr = angle[j].xovr;
	angle_temp.yovr = angle[j].yovr;
	angle_temp.zovr = angle[j].zovr;
	angle_temp.base_id = angle[j].base_id;

	angle[j].xovr   = angle[j-1].xovr;
	angle[j].yovr   = angle[j-1].yovr;
	angle[j].zovr   = angle[j-1].zovr;
	angle[j].base_id = angle[j-1].base_id;

	angle[j-1].xovr = angle_temp.xovr;
	angle[j-1].yovr = angle_temp.yovr;
	angle[j-1].zovr = angle_temp.zovr;
	angle[j-1].base_id = angle_temp.base_id;
      }

    }
  }


  for(i=0; i<N_ANG-1; i++){
    for(j=N_ANG-1; j>i; j--){
      
      if(angle[j-1].base_id > angle[j].base_id){
	if(angle[j-1].corner_id < angle[j].corner_id) continue;
	
	angle_temp.corner_id = angle[j].corner_id;
	angle[j].corner_id   = angle[j-1].corner_id;
	angle[j-1].corner_id = angle_temp.corner_id;
	
	angle_temp.xovr = angle[j].xovr;
	angle_temp.yovr = angle[j].yovr;
	angle_temp.zovr = angle[j].zovr;
	angle_temp.base_id = angle[j].base_id;

	angle[j].xovr   = angle[j-1].xovr;
	angle[j].yovr   = angle[j-1].yovr;
	angle[j].zovr   = angle[j-1].zovr;
	angle[j].base_id = angle[j-1].base_id;

	angle[j-1].xovr = angle_temp.xovr;
	angle[j-1].yovr = angle_temp.yovr;
	angle[j-1].zovr = angle_temp.zovr;
	angle[j-1].base_id = angle_temp.base_id;
      }
      
    }
  }

}
