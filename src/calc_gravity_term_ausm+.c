#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "run_param.h"
#include "fluid.h"
#include "chemistry.h"
#include "prototype.h"

#define __4PFDA__
#define alpha_4PFDA (1.33333333333)

#define MESH(ix,iy,iz)   (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])
#define MESH_1(ix,iy,iz) (mesh1[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])
#define MESH_2(ix,iy,iz) (mesh2[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

void set_pad_x(struct pad_region*, struct fluid_mesh*,
	       struct fluid_mesh**, struct fluid_mesh**, struct fluid_mesh**,
	       struct fluid_mesh**, int, int, int, struct run_param*);
void set_pad_y(struct pad_region*, struct fluid_mesh*,
	       struct fluid_mesh**, struct fluid_mesh**, struct fluid_mesh**,
	       struct fluid_mesh**, int, int, int, struct run_param*);
void set_pad_z(struct pad_region*, struct fluid_mesh*,
	       struct fluid_mesh**, struct fluid_mesh**, struct fluid_mesh**,
	       struct fluid_mesh**, int, int, int, struct run_param*);

void calc_fluid_flux(struct fluid_mesh *mesh_im2, struct fluid_mesh *mesh_im1,
                     struct fluid_mesh *mesh_i,
                     struct fluid_mesh *mesh_ip1, struct fluid_mesh *mesh_ip2,
                     float *flux_plus, float *flux_minus,
                     struct run_param  *this_run, int axis)
{
  /* axis == 0 : x-axis */
  /* axis == 1 : y-axis */
  /* axis == 2 : z-axis */

  float pres_im2,pres_im1,pres_i,pres_ip1,pres_ip2;
  float velx_im2,velx_im1,velx_i,velx_ip1,velx_ip2;
  float vely_im2,vely_im1,vely_i,vely_ip1,vely_ip2;
  float velz_im2,velz_im1,velz_i,velz_ip1,velz_ip2;
  //  float enth_im2,enth_im1,enth_i,enth_ip1,enth_ip2;
  float cs;

  float Mach_L,pres_L,dens_L,velx_L,vely_L,velz_L;
  float enth_L,cs_L,eneg_L;
  float Mach_R,pres_R,dens_R,velx_R,vely_R,velz_R;
  float enth_R,cs_R,eneg_R;

  float Mach_imh,Mach_iph,pres_imh,pres_iph;

  float gamma_im2, gamma_im1, gamma_i, gamma_ip1, gamma_ip2;
  float gamm1_im2, gamm1_im1, gamm1_i, gamm1_ip1, gamm1_ip2;
  float gamma_L, gamma_R, gamm1_L, gamm1_R;

  float coeff_L, coeff_R;
  float cs2_L, cs2_R;
  float vel_L, vel_R;

  /* adabatic index */
  gamma_im2 = gamma_total(mesh_im2, this_run); gamm1_im2 = gamma_im2-1.0;
  gamma_im1 = gamma_total(mesh_im1, this_run); gamm1_im1 = gamma_im1-1.0;
  gamma_i   = gamma_total(mesh_i  , this_run); gamm1_i   = gamma_i  -1.0;
  gamma_ip1 = gamma_total(mesh_ip1, this_run); gamm1_ip1 = gamma_ip1-1.0;
  gamma_ip2 = gamma_total(mesh_ip2, this_run); gamm1_ip2 = gamma_ip2-1.0;

  /*-- pressure --*/
  pres_im2 = fmaxf(gamm1_im2*mesh_im2->uene*mesh_im2->dens, FLUID_TINY);
  pres_im1 = fmaxf(gamm1_im1*mesh_im1->uene*mesh_im1->dens, FLUID_TINY);
  pres_i   = fmaxf(gamm1_i  *mesh_i->uene*mesh_i->dens    , FLUID_TINY);
  pres_ip1 = fmaxf(gamm1_ip1*mesh_ip1->uene*mesh_ip1->dens, FLUID_TINY);
  pres_ip2 = fmaxf(gamm1_ip2*mesh_ip2->uene*mesh_ip2->dens, FLUID_TINY);

  /*-- velocity --*/
  velx_im2 = mesh_im2->momx/mesh_im2->dens;
  velx_im1 = mesh_im1->momx/mesh_im1->dens;
  velx_i   = mesh_i->momx/mesh_i->dens;
  velx_ip1 = mesh_ip1->momx/mesh_ip1->dens;
  velx_ip2 = mesh_ip2->momx/mesh_ip2->dens;

  vely_im2 = mesh_im2->momy/mesh_im2->dens;
  vely_im1 = mesh_im1->momy/mesh_im1->dens;
  vely_i   = mesh_i->momy/mesh_i->dens;
  vely_ip1 = mesh_ip1->momy/mesh_ip1->dens;
  vely_ip2 = mesh_ip2->momy/mesh_ip2->dens;

  velz_im2 = mesh_im2->momz/mesh_im2->dens;
  velz_im1 = mesh_im1->momz/mesh_im1->dens;
  velz_i   = mesh_i->momz/mesh_i->dens;
  velz_ip1 = mesh_ip1->momz/mesh_ip1->dens;
  velz_ip2 = mesh_ip2->momz/mesh_ip2->dens;

  /*-- numerical flux at i+1/2 --*/
  /* left values */
  dens_L = MUSCL_L(mesh_im1->dens, mesh_i->dens, mesh_ip1->dens);
  velx_L = MUSCL_L(velx_im1,       velx_i,       velx_ip1);
  vely_L = MUSCL_L(vely_im1,       vely_i,       vely_ip1);
  velz_L = MUSCL_L(velz_im1,       velz_i,       velz_ip1);
  pres_L = MUSCL_L(pres_im1,       pres_i,       pres_ip1);

  gamma_L = MUSCL_L(gamma_im1,gamma_i,gamma_ip1);
  gamm1_L = gamma_L-1.0;

  eneg_L = 0.5*dens_L*NORML2(velx_L,vely_L,velz_L) + pres_L/gamm1_L;
  enth_L = (eneg_L + pres_L)/dens_L;

  /* right values */
  dens_R = MUSCL_R(mesh_i->dens, mesh_ip1->dens, mesh_ip2->dens);
  velx_R = MUSCL_R(velx_i,       velx_ip1,       velx_ip2);
  vely_R = MUSCL_R(vely_i,       vely_ip1,       vely_ip2);
  velz_R = MUSCL_R(velz_i,       velz_ip1,       velz_ip2);
  pres_R = MUSCL_R(pres_i,       pres_ip1,       pres_ip2);

  gamma_R = MUSCL_R(gamma_i,gamma_ip1,gamma_ip2);
  gamm1_R = gamma_R-1.0;

  eneg_R = 0.5*dens_R*NORML2(velx_R,vely_R,velz_R) + pres_R/gamm1_R;
  enth_R = (eneg_R + pres_R)/dens_R;

  /* sound speed at interface */ 
  coeff_L = 2.0*gamm1_L/(gamma_L+1.0);
  coeff_R = 2.0*gamm1_R/(gamma_R+1.0);

  if(axis==0) {
    vel_L = velx_L;  
    vel_R = velx_R; 
  } else if(axis==1) {
    vel_L = vely_L;  
    vel_R = vely_R; 
  } else {
    vel_L = velz_L;  
    vel_R = velz_R; 
  }

  cs2_L = coeff_L*enth_L;
  cs_L = cs2_L/fmaxf(sqrtf(cs2_L), fabsf(vel_L));

  cs2_R = coeff_R*enth_R;
  cs_R = cs2_R/fmaxf(sqrtf(cs2_R), fabsf(vel_R));

  cs = fminf(cs_L, cs_R);

  /* Mach number */ 
  Mach_L = vel_L/cs;
  Mach_R = vel_R/cs;
  Mach_iph = M_p(Mach_L) + M_m(Mach_R);
  pres_iph = P_p(Mach_L)*pres_L + P_m(Mach_R)*pres_R;

  /* numerical flux */
  if(Mach_iph >= 0.0) {
    *flux_plus = cs*Mach_iph*dens_L;
  }else{
    *flux_plus = -cs*Mach_iph*dens_R;
  }

  /*-- numerical flux at i-1/2 --*/
  /* left values */
  dens_L = MUSCL_L(mesh_im2->dens, mesh_im1->dens, mesh_i->dens);
  velx_L = MUSCL_L(velx_im2      , velx_im1      , velx_i      );
  vely_L = MUSCL_L(vely_im2      , vely_im1      , vely_i      );
  velz_L = MUSCL_L(velz_im2      , velz_im1      , velz_i      );
  pres_L = MUSCL_L(pres_im2      , pres_im1      , pres_i      );

  gamma_L = MUSCL_L(gamma_im2,gamma_im1,gamma_i);
  gamm1_L = gamma_L-1.0;

  eneg_L = 0.5*dens_L*NORML2(velx_L,vely_L,velz_L)+pres_L/gamm1_L;
  enth_L = (eneg_L + pres_L)/dens_L;

  /* right values */
  dens_R = MUSCL_R(mesh_im1->dens, mesh_i->dens, mesh_ip1->dens);
  velx_R = MUSCL_R(velx_im1      , velx_i      , velx_ip1      );
  vely_R = MUSCL_R(vely_im1      , vely_i      , vely_ip1      );
  velz_R = MUSCL_R(velz_im1      , velz_i      , velz_ip1      );
  pres_R = MUSCL_R(pres_im1      , pres_i      , pres_ip1      );

  gamma_R = MUSCL_R(gamma_im1,gamma_i,gamma_ip1);
  gamm1_R = gamma_R-1.0;

  eneg_R = 0.5*dens_R*NORML2(velx_R,vely_R,velz_R) + pres_R/gamm1_R;
  enth_R = (eneg_R + pres_R)/dens_R;

  /* sound speed at interface */
  coeff_L = 2.0*gamm1_L/(gamma_L+1.0);
  coeff_R = 2.0*gamm1_R/(gamma_R+1.0);

  if(axis==0) {
    vel_L = velx_L;  
    vel_R = velx_R; 
  } else if(axis==1) {
    vel_L = vely_L;  
    vel_R = vely_R; 
  } else {
    vel_L = velz_L;  
    vel_R = velz_R; 
  }

  cs2_L = coeff_L*enth_L;
  cs_L = cs2_L/fmaxf(sqrtf(cs2_L), fabsf(vel_L));

  cs2_R = coeff_R*enth_R;
  cs_R = cs2_R/fmaxf(sqrtf(cs2_R), fabsf(vel_R));

  cs = fminf(cs_L, cs_R);

  /* Mach number */ 
  Mach_L = vel_L/cs;
  Mach_R = vel_R/cs;
  Mach_imh = M_p(Mach_L) + M_m(Mach_R);
  pres_imh = P_p(Mach_L)*pres_L + P_m(Mach_R)*pres_R;

  /* numerical flux */
  if(Mach_imh >= 0.0) {
    *flux_minus = cs*Mach_imh*dens_L;
  }else{
    *flux_minus = -cs*Mach_imh*dens_R;
  }

}

inline float grad_pot_x(float pot_im2, float pot_im1, float pot_i,
			float pot_ip1, float pot_ip2,
			int ix, struct run_param *this_run)
{
  float grad_pot;

#ifndef __4PFDA__

  grad_pot = (pot_ip1 - pot_im1)/(2.0*this_run->delta_x);

  /* ix==0 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_X_LO)
  if(this_run->rank_x==0 && ix==0)
    grad_pot = (pot_ip1 - pot_i)/this_run->delta_x;
#endif
  /* ix==gix-1 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_X_HI)
  if(this_run->rank_x==NNODE_X-1 && ix==NMESH_X_LOCAL-1)
    grad_pot = (pot_i - pot_im1)/this_run->delta_x;
#endif
  
#else /*__4PFDA__*/

  grad_pot = alpha_4PFDA*(pot_ip1 - pot_im1)/(2.0*this_run->delta_x) 
    + (1.0 - alpha_4PFDA)*(pot_ip2 - pot_im2)/(4.0*this_run->delta_x);

  /* ix==0 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_X_LO)
  if(this_run->rank_x==0 && ix==0)
    grad_pot = alpha_4PFDA*(pot_ip1 - pot_i)/this_run->delta_x 
      + (1.0 - alpha_4PFDA)*(pot_ip2 - pot_i)/(2.0*this_run->delta_x);
#endif
  /* ix==gix-1 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_X_HI)
  if(this_run->rank_x==NNODE_X-1 && ix==NMESH_X_LOCAL-1)
    grad_pot = alpha_4PFDA*(pot_i - pot_im1)/this_run->delta_x 
      + (1.0 - alpha_4PFDA)*(pot_i - pot_im2)/(2.0*this_run->delta_x);
#endif

  /* ix==1 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_X_LO)
  if(this_run->rank_x==0 && ix==1)
    grad_pot = alpha_4PFDA*(pot_ip1 - pot_im1)/(2.0*this_run->delta_x) 
      + (1.0 - alpha_4PFDA)*(pot_ip2 - pot_im1)/(3.0*this_run->delta_x);
#endif
  /* ix==gix-2 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_X_HI)
  if(this_run->rank_x==NNODE_X-1 && ix==NMESH_X_LOCAL-2)
    grad_pot = alpha_4PFDA*(pot_ip1 - pot_im1)/(2.0*this_run->delta_x) 
      + (1.0 - alpha_4PFDA)*(pot_ip1 - pot_im2)/(3.0*this_run->delta_x);
#endif
#endif /*__4PFDA__*/
  
  return grad_pot;
}

inline float grad_pot_y(float pot_im2, float pot_im1, float pot_i,
			float pot_ip1, float pot_ip2,
			int iy, struct run_param *this_run)
{
  float grad_pot;

#ifndef __4PFDA__

  grad_pot = (pot_ip1 - pot_im1)/(2.0*this_run->delta_y);

  /* iy==0 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Y_LO)
  if(this_run->rank_y==0 && iy==0)
    grad_pot = (pot_ip1 - pot_i)/this_run->delta_y;
#endif
  /* iy==giy-1 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Y_HI)
  if(this_run->rank_y==NNODE_Y-1 && iy==NMESH_Y_LOCAL-1)
    grad_pot = (pot_i - pot_im1)/this_run->delta_y;
#endif
  
#else /*__4PFDA__*/

  grad_pot = alpha_4PFDA*(pot_ip1 - pot_im1)/(2.0*this_run->delta_y) 
    + (1.0 - alpha_4PFDA)*(pot_ip2 - pot_im2)/(4.0*this_run->delta_y);

  /* iy==0 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Y_LO)
  if(this_run->rank_y==0 && iy==0)
    grad_pot = alpha_4PFDA*(pot_ip1 - pot_i)/this_run->delta_y 
      + (1.0 - alpha_4PFDA)*(pot_ip2 - pot_i)/(2.0*this_run->delta_y);
#endif
  /* iy==giy-1 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Y_HI)
  if(this_run->rank_y==NNODE_Y-1 && iy==NMESH_Y_LOCAL-1)
    grad_pot = alpha_4PFDA*(pot_i - pot_im1)/this_run->delta_y 
      + (1.0 - alpha_4PFDA)*(pot_i - pot_im2)/(2.0*this_run->delta_y);
#endif

  /* iy==1 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Y_LO)
  if(this_run->rank_y==0 && iy==1)
    grad_pot = alpha_4PFDA*(pot_ip1 - pot_im1)/(2.0*this_run->delta_y) 
      + (1.0 - alpha_4PFDA)*(pot_ip2 - pot_im1)/(3.0*this_run->delta_y);
#endif
  /* iy==giy-2 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Y_HI)
  if(this_run->rank_y==NNODE_Y-1 && iy==NMESH_Y_LOCAL-2)
    grad_pot = alpha_4PFDA*(pot_ip1 - pot_im1)/(2.0*this_run->delta_y) 
      + (1.0 - alpha_4PFDA)*(pot_ip1 - pot_im2)/(3.0*this_run->delta_y);
#endif 
#endif /*__4PFDA__*/
  
  return grad_pot;
}

inline float grad_pot_z(float pot_im2, float pot_im1, float pot_i,
			float pot_ip1, float pot_ip2,
			int iz, struct run_param *this_run)
{
  float grad_pot;

#ifndef __4PFDA__

  grad_pot = (pot_ip1 - pot_im1)/(2.0*this_run->delta_z);

  /* iz==0 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Z_LO)
  if(this_run->rank_z==0 && iz==0)
    grad_pot = (pot_ip1 - pot_i)/this_run->delta_z;
#endif
  /* iz==giz-1 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Z_HI)
  if(this_run->rank_z==NNODE_Z-1 && iz==NMESH_Z_LOCAL-1)
    grad_pot = (pot_i - pot_im1)/this_run->delta_z;
#endif
  
#else /*__4PFDA__*/

  grad_pot = alpha_4PFDA*(pot_ip1 - pot_im1)/(2.0*this_run->delta_z) 
    + (1.0 - alpha_4PFDA)*(pot_ip2 - pot_im2)/(4.0*this_run->delta_z);

  /* iz==0 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Z_LO)
  if(this_run->rank_z==0 && iz==0)
    grad_pot = alpha_4PFDA*(pot_ip1 - pot_i)/this_run->delta_z 
      + (1.0 - alpha_4PFDA)*(pot_ip2 - pot_i)/(2.0*this_run->delta_z);
#endif
  /* iz==giz-1 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Z_HI)
  if(this_run->rank_z==NNODE_Z-1 && iz==NMESH_Z_LOCAL-1)
    grad_pot = alpha_4PFDA*(pot_i - pot_im1)/this_run->delta_z 
      + (1.0 - alpha_4PFDA)*(pot_i - pot_im2)/(2.0*this_run->delta_z);
#endif
  /* iz==1 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Z_LO)
  if(this_run->rank_z==0 && iz==1)
    grad_pot = alpha_4PFDA*(pot_ip1 - pot_im1)/(2.0*this_run->delta_z) 
      + (1.0 - alpha_4PFDA)*(pot_ip2 - pot_im1)/(3.0*this_run->delta_z);
#endif
  /* iz==giz-2 */
#if defined(__ISOLATED__) && defined(OUTFLOW_BOUNDARY_Z_HI)
  if(this_run->rank_z==NNODE_Z-1 && iz==NMESH_Z_LOCAL-2)
    grad_pot = alpha_4PFDA*(pot_ip1 - pot_im1)/(2.0*this_run->delta_z) 
      + (1.0 - alpha_4PFDA)*(pot_ip1 - pot_im2)/(3.0*this_run->delta_z);
#endif
#endif /*__4PFDA__*/
  
  return grad_pot;
}


float calc_grad_pot_x(struct fluid_mesh *mesh, struct pad_region *pad, 
                      int ix, int iy, int iz, 
                      struct run_param *this_run)
{
  struct fluid_mesh *mesh_im2,*mesh_im1,*mesh_i,*mesh_ip1,*mesh_ip2;
  float grad_pot;

  mesh_i = &MESH(ix,iy,iz);
  
  set_pad_x(pad, mesh, &mesh_im2, &mesh_im1, &mesh_ip1, &mesh_ip2,
	    ix, iy, iz, this_run);

  grad_pot = grad_pot_x(mesh_im2->pot, mesh_im1->pot, mesh_i->pot,
			mesh_ip1->pot, mesh_ip2->pot ,ix ,this_run);
  
  return grad_pot;
}

float calc_grad_pot_y(struct fluid_mesh *mesh, struct pad_region *pad, 
                      int ix, int iy, int iz, 
                      struct run_param *this_run)
{
  struct fluid_mesh *mesh_im2,*mesh_im1,*mesh_i,*mesh_ip1,*mesh_ip2;
  float grad_pot;

  mesh_i = &MESH(ix,iy,iz);
  
  set_pad_y(pad, mesh, &mesh_im2, &mesh_im1, &mesh_ip1, &mesh_ip2,
	    ix, iy, iz, this_run);

  grad_pot = grad_pot_y(mesh_im2->pot, mesh_im1->pot, mesh_i->pot,
			mesh_ip1->pot, mesh_ip2->pot ,iy ,this_run);
  
  return grad_pot;
}

float calc_grad_pot_z(struct fluid_mesh *mesh, struct pad_region *pad,
                      int ix, int iy, int iz, 
                      struct run_param *this_run)
{
  struct fluid_mesh *mesh_im2,*mesh_im1,*mesh_i,*mesh_ip1,*mesh_ip2;
  float grad_pot;
  
  mesh_i = &MESH(ix,iy,iz);
  
  set_pad_z(pad, mesh, &mesh_im2, &mesh_im1, &mesh_ip1, &mesh_ip2,
	    ix, iy, iz, this_run);

  grad_pot = grad_pot_z(mesh_im2->pot, mesh_im1->pot, mesh_i->pot,
			mesh_ip1->pot, mesh_ip2->pot ,iz ,this_run);
  
  return grad_pot;
}

float calc_grav_ene_x(struct fluid_mesh *mesh, struct pad_region *pad,
                      int ix, int iy, int iz, 
                      struct run_param *this_run)
{
  struct fluid_mesh *mesh_im2,*mesh_im1,*mesh_i,*mesh_ip1,*mesh_ip2;
  float flux_plus,flux_minus;
  float plus_gene, minus_gene;
  float area = this_run->delta_y*this_run->delta_z; 

  mesh_i = &MESH(ix,iy,iz);
  
  set_pad_x(pad, mesh, &mesh_im2, &mesh_im1, &mesh_ip1, &mesh_ip2,
	    ix, iy, iz, this_run);
  
  calc_fluid_flux(mesh_im2, mesh_im1, mesh_i, mesh_ip1, mesh_ip2,
                  &flux_plus, &flux_minus, this_run, 0);

  plus_gene  = flux_plus *(mesh_i->pot - mesh_ip1->pot);
  minus_gene = flux_minus*(mesh_i->pot - mesh_im1->pot); 

  return (area*(plus_gene+minus_gene));
}

float calc_grav_ene_y(struct fluid_mesh *mesh, struct pad_region *pad,
                      int ix, int iy, int iz, 
                      struct run_param *this_run)
{
  struct fluid_mesh *mesh_im2,*mesh_im1,*mesh_i,*mesh_ip1,*mesh_ip2;
  float flux_plus,flux_minus;
  float plus_gene, minus_gene;
  float area = this_run->delta_z*this_run->delta_x; 

  mesh_i = &MESH(ix,iy,iz);
  
  set_pad_y(pad, mesh, &mesh_im2, &mesh_im1, &mesh_ip1, &mesh_ip2,
	    ix, iy, iz, this_run);
  
  calc_fluid_flux(mesh_im2, mesh_im1, mesh_i, mesh_ip1, mesh_ip2,
                  &flux_plus, &flux_minus, this_run, 1);

  plus_gene  = flux_plus *(mesh_i->pot - mesh_ip1->pot); 
  minus_gene = flux_minus*(mesh_i->pot - mesh_im1->pot); 

  return (area*(plus_gene+minus_gene));
}

float calc_grav_ene_z(struct fluid_mesh *mesh, struct pad_region *pad,
                      int ix, int iy, int iz, 
                      struct run_param *this_run)
{
  struct fluid_mesh *mesh_im2,*mesh_im1,*mesh_i,*mesh_ip1,*mesh_ip2;
  float flux_plus,flux_minus;
  float plus_gene, minus_gene;
  float area = this_run->delta_x*this_run->delta_y; 

  mesh_i = &MESH(ix,iy,iz);
  
  set_pad_z(pad, mesh, &mesh_im2, &mesh_im1, &mesh_ip1, &mesh_ip2,
	    ix, iy, iz, this_run);
  
  calc_fluid_flux(mesh_im2, mesh_im1, mesh_i, mesh_ip1, mesh_ip2,
                  &flux_plus, &flux_minus, this_run, 2);

  plus_gene  = flux_plus *(mesh_i->pot - mesh_ip1->pot); 
  minus_gene = flux_minus*(mesh_i->pot - mesh_im1->pot); 

  return (area*(plus_gene+minus_gene));
}


#define GRAV(ix,iy,iz) (grav[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

struct add_grav{
  float momx;
  float momy;
  float momz;
  float eneg;
};


void add_gravity_term(struct fluid_mesh *mesh1, struct fluid_mesh *mesh2,
		      struct pad_region *pad1, struct pad_region *pad2,
		      struct run_param *this_run, struct mpi_param *this_mpi, 
		      float dtime, int stage)
{
  struct add_grav *grav;
  grav = (struct add_grav*)malloc(sizeof(struct add_grav)*NMESH_LOCAL);

  float coeff;
  if(stage == 0) { // first stage  : mesh1 = mesh(t=t^n) / mesh2 = mesh_mid
    coeff = 1.0;
  } else {         // second stage : mesh1 = mesh_mid / mesh2 = mesh(t=t^{n+1})
    coeff = 0.5;
  }
  
#pragma omp parallel for schedule(auto)
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {
        
        float grad_pot_x1, grad_pot_y1, grad_pot_z1;
        float grad_pot_x2, grad_pot_y2, grad_pot_z2;

	float therm_eneg, kin_eneg, grav_eneg;
	
        struct fluid_mesh *target_mesh, *dest_mesh;
        target_mesh = &MESH_1(ix,iy,iz);
        dest_mesh   = &MESH_2(ix,iy,iz);

        grad_pot_x1 = calc_grad_pot_x(mesh1, pad1, ix, iy, iz, this_run);
        grad_pot_y1 = calc_grad_pot_y(mesh1, pad1, ix, iy, iz, this_run);
        grad_pot_z1 = calc_grad_pot_z(mesh1, pad1, ix, iy, iz, this_run);
        
        grad_pot_x2 = calc_grad_pot_x(mesh2, pad2, ix, iy, iz, this_run);
        grad_pot_y2 = calc_grad_pot_y(mesh2, pad2, ix, iy, iz, this_run);
        grad_pot_z2 = calc_grad_pot_z(mesh2, pad2, ix, iy, iz, this_run);
		
	// add gravity term	
	GRAV(ix,iy,iz).momx = dest_mesh->momx
	  - 0.5*coeff*dtime*(target_mesh->dens*grad_pot_x1 + dest_mesh->dens*grad_pot_x2);
        GRAV(ix,iy,iz).momy = dest_mesh->momy
	  - 0.5*coeff*dtime*(target_mesh->dens*grad_pot_y1 + dest_mesh->dens*grad_pot_y2);
        GRAV(ix,iy,iz).momz = dest_mesh->momz
	  - 0.5*coeff*dtime*(target_mesh->dens*grad_pot_z1 + dest_mesh->dens*grad_pot_z2);
	
	therm_eneg = dest_mesh->uene*dest_mesh->dens;
	
        kin_eneg = 0.5*NORML2(GRAV(ix,iy,iz).momx,
                              GRAV(ix,iy,iz).momy,
                              GRAV(ix,iy,iz).momz)/dest_mesh->dens;
	
#if 1 /* springel */
	float gene = 0.0;
	gene  = calc_grav_ene_x(mesh1, pad1, ix, iy, iz, this_run)
	  + calc_grav_ene_x(mesh2, pad2, ix, iy, iz, this_run);
        gene += calc_grav_ene_y(mesh1, pad1, ix, iy, iz, this_run)
	  + calc_grav_ene_y(mesh2, pad2, ix, iy, iz, this_run);
        gene += calc_grav_ene_z(mesh1, pad1, ix, iy, iz, this_run)
	  + calc_grav_ene_z(mesh2, pad2, ix, iy, iz, this_run);

	grav_eneg = - 0.5*coeff*dtime*gene;
	
	GRAV(ix,iy,iz).eneg = therm_eneg + kin_eneg + grav_eneg;
	
#else /* truelove */
	GRAV(ix,iy,iz).eneg = therm_eneg + kin_eneg;
#endif
		
      }
    }
  }

  
#pragma omp parallel for schedule(auto)
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {

	MESH_2(ix,iy,iz).momx = GRAV(ix,iy,iz).momx;
	MESH_2(ix,iy,iz).momy = GRAV(ix,iy,iz).momy;
	MESH_2(ix,iy,iz).momz = GRAV(ix,iy,iz).momz;
	MESH_2(ix,iy,iz).eneg = GRAV(ix,iy,iz).eneg;
	
      }
    }
  }
   
  free(grav);
}

#undef GRAV

