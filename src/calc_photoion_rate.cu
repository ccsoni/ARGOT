#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <sys/times.h>

#include "constants.h"
#include "run_param.h"
#include "radiation.h"
#include "cuda_mem_space.h"

extern "C" float timing(struct tms, struct tms);
extern "C" float wallclock_timing(struct timeval, struct timeval);

__global__ void zero_out_photoion_rate_kernel(struct fluid_mesh*);
__global__ void calc_photoion_rate_kernel(struct fluid_mesh*, const struct light_ray_IO* __restrict__, const struct run_param* __restrict__, uint64_t offset);

#ifndef TINY
#define TINY (1.0e-31)
#endif

__device__
double shielding_func_H2_dev(double NH2, double tmpr) 
{
  double x, bth, fsh;

#if 0 
  // Draine & Bertoldi (1996)
  x = NH2/1.0e14;
  fsh = x < 1.0 ? 1.0 : pow(x,-0.75);
#else
  // Wolcott-Green, Haiman, Bryan (2011)
  x = NH2/5.0e14;
  bth = 0.90854e4*sqrt(tmpr); // (2*kB*T/m)^(1/2)
  bth /= 1.0e5;
  
  fsh = 0.965/pow(1.0+x/bth, 1.1) + 0.035/sqrt(1.0+x)*exp(-8.5e4*sqrt(1.0+x));
#endif

  return fsh;
}


extern "C" void zero_out_photoion_rate(struct cuda_mem_space *cuda_mem, 
				       struct cuda_param *this_cuda,
				       struct run_param *this_run)
{
  cudaError_t err;

  dim3 nblk(NMESH_LOCAL/NMESH_PER_BLOCK, 1, 1);
  dim3 nthd(NMESH_PER_BLOCK, 1, 1);

  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    zero_out_photoion_rate_kernel<<<nblk, nthd, 0, this_cuda->strm[idev]>>>
      (cuda_mem[idev].mesh_dev);
  }

  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaStreamSynchronize(this_cuda->strm[idev]);
  }
}

__global__ void zero_out_photoion_rate_kernel(struct fluid_mesh *mesh)
{
  /* zero out the Gamma */
  uint64_t tid = blockIdx.x*blockDim.x + threadIdx.x;

  mesh[tid].prev_chem.GammaHI = 0.0;
  mesh[tid].prev_chem.HeatHI  = 0.0;
#ifdef __HELIUM__
  mesh[tid].prev_chem.GammaHeI  = 0.0;
  mesh[tid].prev_chem.GammaHeII = 0.0;
  mesh[tid].prev_chem.HeatHeI   = 0.0;
  mesh[tid].prev_chem.HeatHeII  = 0.0;
#endif
#ifdef __HYDROGEN_MOL__
  mesh[tid].prev_chem.GammaHM      = 0.0;
  mesh[tid].prev_chem.GammaH2I_I   = 0.0;
  mesh[tid].prev_chem.GammaH2I_II  = 0.0;
  mesh[tid].prev_chem.GammaH2II_I  = 0.0;
  mesh[tid].prev_chem.GammaH2II_II = 0.0;
  mesh[tid].prev_chem.HeatHM       = 0.0;
  mesh[tid].prev_chem.HeatH2I_I    = 0.0;
  mesh[tid].prev_chem.HeatH2I_II   = 0.0;
  mesh[tid].prev_chem.HeatH2II_I   = 0.0;
  mesh[tid].prev_chem.HeatH2II_II  = 0.0;
#endif

}

extern "C" 
void calc_photoion_rate(struct cuda_mem_space *cuda_mem, 
			struct light_ray *ray, struct light_ray_IO *ray_IO, 
			struct cuda_param *this_cuda, struct run_param *this_run)
{
  cudaError_t err;

  struct timeval start_tv, end_tv;
  struct tms start_tms, end_tms;

#ifdef __ARGOT_PROFILE__
  times(&start_tms);
  gettimeofday(&start_tv, NULL);
#endif /* __ARGOT_PROFILE__ */

  /* Copy the light_ray data to an array of the light_ray_IO structure */
  for(uint64_t iray=0;iray<this_run->nray;iray++) {
    ray_IO[iray].src.type  = ray[iray].src.type;
    ray_IO[iray].src.param = ray[iray].src.param;
    ray_IO[iray].src.xpos  = ray[iray].src.xpos;
    ray_IO[iray].src.ypos  = ray[iray].src.ypos;
    ray_IO[iray].src.zpos  = ray[iray].src.zpos;
    for(int inu=0;inu<NGRID_NU;inu++) {
      ray_IO[iray].src.photon_rate[inu] = ray[iray].src.photon_rate[inu];
    }

    ray_IO[iray].ix_target = ray[iray].ix_target;
    ray_IO[iray].iy_target = ray[iray].iy_target;
    ray_IO[iray].iz_target = ray[iray].iz_target;

    ray_IO[iray].optical_depth_HI  = ray[iray].optical_depth_HI;
#ifdef __HELIUM__
    ray_IO[iray].optical_depth_HeI = ray[iray].optical_depth_HeI;
    ray_IO[iray].optical_depth_HeII = ray[iray].optical_depth_HeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
    ray_IO[iray].optical_depth_HM    = ray[iray].optical_depth_HM;
    ray_IO[iray].optical_depth_H2I   = ray[iray].optical_depth_H2I;
    ray_IO[iray].optical_depth_H2II  = ray[iray].optical_depth_H2II;
#endif /* __HYDROGEN_MOL__ */
  }

  /* Allocate the device memory */
  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    err = cudaMalloc((void **) &(cuda_mem[idev].ray_IO_dev), 
		     sizeof(struct light_ray_IO)*this_run->nray);
    assert(err == cudaSuccess);
  }

  /* sending the light_ray_IO data to GPUs and executing the calculations */
  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    err = cudaMemcpyAsync(cuda_mem[idev].ray_IO_dev, ray_IO, 
			  sizeof(struct light_ray_IO)*this_run->nray,
			  cudaMemcpyHostToDevice, this_cuda->strm[idev]);
    assert(err == cudaSuccess);
  }

  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);

    uint64_t nray_max_per_dev = 
#if 0
      this_cuda->max_thread_dimx*this_cuda->max_block_dimx;
#else
      NMESH_PER_BLOCK*this_cuda->max_block_dimx;
#endif
    
    uint64_t nray_waiting = this_run->nray;
    uint64_t offset = 0;

    while(nray_waiting > 0) {
      int nthread, nblock;
      
      if(nray_waiting >= nray_max_per_dev) {
	//	nthread = this_cuda->max_thread_dimx;
	nthread = NMESH_PER_BLOCK;
	nblock = this_cuda->max_block_dimx;
	//      }else if(nray_waiting >= this_cuda->max_thread_dimx){
	//	nthread = this_cuda->max_thread_dimx;
	//	nblock = nray_waiting/this_cuda->max_thread_dimx;
      }else if(nray_waiting >= NMESH_PER_BLOCK){
        nthread = NMESH_PER_BLOCK;
        nblock = nray_waiting/NMESH_PER_BLOCK;
      }else{
	nthread = nray_waiting;
	nblock = 1;
      }

      dim3 nthrd(nthread, 1, 1);
      dim3 nblk(nblock, 1, 1);

      calc_photoion_rate_kernel<<<nblk, nthrd, 0, this_cuda->strm[idev]>>>
	(cuda_mem[idev].mesh_dev, cuda_mem[idev].ray_IO_dev, 
	 cuda_mem[idev].this_run_dev,offset);

      nray_waiting -= (nblock*nthread);
      offset += nblock*nthread;

    }
  }
  
  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaStreamSynchronize(this_cuda->strm[idev]);
  }

  /* Free the allocated device memory */
  for(int idev=0;idev<this_cuda->num_cuda_dev;idev++) {
    cudaSetDevice(idev);
    cudaFree(cuda_mem[idev].ray_IO_dev);
  }

#ifdef __ARGOT_PROFILE__
    times(&end_tms);
    gettimeofday(&end_tv, NULL);

    fprintf(this_run->proc_file,
            "# calc_photoion_rate : %12.4e [sec] (CPU) / %12.4e [sec] (Wall) \n",
            timing(start_tms, end_tms), wallclock_timing(start_tv, end_tv));
    fflush(this_run->proc_file);
#endif /* __ARGOT_PROFILE__ */

}

#ifdef __PHOTON_CONSERVING__
__device__
float delta_length(float dx, float dy, float dz, 
                   const struct run_param* __restrict__ this_run)
{
  float dl, dxy;
  float sin_theta, cos_theta;
  float sin_phi,   cos_phi;
  float xovr, yovr, zovr;
  float rmin, rx, ry, rz;
  
  dl  = sqrt(NORML2(dx,dy,dz)+TINY);
  dxy = sqrt(SQR(dx)+SQR(dy)+TINY);
  
  cos_theta = dz/dl;        
  sin_theta = dxy/dl;
  
  cos_phi = dx/dxy;  
  sin_phi = dy/dxy;
  
  xovr = cos_phi*sin_theta;
  if(fabsf(xovr)<TINY) {
    xovr = (xovr >= 0.0 ? TINY : -TINY);
  }
    
  yovr = sin_phi*sin_theta;
  if(fabsf(yovr)<TINY) {
    yovr = (yovr >= 0.0 ? TINY : -TINY);
  }
  
  zovr = cos_theta;
  if(fabsf(zovr)<TINY) {
    zovr = (zovr >= 0.0 ? TINY : -TINY);
  }
  
  rx = fabsf(this_run->delta_x/xovr);
  ry = fabsf(this_run->delta_y/yovr);
  rz = fabsf(this_run->delta_z/zovr);
  rmin = fminf(rx, fminf(ry, rz));
  
  return rmin;
}
#endif

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

__global__ 
void calc_photoion_rate_kernel(struct fluid_mesh *mesh, 
			       const struct light_ray_IO* __restrict__ ray_IO,
			       const struct run_param* __restrict__ this_run,
			       uint64_t offset)
{
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x + offset;

  struct fluid_mesh *target_mesh;

  double optical_depth_HI;
#ifdef __HELIUM__
  double optical_depth_HeI, optical_depth_HeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
  double optical_depth_HM, optical_depth_H2I, optical_depth_H2II;
#endif /* __HYDROGEN_MOL__ */
#ifdef __PHOTON_CONSERVING__ 
  double delta_depth_HI;
#ifdef __HELIUM__
  double delta_depth_HeI, delta_depth_HeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
  double delta_depth_HM, delta_depth_H2I, delta_depth_H2II;
#endif /* __HYDROGEN_MOL__ */
#endif /* __PHOTON_CONSERVING__ */
  
  double GammaHI, HeatHI;
#ifdef __HELIUM__
  double GammaHeI, GammaHeII, HeatHeI, HeatHeII;
#endif
#ifdef __HYDROGEN_MOL__
  double GammaHM, GammaH2I_I, GammaH2I_II, GammaH2II_I, GammaH2II_II;
  double HeatHM, HeatH2I_I, HeatH2I_II, HeatH2II_I, HeatH2II_II;
#endif /* __HYDROGEN_MOL__ */
    
  optical_depth_HI = 
    ray_IO[tid].optical_depth_HI*(this_run->lunit*this_run->denstonh);
#ifdef __HELIUM__
  optical_depth_HeI = 
    ray_IO[tid].optical_depth_HeI*(this_run->lunit*this_run->denstonh);
  optical_depth_HeII = 
    ray_IO[tid].optical_depth_HeII*(this_run->lunit*this_run->denstonh);
#endif
#ifdef __HYDROGEN_MOL__
  optical_depth_HM =
    ray_IO[tid].optical_depth_HM*(this_run->lunit*this_run->denstonh);
  optical_depth_H2I = 
    ray_IO[tid].optical_depth_H2I*(this_run->lunit*this_run->denstonh);
  optical_depth_H2II = 
    ray_IO[tid].optical_depth_H2II*(this_run->lunit*this_run->denstonh);
#endif /* __HYDROGEN_MOL__ */
  
  int ix, iy, iz; /* local indices of the target mesh */
  float x, y, z; /* global position of the target mesh */
  float dx,dy,dz;

  ix = ray_IO[tid].ix_target-this_run->rank_x*NMESH_X_LOCAL;
  iy = ray_IO[tid].iy_target-this_run->rank_y*NMESH_Y_LOCAL;
  iz = ray_IO[tid].iz_target-this_run->rank_z*NMESH_Z_LOCAL;

  target_mesh = &(MESH(ix,iy,iz));

  x = this_run->delta_x*((float)(ray_IO[tid].ix_target)+0.5)+this_run->xmin;
  y = this_run->delta_y*((float)(ray_IO[tid].iy_target)+0.5)+this_run->ymin;
  z = this_run->delta_z*((float)(ray_IO[tid].iz_target)+0.5)+this_run->zmin;

  dx = x-ray_IO[tid].src.xpos;
  dy = y-ray_IO[tid].src.ypos;
  dz = z-ray_IO[tid].src.zpos;

  double dilution;
  double dist2;
  dist2 = NORML2(dx, dy, dz);
  dilution = (double)1.0/(4.0*PI*dist2*SQR(this_run->lunit));

#ifdef __PHOTON_CONSERVING__
  float dist,delta;
  double Vsh;
  double nH,nHI,rNHI; // number density, 1.0/atom number
#ifdef __HELIUM__
  double nHe;
  double nHeI, nHeII;
  double rNHeI, rNHeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
  double nHM, nH2I, nH2II;
  double rNHM, rNH2I, rNH2II;
#endif /* __HYDROGEN_MOL__ */

  
  /* determine ray length towards target mesh */
  delta = delta_length(dx, dy, dz, this_run);
  dist = sqrt(SQR(dx)+SQR(dy)+SQR(dz));
  
  nH = target_mesh->dens*this_run->denstonh;
  nHI = nH*target_mesh->chem.fHI;
  delta_depth_HI    = nHI*delta*(this_run->lunit);      /* like a delta optical_depth : nHI*r */
  optical_depth_HI -= 0.5*delta_depth_HI;
#ifdef __HELIUM__
  nHe   = nH*HELIUM_FACT;
  nHeI  = nHe*target_mesh->chem.fHeI;
  nHeII = nHe*target_mesh->chem.fHeII;
  delta_depth_HeI  = nHeI*delta*(this_run->lunit);
  delta_depth_HeII = nHeII*delta*(this_run->lunit);
  optical_depth_HeI  -= 0.5*delta_depth_HeI;
  optical_depth_HeII -= 0.5*delta_depth_HeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
  nHM   = nH*target_mesh->chem.fHM;
  nH2I  = nH*target_mesh->chem.fH2I;
  nH2II = nH*target_mesh->chem.fH2II;
  delta_depth_HM    = nHM*delta*(this_run->lunit);
  delta_depth_H2I   = nH2I*delta*(this_run->lunit);
  delta_depth_H2II  = nH2II*delta*(this_run->lunit);
  optical_depth_HM    -= 0.5*delta_depth_HM;
  optical_depth_H2I   -= 0.5*delta_depth_H2I;
  optical_depth_H2II  -= 0.5*delta_depth_H2II;
#endif /* __HYDROGEN_MOL__ */

  Vsh = 4.0*PI/3.0 * (CUBE(dist+0.5*delta)-CUBE(dist-0.5*delta)) * CUBE(this_run->lunit);
  rNHI = (double)1.0/(nHI*Vsh + TINY);
#ifdef __HELIUM__
  rNHeI  = (double)1.0/(nHeI*Vsh  + TINY);
  rNHeII = (double)1.0/(nHeII*Vsh + TINY);
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
  rNHM   = (double)1.0/(nHM*Vsh   + TINY);
  rNH2I  = (double)1.0/(nH2I*Vsh  + TINY);
  rNH2II = (double)1.0/(nH2II*Vsh + TINY);
#endif /* __HYDROGEN_MOL__ */

#endif /* __PHOTON_CONSERVING__ */
  
  GammaHI = 0.0;
  HeatHI  = 0.0;
#ifdef __HELIUM__
  GammaHeI = GammaHeII = 0.0;
  HeatHeI = HeatHeII = 0.0;
#endif
#ifdef __HYDROGEN_MOL__
  GammaHM = GammaH2I_I = GammaH2I_II = GammaH2II_I = GammaH2II_II = 0.0;
  HeatHM = HeatH2I_I = HeatH2I_II = HeatH2II_I = HeatH2II_II = 0.0;
#endif /* __HYDROGEN_MOL__ */
  
#ifdef __COSMOLOGICAL__
    optical_depth_HI   /= SQR(this_run->anow);
  #ifdef __HELIUM__
    optical_depth_HeI  /= SQR(this_run->anow);
    optical_depth_HeII /= SQR(this_run->anow);
  #endif /* __HELIUM__ */
  #ifdef __HYDROGEN_MOL__
    optical_depth_HM   /= SQR(this_run->anow);
    optical_depth_H2I  /= SQR(this_run->anow);
    optical_depth_H2II /= SQR(this_run->anow);
  #endif /* __HYDROGEN_MOL__ */
    
 #ifdef __PHOTON_CONSERVING__ 
    delta_depth_HI   /= SQR(this_run->anow);
  #ifdef __HELIUM__
    delta_depth_HeI  /= SQR(this_run->anow);
    delta_depth_HeII /= SQR(this_run->anow);
  #endif /* __HELIUM__ */
  #ifdef __HYDROGEN_MOL__
    delta_depth_HM   /= SQR(this_run->anow);
    delta_depth_H2I  /= SQR(this_run->anow);
    delta_depth_H2II /= SQR(this_run->anow);
  #endif /* __HYDROGEN_MOL__ */

 #else /* !__PHOTON_CONSERVING__ */
    dilution /= SQR(this_run->anow);
 #endif /* __PHOTON_CONSERVING__ */
#endif /* __COSMOLOGICAL__ */

  for(int inu=0;inu<NGRID_NU;inu++) {
    double tau;
    float nu;

    nu = this_run->freq.nu[inu];

#ifndef __PHOTON_CONSERVING__
    double photon_flux;
    tau = optical_depth_HI*this_run->csect[inu].csect_HI;
#ifdef __HELIUM__
    tau += optical_depth_HeI*this_run->csect[inu].csect_HeI;
    tau += optical_depth_HeII*this_run->csect[inu].csect_HeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
    tau += optical_depth_HM*this_run->csect[inu].csect_HM;
    tau += optical_depth_H2I*this_run->csect[inu].csect_H2I_I;
    tau += optical_depth_H2II*this_run->csect[inu].csect_H2II_I;
    tau += optical_depth_H2II*this_run->csect[inu].csect_H2II_II;
#endif /* __HYDROGEN_MOL__ */

    photon_flux = ray_IO[tid].src.photon_rate[inu]*exp(-tau)*dilution;

    GammaHI += photon_flux*this_run->csect[inu].csect_HI;
    if(nu>HI_LYMAN_LIMIT) HeatHI += photon_flux*(nu-1.0)*this_run->csect[inu].csect_HI;
#ifdef __HELIUM__
    GammaHeI += photon_flux*this_run->csect[inu].csect_HeI;
    GammaHeII += photon_flux*this_run->csect[inu].csect_HeII;
    if(nu>HeI_LYMAN_LIMIT) HeatHeI  += photon_flux*(nu-HeI_LYMAN_LIMIT)*this_run->csect[inu].csect_HeI;
    if(nu>HeII_LYMAN_LIMIT) HeatHeII += photon_flux*(nu-HeII_LYMAN_LIMIT)*this_run->csect[inu].csect_HeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
    GammaHM      += photon_flux*this_run->csect[inu].csect_HM;
    GammaH2I_I   += photon_flux*this_run->csect[inu].csect_H2I_I;
    GammaH2II_I  += photon_flux*this_run->csect[inu].csect_H2II_I;
    GammaH2II_II += photon_flux*this_run->csect[inu].csect_H2II_II;
    if(nu>5.551e-2) HeatHM       += photon_flux*(nu-5.551e-2)*this_run->csect[inu].csect_HM;
    if(nu>1.144)    HeatH2I_I    += photon_flux*(nu-1.134)*this_run->csect[inu].csect_H2I_I;
    if(nu>0.195)    HeatH2II_I   += photon_flux*(nu-0.195)*this_run->csect[inu].csect_H2II_I;
    if(nu>2.2058)   HeatH2II_II  += photon_flux*(nu-2.2058)*this_run->csect[inu].csect_H2II_II;
#endif /* __HYDROGEN_MOL__ */

    
#else /* __PHOTON_CONSERVING__ */
    double photon_abs;
    double dtau_tot, dtauHI;
#ifdef __HELIUM__
    double dtauHeI, dtauHeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
    double dtauHM, dtauH2I_I, dtauH2II_I, dtauH2II_II;
#endif /* __HYDROGEN_MOL__ */
    
    tau = optical_depth_HI*this_run->csect[inu].csect_HI;
    dtauHI = delta_depth_HI*this_run->csect[inu].csect_HI;
    dtau_tot = dtauHI;
#ifdef __HELIUM__
    tau += optical_depth_HeI*this_run->csect[inu].csect_HeI;
    tau += optical_depth_HeII*this_run->csect[inu].csect_HeII;
    dtauHeI  = delta_depth_HeI*this_run->csect[inu].csect_HeI;
    dtauHeII = delta_depth_HeII*this_run->csect[inu].csect_HeII;
    dtau_tot += dtauHeI + dtauHeII; 
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
    tau += optical_depth_HM*this_run->csect[inu].csect_HM;
    tau += optical_depth_H2I*this_run->csect[inu].csect_H2I_I;
    tau += optical_depth_H2II*this_run->csect[inu].csect_H2II_I;
    tau += optical_depth_H2II*this_run->csect[inu].csect_H2II_II;
    dtauHM      = delta_depth_HM*this_run->csect[inu].csect_HM;
    dtauH2I_I   = delta_depth_H2I*this_run->csect[inu].csect_H2I_I;
    dtauH2II_I  = delta_depth_H2II*this_run->csect[inu].csect_H2II_I;
    dtauH2II_II = delta_depth_H2II*this_run->csect[inu].csect_H2II_II;
    dtau_tot   += dtauHM + dtauH2I_I + dtauH2II_I + dtauH2II_II;
#endif /* __HYDROGEN_MOL__ */
    
    photon_abs = ray_IO[tid].src.photon_rate[inu]*exp(-tau) * (1.0e0 - exp(-dtau_tot));

    dtauHI = (double)dtauHI/(dtau_tot +TINY);
    GammaHI += dtauHI*photon_abs*rNHI;
    if(nu>HI_LYMAN_LIMIT) HeatHI += dtauHI*photon_abs*(nu-1.0)*rNHI;
#ifdef __HELIUM__
    dtauHeI  = (double)dtauHeI/(dtau_tot +TINY);
    dtauHeII = (double)dtauHeII/(dtau_tot +TINY);
    GammaHeI  += dtauHeI * photon_abs*rNHeI;
    GammaHeII += dtauHeII* photon_abs*rNHeII;
    if(nu>HeI_LYMAN_LIMIT)  HeatHeI  += dtauHeI *photon_abs*(nu-HeI_LYMAN_LIMIT)*rNHeI;
    if(nu>HeII_LYMAN_LIMIT) HeatHeII += dtauHeII*photon_abs*(nu-HeII_LYMAN_LIMIT)*rNHeII;
#endif /* __HELIUM__ */
#ifdef __HYDROGEN_MOL__
    dtauHM      /= (dtau_tot+TINY);
    dtauH2I_I   /= (dtau_tot+TINY);
    dtauH2II_I  /= (dtau_tot+TINY);
    dtauH2II_II /= (dtau_tot+TINY);
    GammaHM      += dtauHM*photon_abs*rNHM;
    GammaH2I_I   += dtauH2I_I*photon_abs*rNH2I;
    GammaH2II_I  += dtauH2II_I*photon_abs*rNH2II;
    GammaH2II_II += dtauH2II_II*photon_abs*rNH2II;
    if(nu>5.551e-2) HeatHM      += dtauHM*photon_abs*(nu-5.551e-2)*rNHM;
    if(nu>1.144)    HeatH2I_I   += dtauH2I_I*photon_abs*(nu-1.134)*rNH2I;
    if(nu>0.195)    HeatH2II_I  += dtauH2II_I*photon_abs*(nu-0.195)*rNH2II;
    if(nu>2.2058)   HeatH2II_II += dtauH2II_II*photon_abs*(nu-2.2058)*rNH2II;
#endif /* __HYDROGEN_MOL__ */
#endif /* __PHOTON_CONSERVING__ */
  }

#ifdef __HYDROGEN_MOL__
  /* H2 photo-dissociation in the LW band */
  int indx_freq_LW_band=-1;
  double nu_LW_band = 12.87/13.6; /* in units of nuL */
  double delta_nu_LW; /* in units of Hz */
  for(int indx=0;indx<NGRID_NU-1;indx++) {
    double nu_lo = this_run->freq.nu[indx]/pow(10.0,0.5*this_run->freq.dlog_nu);
    double nu_hi = this_run->freq.nu[indx]*pow(10.0,0.5*this_run->freq.dlog_nu);
    if(nu_lo < nu_LW_band && nu_hi > nu_LW_band) {
      indx_freq_LW_band = indx;
      delta_nu_LW = (nu_hi-nu_lo)*nuL;
    }
  }
  assert(indx_freq_LW_band >= 0);
  assert(indx_freq_LW_band < NGRID_NU-2);
  
  double tau_LW = 
    optical_depth_HM*this_run->csect[indx_freq_LW_band].csect_HM + 
    optical_depth_H2I*this_run->csect[indx_freq_LW_band].csect_H2I_I +
    optical_depth_H2II*this_run->csect[indx_freq_LW_band].csect_H2II_I +
    optical_depth_H2II*this_run->csect[indx_freq_LW_band].csect_H2II_II;
  double photon_flux = 
    ray_IO[tid].src.photon_rate[indx_freq_LW_band]*exp(-tau_LW)*dilution;
  double jnu = photon_flux*(hplanck*nu_LW_band*nuL)/delta_nu_LW;
  GammaH2I_II = 1.13e8*jnu*shielding_func_H2_dev(optical_depth_H2I, 1.0e2);
  HeatH2I_II = 6.4e-13*GammaH2I_II;
#endif /* __HYDROGEN_MOL__ */
  
  atomicAdd(&target_mesh->prev_chem.GammaHI, GammaHI);
  atomicAdd(&target_mesh->prev_chem.HeatHI, HeatHI*(hplanck*nuL));
#ifdef __HELIUM__
  atomicAdd(&target_mesh->prev_chem.GammaHeI, GammaHeI);
  atomicAdd(&target_mesh->prev_chem.HeatHeI, HeatHeI*(hplanck*nuL));
  atomicAdd(&target_mesh->prev_chem.GammaHeII, GammaHeII);
  atomicAdd(&target_mesh->prev_chem.HeatHeII, HeatHeII*(hplanck*nuL));
#endif
#ifdef __HYDROGEN_MOL__
  atomicAdd(&target_mesh->prev_chem.GammaHM,      GammaHM);
  atomicAdd(&target_mesh->prev_chem.GammaH2I_I,   GammaH2I_I);
  atomicAdd(&target_mesh->prev_chem.GammaH2I_II,  GammaH2I_II);
  atomicAdd(&target_mesh->prev_chem.GammaH2II_I,  GammaH2II_I);
  atomicAdd(&target_mesh->prev_chem.GammaH2II_II, GammaH2II_II);

  atomicAdd(&target_mesh->prev_chem.HeatHM,      HeatHM*(hplanck*nuL));
  atomicAdd(&target_mesh->prev_chem.HeatH2I_I,   HeatH2I_I*(hplanck*nuL));
  atomicAdd(&target_mesh->prev_chem.HeatH2I_II,  HeatH2I_II); // hplanck*nuL is unnecessary
  atomicAdd(&target_mesh->prev_chem.HeatH2II_I,  HeatH2II_I*(hplanck*nuL));
  atomicAdd(&target_mesh->prev_chem.HeatH2II_II, HeatH2II_II*(hplanck*nuL));
#endif /* __HYDROGEN_MOL__ */

}
