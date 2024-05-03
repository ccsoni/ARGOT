#ifndef __DIFFUSE_PHOTON__
#define __DIFFUSE_PHOTON__

#include "run_param.h"
#include "chemistry.h"
#include "constants.h"
#include "fluid.h"

#ifndef __CUDACC__
#include "../mpi_param.h"
#endif

#include <math.h>

#define NMESH_XY_LOCAL (NMESH_X_LOCAL*NMESH_Y_LOCAL)
#define NMESH_YZ_LOCAL (NMESH_Y_LOCAL*NMESH_Z_LOCAL)
#define NMESH_ZX_LOCAL (NMESH_Z_LOCAL*NMESH_X_LOCAL)

#define NMESH_MAX_FACE MAX(NMESH_XY_LOCAL,MAX(NMESH_YZ_LOCAL,NMESH_ZX_LOCAL))

///12 * 4^l * N_TOTAL^2 = 12*SQR((2^l)*N_TOTAL)
//#ifndef ANG_LEVEL
//#define ANG_LEVEL (-6)  // value set in ../run_param.h
//#endif
#define N_MESH_T  MAX(NMESH_X_TOTAL,MAX(NMESH_Y_TOTAL,NMESH_Z_TOTAL))
//#define N_SIDE    ((long)(pow(2.0e0, ANG_LEVEL)*(N_MESH_T)))         // >>ANG_LEVEL: ANG_LEVEL is angle resolution level.
///#define N_SIDE    (N_MESH_T  << ANG_LEVEL)         // negative shift operation depends on compiler
#define N_ANG     (12 * SQR(N_SIDE))

//ray group number
#define RAY_GROUP_NUM (4)

//step function range
#ifdef __HEATCOOL__
#define STEP_FUNC_RANGE (0.0632353)    // 13.6 eV -> 1.0 (nu0 unit): T=10000K , 0.8617 eV -> 0.0632353 (nu0 unit)
#define NBIN_STEP_NU (64)
//#define NBIN_STEP_NU (NGRID_NU)  //run_param.h
#else /* !__HEATCOOL__ */
#define STEP_FUNC_RANGE (0.0)
#define NBIN_STEP_NU (1)
#endif


struct angle_info {           
  float xovr, yovr, zovr;
  short base_id, corner_id;
};


struct ray_info { 
  float x, y, z;
  float I_inHI;
#ifdef __HELIUM__
  float I_inHeI, I_inHeII;
#endif //__HELIUM__
};


struct ray_face{ 
  struct ray_info *xy;
  struct ray_info *yz;
  struct ray_info *zx;
};


struct radiation_mesh {
  float length;
  float IHI;
#ifdef __HELIUM__
  float IHeI, IHeII;
#endif //__HELIUM__
  
  float absorptionHI, source_funcHI;
#ifdef __HELIUM__
  float absorptionHeI,  source_funcHeI;
  float absorptionHeII, source_funcHeII;
#endif //__HELIUM__
  
  float GHI_tot, HHI_tot;
#ifdef __HELIUM__
  float GHeI_tot, HHeI_tot;
  float GHeII_tot, HHeII_tot;
#endif //__HELIUM__
};


struct step_func_factor {
  /// 0: ionization , 1: heating
  /// 0: I_seg      , 1: source_func
  float HI[2][2];
#ifdef __HELIUM__
  float HeI[2][2];
  float HeII[2][2];
#endif /* __HELIUM__ */
};


struct host_diffuse_param {
  int corner_id_num[8];
  struct step_func_factor *step_fact; 
  struct angle_info *angle;             //[N_ANG]
#ifndef __USE_GPU__
  struct radiation_mesh *rmesh;         //[NMESH_LOCAL]
#endif
};


struct cuda_diffuse_param {             //[GPU_NUM]
  struct step_func_factor *step_fact; 
  struct angle_info *angle;             //[N_ANG]
  struct radiation_mesh *rmesh;
};

#ifndef __CUDACC__
#ifndef cudaStream_t
#define cudaStream_t int64_t     //both 64bit 
#endif
#endif


#ifndef M_PI
#define M_PI (3.1415926535897932384626433832795029)
#endif

#ifndef M_PI_4
#define M_PI_4 (0.7853981633974483096156608458198757)
#endif

#ifndef TINY
#define TINY (1.0e-31)
#endif


#define CUDA_SAFE_CALL_NO_SYNC( call) {                                 \
    cudaError err = call;						\
    if( cudaSuccess != err) {						\
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	      __FILE__, __LINE__, cudaGetErrorString( err) );		\
      exit(EXIT_FAILURE);						\
    } }


#define CUDA_SAFE(call)     CUDA_SAFE_CALL_NO_SYNC(call);


#endif  //__DIFFUSE_PHOTON__
