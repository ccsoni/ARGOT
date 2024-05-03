#ifndef __CASEA_PROTOTYPE_H__
#define __CASEA_PROTOTYPE_H__

#include "diffuse_photon.h"


void init_hd_param(struct host_diffuse_param*);
void free_hd_param(struct host_diffuse_param*);
void setup_diffuse_photon(struct host_diffuse_param *, struct run_param*);
void set_angle_info(struct angle_info*, int*, struct run_param*);
void sort_angle_id(struct angle_info*);
void set_step_func_factor(struct step_func_factor*);
void set_ray_start_position(struct ray_face*, long, struct host_diffuse_param*, struct run_param*);



#ifdef __USE_GPU__
#include "../cuda_mem_space.h"
void init_diffuse_gpu(struct cuda_diffuse_param*, struct cuda_param*,  struct run_param*);
void free_diffuse_gpu(struct cuda_diffuse_param*, struct cuda_param*);
void send_diffuse_data(struct host_diffuse_param*, struct cuda_diffuse_param*, struct cuda_param*);
void cuda_set_device(int);

void setup_ray_face_dev(struct ray_face*, struct cuda_param*);
void finalize_ray_face_dev(struct ray_face*, struct cuda_param*);
void zero_set_rmesh(struct radiation_mesh*, cudaStream_t, int);
void send_ray_face(struct ray_face*, struct ray_face*, cudaStream_t, int);
void calc_rmesh_data(struct cuda_mem_space*, struct cuda_diffuse_param*, struct cuda_param*);
void ray_tracing(long , struct ray_face*, struct cuda_mem_space*, struct cuda_diffuse_param*, struct host_diffuse_param*, cudaStream_t, int);
void calc_GH_tot(struct cuda_mem_space*, struct cuda_diffuse_param*, cudaStream_t, int);
void calc_GH_sum(struct cuda_mem_space*, struct cuda_diffuse_param*, struct cuda_param*);
void calc_diffuse_photon_radiation(struct fluid_mesh*, struct run_param*, struct cuda_mem_space*, struct cuda_param*, struct host_diffuse_param* ,struct cuda_diffuse_param*);
void step_radiation_tree(struct fluid_mesh*, struct radiation_src*, struct run_param*, struct mpi_param*, struct cuda_mem_space*, struct cuda_param*, struct host_diffuse_param*, struct cuda_diffuse_param*, float);

#else //!__USE_GPU__

void calc_rmesh_data(struct fluid_mesh*, struct radiation_mesh*, struct run_param*);
void zero_set_rmesh(struct radiation_mesh*);
void ray_tracing(long, struct ray_face*, struct host_diffuse_param*, struct run_param*);
void calc_GH_tot(struct radiation_mesh*, struct step_func_factor*);
void calc_GH_sum(struct fluid_mesh*, struct radiation_mesh*);
void calc_diffuse_photon_radiation(struct fluid_mesh*, struct run_param*, struct host_diffuse_param*);
void step_radiation_tree(struct fluid_mesh*, struct radiation_src*, struct run_param*, struct mpi_param*, struct host_diffuse_param*, float);
#endif //__USE_GPU__


#endif  //__CASEA_PROTOTYPE_H__
