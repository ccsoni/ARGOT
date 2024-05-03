#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "run_param.h"
#include "cross_section.h"

#include "prototype.h"

#define FILENAME_LENGTH (256)
void make_directory(char*);

void init_run(struct run_param *this_run)
{
  static char dirname[FILENAME_LENGTH];
  static char proc_filename[FILENAME_LENGTH];

  sprintf(dirname,"%s-out",this_run->model_name);
  make_directory(dirname);
  
  sprintf(proc_filename, "%s-out/out_%03d_%03d_%03d", this_run->model_name,
	  this_run->rank_x, this_run->rank_y, this_run->rank_z);

  this_run->proc_file = fopen(proc_filename,"a");

  this_run->ngrid_nu = (int)NGRID_NU;

  int diffuse_photon_flag, helium_bb_flag;
  diffuse_photon_flag = helium_bb_flag = 0;
  
  this_run->helium_flag = 0;
  this_run->hydrogen_mol_flag = 0;
  this_run->cosmology_flag = 0;
  
#ifdef __HELIUM__
  this_run->helium_flag = 1;
#ifdef __HELIUM_BB__ 
  helium_bb_flag = 1;
#endif
#endif //__HELIUM__
#ifdef __HYDROGEN_MOL__
  this_run->hydrogen_mol_flag = 1;
#endif
#ifdef __COSMOLOGICAL__
  this_run->cosmology_flag = 1;
#endif
#ifdef __DIFFUSE_RADIATION__
  diffuse_photon_flag = 1;
#endif
  
#ifdef _OPENMP
  omp_set_num_threads(OPENMP_NUMBER_OF_THREADS);
#endif /* _OPENMP */

  /* checking parameter configuration of this run */
  fprintf(this_run->proc_file,"\n\n\n\n#==================================================\n");
  fprintf(this_run->proc_file,"# model name : %s\n", this_run->model_name);
  fprintf(this_run->proc_file,"# total number of mesh along X-axis : %d\n", 
          this_run->nmesh_x_total);
  fprintf(this_run->proc_file,"# total number of mesh along Y-axis : %d\n", 
          this_run->nmesh_y_total);
  fprintf(this_run->proc_file,"# total number of mesh along Z-axis : %d\n", 
          this_run->nmesh_z_total);
  fprintf(this_run->proc_file,"# local number of mesh along X-axis : %d\n", 
          this_run->nmesh_x_local);
  fprintf(this_run->proc_file,"# local number of mesh along Y-axis : %d\n", 
          this_run->nmesh_y_local);
  fprintf(this_run->proc_file,"# local number of mesh along Z-axis : %d\n", 
          this_run->nmesh_z_local);
  fprintf(this_run->proc_file,"# number of chemical species : %d\n", 
          this_run->nspecies);

  fprintf(this_run->proc_file,"# number of grid of frequency, ngrid_nu : %d\n", 
          this_run->ngrid_nu);
  fprintf(this_run->proc_file,"# helium flag [0:off, 1:on]: %d\n", 
	  this_run->helium_flag);
  fprintf(this_run->proc_file,"# hydrogen mol flag [0:off, 1:on]: %d\n", 
	  this_run->hydrogen_mol_flag);
  fprintf(this_run->proc_file,"# cosmology flag [0:off, 1:on]: %d\n", 
	  this_run->cosmology_flag);

  fprintf(this_run->proc_file,"# diffuse photon flag [0:off, 1:on]: %d\n", 
	  diffuse_photon_flag);
  fprintf(this_run->proc_file,"# helium bound-bound flag [0:off, 1:on]: %d\n", 
	  helium_bb_flag);
  
  fprintf(this_run->proc_file,"# omp get max threads: %d\n", 
	  omp_get_max_threads());
  
  setup_cross_section(this_run->csect, &this_run->freq);
}
