use_gpu_argot = yes
use_gpu_art = yes

compiler = gnu
#compiler = intel
radiation_transfer = yes
photon_conserving = yes
diffuse = yes
#case_B = yes
gravity = yes
isolated = yes
#cosmological = yes

### approximate Riemann solver ###
#riemann=ausmp
riemann=hllc

#site = gorilla
#site = ppx
site = cosmos-g03
#site = cygnus

ifeq ($(diffuse), yes)
ifeq ($(use_gpu_argot), yes)
ifneq ($(use_gpu_art), yes)
$(error Not support "use_gpu_argot=yes" && "use_gpu_art=no".)
endif
endif
endif

# Cygnus
ifeq ($(site), cygnus)
# GPU Compute Capability
# Pascal arch (P100)
SM:=70

ifeq ($(compiler), intel)
CC=mpicc
CFLAGS = -O3 -funroll-loops -std=c99 -xCORE-AVX2 -mcmodel=medium -shared-intel -qopenmp -Iprim_chem
FC=ifort
FFLAGS= -nofor-main -qopenmp
AR=xiar
FFTW3_DIR = /work/RTRHD/kohji/depot/fftw3_openmpi4_intel19
HPIX_DIR = /work/RTRHD/kohji/depot/Healpix_3.60_intel19
PGPLOT_DIR = /work/RTRHD/kohji/depot/pgplot_intel19
endif

ifeq ($(compiler), gnu)
CC=mpicc
CFLAGS = -O3 -ffast-math -funroll-loops -std=c99 -mcmodel=medium -fopenmp -Iprim_chem
FC=gfortran
FFLAGS= -fopenmp
AR=ar

FFTW3_DIR = /work/RTRHD/kohji/depot/fftw3_mvapich2_gcc4
HPIX_DIR = /work/RTRHD/kohji/depot/Healpix_3.60_gcc4
PGPLOT_DIR = /work/RTRHD/kohji/depot/pgplot_gcc4
endif

CUDAROOT = /system/apps/cuda/11.2.1

endif #site-cygnus

# CCS cosmos-g03
ifeq ($(site), cosmos-g03)
CC=mpicc
FC=gfortran

CFLAGS = -O3 -fPIC -ffast-math -funroll-loops -std=c99 -mcmodel=medium -fopenmp -Iprim_chem
FFLAGS = -fopenmp
AR = ar

# GPU Compute Capability
# Ampere arch
SM:=80

FFTW3_DIR = /usr/local
HPIX_DIR = /import/work1/kohji/Healpix_3.70
PGPLOT_DIR = /usr/local/pgplot

CUDAROOT = /opt/cuda/11.0
endif # CCS cosmos-g03


# CCS gorilla
ifeq ($(site), gorilla)
CC=mpicc
FC=ftn

CFLAGS = -O3 -ffast-math -funroll-loops -std=c99 -mcmodel=medium -fopenmp -Iprim_chem
FFLAGS = -fopenmp
AR = ar

FFTW3_DIR = /usr/local/fftw-3
HPIX_DIR = /home/stanaka/software/Healpix/3.20
PGPLOT_DIR = /home/kohji/software/pgplot
PKG_CONFIG_ENV = PKG_CONFIG_PATH=/home/stanaka/software/plplot/5.11.1/lib/pkgconfig
endif #site=gorilla

# CCS ppx
ifeq ($(site), ppx)
# GPU Compute Capability
# Pascal arch
SM:=61

CC=mpicc
FC=gfortran

CFLAGS = -O3 -ffast-math -funroll-loops -std=c99 -mcmodel=medium -fopenmp -Iprim_chem
FFLAGS = -fopenmp
AR = ar


FFTW3_DIR = /home/kohji/depot/fftw
HPIX_DIR = /home/kohji/depot/Healpix
PGPLOT_DIR = /home/kohji/software/pgplot

CUDAROOT = /usr/local/cuda
endif # CCS ppx

ifeq ($(diffuse), yes)
CFLAGS += -Idiffuse_photon
endif

CLIBS=-Lprim_chem -lprim_chem 

#DIFFUSE_DIR=diffuse_photon.old
DIFFUSE_DIR=diffuse_photon

ifeq ($(diffuse), yes)
CLIBS += -L$(DIFFUSE_DIR) -ldiffuse_photon
endif

ifeq ($(diffuse), yes)
ifeq ($(use_gpu_art), yes)
use_gpu = yes
CPPFLAGS += -D__USE_GPU__ -D__USE_GPU_ART__
ifeq ($(use_gpu_argot), yes)
CPPFLAGS += -D__USE_GPU_ARGOT__
endif
endif
else #diffuse!=yes
ifeq ($(use_gpu_argot), yes)
use_gpu = yes
CPPFLAGS += -D__USE_GPU__ -D__USE_GPU_ARGOT__
endif
endif

ifeq ($(radiation_transfer), yes)
CPPFLAGS += -D__RADIATION_TRANSFER__
ifeq ($(photon_conserving), yes)
CPPFLAGS += -D__PHOTON_CONSERVING__
endif
endif

# isolated boundary condition
ifeq ($(isolated), yes)	
CPPFLAGS += -D__ISOLATED__
endif

# cosmological simulations
ifeq ($(cosmological), yes)
CPPFLAGS += -D__COSMOLOGICAL__
endif

# 2nd order time integration for fluid dynamics 
CPPFLAGS += -D__SECOND_ORDER_RUNGE_KUTTA__

# PGPLOT
PGPLOT_INC = $(PGPLOT_DIR)
PGPLOT_LIB_DIR = $(PGPLOT_DIR)
PGPLOT_LIBS = -L$(PGPLOT_LIB_DIR) -lcpgplot -lpgplot -lpng -lz -L/usr/X11R6/lib -lX11 -lm

# PLPLOT PATH
PLPLOT_LIBS = `$(PKG_CONFIG_ENV) pkg-config --cflags --libs plplot` -lm
PLPLOT_VIEW_DIR=plplot_view


# debug mode for assign_ray_segment()
#CPPFLAGS += -D__ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__ 
#CPPFLAGS += -DNDEBUG

# profiling mode 
CPPFLAGS += -D__ARGOT_PROFILE__

# compiler flags for prim chem library.
CPPFLAGS += -D__HEATCOOL__
#CPPFLAGS += -D__CUBA__
#CPPFLAGS += -D__HELIUM__
#CPPFLAGS += -D__HELIUM_BB__
CPPFLAGS += -D__HYDROGEN_MOL__
#CPPFLAGS += -D__UV_BACKGROUND__
CPPFLAGS += -D__NOUV__

ifeq ($(diffuse), yes)
CPPFLAGS += -D__DIFFUSE_RADIATION__
endif

ifeq ($(case_B), yes)
CPPFLAGS += -D__CASE_B__
endif

# atomic operation in CUDA or OpenMP
#CPPFLAGS += -D__USE_ATOMIC__

# CUDA compiler
NVCC=nvcc
NVCCFLAGS := -arch=sm_$(SM) -use_fast_math -O3 $(CPPFLAGS) -Iprim_chem -Idiffuse_photon
ifeq ($(compiler), gnu)
NVCCFLAGS += -Xcompiler '-mcmodel=medium -fPIC -O3'
endif
ifeq ($(compiler), intel)
NVCCFLAGS += --compiler-bindir=mpiicc -Xcompiler '-O3 -xHost -qopenmp -mcmodel=medium'
endif
ifeq ($(use_gpu), yes)
CLIBS += -L$(CUDAROOT)/lib64 -lcudart  -lstdc++
endif

#healpix
ifeq ($(diffuse), yes)
HPIX_INC := -I$(HPIX_DIR)/include
HPIX_LIB := -L$(HPIX_DIR)/lib -lchealpix
HPIX     := $(HPIX_INC) $(HPIX_LIB)
endif

#fftw3
FFTW3_INC := -I$(FFTW3_DIR)/include
FFTW3_LIB := -L$(FFTW3_DIR)/lib -lfftw3f_mpi -lfftw3f
ifeq ($(gravity), yes)
CFLAGS += $(FFTW3_INC) -D__GRAVITY__
ifeq ($(isolated), yes)
CFLAGS += -D__ISOLATED_GRAV__
endif
endif


#IC_DIR= .
IC_DIR=test_IC

#IC_PREFIX = sod_shock_IC
#IC_PREFIX = uniform_medium_IC
#IC_PREFIX = test_grav_IC
#IC_PREFIX = evrard_collapse_IC
#IC_PREFIX = bonnor_ebert_IC
#IC_PREFIX = bonnor_ebert_side_ray_IC
#IC_PREFIX = jeans_instability_IC
#IC_PREFIX = sedov_taylor_IC

#IC_PREFIX = test_clump_smooth_IC
IC_PREFIX = cmpproj_test2_IC
#IC_PREFIX = cmpproj_test2_clump_IC
#IC_PREFIX = cmpproj_test3_IC
### step_radiation_tree.o ->
#IC_PREFIX = cmpproj_test4_IC
#IC_PREFIX = cmpproj_test5_IC
#IC_PREFIX = cmpproj_test6_IC
#IC_PREFIX = cmpproj_test7_IC

#IC_PREFIX = make_rad_source_IC


ifeq ($(IC_PREFIX), evrard_collapse_IC)
DIAGNOSTICS_OBJ = diagnostics_evrard.o
else ifeq ($(IC_PREFIX), bonnor_ebert_IC)
DIAGNOSTICS_OBJ = diagnostics_evrard.o
else
DIAGNOSTICS_OBJ = diagnostics.o
endif


ifeq ($(IC_PREFIX), cmpproj_test3_IC)
STEP_RADIATION_OBJ = step_radiation.o 
else ifeq ($(IC_PREFIX),bonnor_ebert_side_ray_IC)
STEP_RADIATION_OBJ = step_radiation.o
else
STEP_RADIATION_OBJ = step_radiation_tree.o
endif

IC_INC := -I$(CURDIR)


ifeq ($(riemann), ausmp)
FLUID_OBJ=fluid_ausm+.o
CALC_GRAVITY_TERM_OBJ=calc_gravity_term_ausm+.o

else ifeq ($(riemann), hllc)
FLUID_OBJ=fluid_hllc.o
CALC_GRAVITY_TERM_OBJ=calc_gravity_term_hllc.o
endif


ALL_EXE = argot_hydro_mpi argot_mpi setup_IC
#UTIL_EXE = surf_view/profile_1D surf_view/surf_view check_source plplot_view/surf_view plplot_view/profile_1D plplot_view/profile_view
UTIL_EXE = check_source plplot_view/surf_view plplot_view/profile_1D
TEST_EXE = test_calc_ray_segment test_ray_segment_comm test_optical_depth test_tree

#IC_OBJ = $(IC_PREFIX).o
IC_OBJ = $(IC_DIR)/$(IC_PREFIX).o


PRIM_CHEM_DIR=prim_chem

ARGOT_HYDRO_MPI_OBJ = argot_hydro_mpi.o calc_ray_segment.o init_mpi.o \
        cross_section.o init_run.o mpi_rank.o input_data.o \
        calc_optical_depth.o make_directory.o setup_light_ray_long.o \
        setup_light_ray_tree.o assign_ray_segment.o accum_optical_depth.o \
        calc_photoion_rate.o smooth_photoion_rate.o calc_timestep.o       \
        step_chemistry.o  timing.o set_timing.o                           \
        output_data.o update_chemistry.o input_params.o                   \
        $(FLUID_OBJ) pad_region.o set_fluid_pad.o			  \
	${STEP_RADIATION_OBJ} ${DIAGNOSTICS_OBJ}

ARGOT_MPI_OBJ = argot_mpi.o calc_ray_segment.o init_mpi.o cross_section.o \
        init_run.o mpi_rank.o input_data.o calc_optical_depth.o           \
        make_directory.o setup_light_ray_long.o setup_light_ray_tree.o    \
        assign_ray_segment.o accum_optical_depth.o                        \
        calc_photoion_rate.o smooth_photoion_rate.o calc_timestep.o       \
        step_chemistry.o  timing.o  set_timing.o                          \
        output_data.o update_chemistry.o input_params.o                   \
        ${STEP_RADIATION_OBJ} ${DIAGNOSTICS_OBJ}



SETUP_IC_OBJ = $(IC_OBJ) output_data.o mpi_rank.o make_directory.o source.o

CHECK_SOURCE_OBJ = check_source.o input_data.o 

TEST_GRAV_OBJ = test_grav.o init_mpi.o input_data.o input_params.o \
	init_run.o mpi_rank.o make_directory.o cross_section.o \
	init_fftw_mpi.o init_green.o calc_grav_pot.o

TEST_CALC_RAY_SEGMENT_OBJ = test_calc_ray_segment.o init_mpi.o		\
	input_data.o init_run.o calc_ray_segment.o			\
	calc_optical_depth.o mpi_rank.o make_directory.o timing.o

TEST_RAY_SEGMENT_COMM_OBJ = test_ray_segment_comm.o init_mpi.o	\
	init_run.o mpi_rank.o make_directory.o timing.o

TEST_OPTICAL_DEPTH_OBJ = test_optical_depth.o init_mpi.o input_data.o	\
	init_run.o calc_ray_segment.o calc_optical_depth.o mpi_rank.o	\
	make_directory.o setup_light_ray_long.o assign_ray_segment.o	\
	accum_optical_depth.o timing.o calc_photoion_rate.o		\
	smooth_photoion_rate.o calc_timestep.o step_chemistry.o update_chemistry.o 

TEST_TREE_OBJ = test_tree.o input_data.o setup_light_ray_tree.o calc_ray_segment.o mpi_rank.o

DIV_NODES_OBJ = divide_nodes.o make_directory.o 

ifeq ($(gravity), yes)
ARGOT_HYDRO_MPI_OBJ += init_fftw_mpi.o init_green.o calc_grav_pot.o $(CALC_GRAVITY_TERM_OBJ)
endif

ifeq ($(use_gpu), yes) 
ARGOT_HYDRO_MPI_OBJ += init_gpu.o send_mesh_data.o send_photoion_data.o recv_photoion_data.o \
	chemistry.o recv_mesh_data.o allocate_pinned_segment.o send_run_param_data.o merge_photoion_data.o \
	calc_photoion_rate_cuda.o calc_optical_depth_cuda.o
ARGOT_MPI_OBJ += init_gpu.o send_mesh_data.o send_photoion_data.o recv_photoion_data.o \
	chemistry.o recv_mesh_data.o allocate_pinned_segment.o send_run_param_data.o merge_photoion_data.o \
	calc_photoion_rate_cuda.o calc_optical_depth_cuda.o
TEST_OPTICAL_DEPTH_OBJ += init_gpu.o send_mesh_data.o send_photoion_data.o recv_photoion_data.o allocate_pinned_segment.o chemistry.o 
endif

ifeq ($(cosmological), yes)
ARGOT_MPI_OBJ += update_expansion.o cosmology.o
endif

ARGOT_HYDRO_MPI_DEP := $(ARGOT_HYDRO_MPI_OBJ)
ARGOT_MPI_DEP := $(ARGOT_MPI_OBJ)
SETUP_IC_DEP := $(SETUP_IC_OBJ)
CHECK_SOURCE_DEP := $(CHECK_SOURCE_OBJ)
TEST_GRAV_DEP := $(TEST_GRAV_OBJ)
TEST_CALC_RAY_SEGMENT_DEP := $(TEST_CALC_RAY_SEGMENT_OBJ)
TEST_RAY_SEGMENT_COMM_DEP := $(TEST_REY_SEGMENT_COMM_OBJ) 
TEST_OPTICAL_DEPTH_DEP := $(TEST_OPTICAL_DEPTH_OBJ)
TEST_TREE_DEP := $(TEST_TREE_OBJ)
DIV_NODES_DEP := $(DIV_NODES_OBJ)

ifeq ($(diffuse), yes)
ARGOT_HYDRO_MPI_DEP += $(DIFFUSE_DIR)/libdiffuse_photon.a
ARGOT_MPI_DEP += $(DIFFUSE_DIR)/libdiffuse_photon.a
endif

ifeq ($(diffuse), yes)
all : $(ALL_EXE) $(UTIL_EXE) prim_chem/libprim_chem.a $(DIFFUSE_DIR)/libdiffuse_photon.a
else
all : $(ALL_EXE) $(UTIL_EXE) prim_chem/libprim_chem.a
endif
exec : $(ALL_EXE)
util : $(UTIL_EXE)
test : $(TEST_EXE)

argot_hydro_mpi: prim_chem/libprim_chem.a $(ARGOT_HYDRO_MPI_DEP)
	$(CC) $(CFLAGS) -o $@ $(ARGOT_HYDRO_MPI_OBJ) $(CLIBS) $(HPIX) $(FFTW3_LIB) -lm

argot_mpi: prim_chem/libprim_chem.a $(ARGOT_MPI_DEP)
	$(CC) $(CFLAGS) -o $@ $(ARGOT_MPI_OBJ) $(CLIBS) $(HPIX) -lm

setup_IC: prim_chem/libprim_chem.a $(SETUP_IC_DEP) 
	$(CC) $(CFLAGS) -o $@ $(SETUP_IC_OBJ) $(CLIBS) -lm

check_source: $(CHECK_SOURCE_DEP)
	$(CC) $(CFLAGS) -o $@ $(CHECK_SOURCE_OBJ) $(CLIBS) -lm

test_grav: prim_chem/libprim_chem.a $(TEST_GRAV_DEP)
	$(CC) $(CFLAGS) -o $@ $(TEST_GRAV_OBJ) $(CLIBS) $(FFTW3_LIB) -lm

test_calc_ray_segment: $(TEST_CALC_RAY_SEGMENT_DEP)
	$(CC) $(CFLAGS) -o $@ $(TEST_CALC_RAY_SEGMENT_OBJ) $(CLIBS) -lm

test_ray_segment_comm: $(TEST_RAY_SEGMENT_COMM_DEP)
	$(CC) $(CFLAGS) -o $@ $(TEST_RAY_SEGMENT_COMM_OBJ) $(CLIBS) -lm

test_optical_depth: $(TEST_OPTICAL_DEPTH_DEP)
	$(CC) $(CFLAGS) -o $@ $(TEST_OPTICAL_DEPTH_OBJ) $(CLIBS) -lm

test_tree: $(TEST_TREE_DEP)
	$(CC) $(CFLAGS) -o $@ $(TEST_TREE_OBJ) $(CLIBS) -lm

prim_chem/libprim_chem.a: 
	cd $(PRIM_CHEM_DIR) && $(MAKE) CC="$(CC)" AR="$(AR)" CFLAGS="$(CFLAGS) $(CPPFLAGS)"

divide_nodes: $(DIV_NODES_DEP)
	$(CC) $(CFLAGS) -o $@ $(DIV_NODES_OBJ) 


$(DIFFUSE_DIR)/libdiffuse_photon.a: 
	cd $(DIFFUSE_DIR)/ && $(MAKE) use_gpu="$(use_gpu)" compiler="$(compiler)" CC="$(CC)"  AR="$(AR)" CFLAGS="$(CFLAGS) $(CPPFLAGS)" HPIX="$(HPIX)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" CLIBS="$(CLIBS)" SM="$(SM)"

surf_view/surf_view:
	cd surf_view/ && $(MAKE) surf_view CC="$(CC)" FC="$(FC)" CFLAGS="$(CFLAGS) $(CPPFLAGS) -I.. -I$(PGPLOT_INC)" CLIBS="$(PGPLOT_LIBS)" FFLAGS="$(FFLAGS)"

surf_view/profile_1D:
	cd surf_view/ && $(MAKE) profile_1D CC="$(CC)" CFLAGS="$(CFLAGS) $(CPPFLAGS) -I.."


$(PLPLOT_VIEW_DIR)/surf_view:
	cd $(PLPLOT_VIEW_DIR)/ && $(MAKE) surf_view compiler="$(compiler)" CC="$(CC)"  CFLAGS="$(CFLAGS) $(CPPFLAGS)  -I.. $(PLPLOT_LIBS) $(PLPLOT_FLAG)" cosmological="$(cosmological)"

$(PLPLOT_VIEW_DIR)/surf_view_multi_frac:
	cd $(PLPLOT_VIEW_DIR)/ && $(MAKE) surf_view_multi_frac compiler="$(compiler)" CC="$(CC)"  CFLAGS="$(CFLAGS) $(CPPFLAGS)  -I.. $(PLPLOT_LIBS) $(PLPLOT_FLAG)" cosmological="$(cosmological)"

$(PLPLOT_VIEW_DIR)/profile_1D:
#	cd $(PLPLOT_VIEW_DIR)/ && $(MAKE) profile_1D compiler="$(compiler)" CC="$(CC)"  CFLAGS="$(CFLAGS) $(CPPFLAGS) -I.. $(PLPLOT_LIBS) $(PLPLOT_FLAG)" cosmological="$(cosmological)"
	cd $(PLPLOT_VIEW_DIR)/ && $(MAKE) profile_1D compiler="$(compiler)" CC="$(CC)"  CFLAGS="$(CFLAGS) $(CPPFLAGS) -I.." cosmological="$(cosmological)"

$(PLPLOT_VIEW_DIR)/profile_1D_diag:
#	cd $(PLPLOT_VIEW_DIR)/ && $(MAKE) profile_1D_diag compiler="$(compiler)" CC="$(CC)"  CFLAGS="$(CFLAGS) $(CPPFLAGS) -I.. $(PLPLOT_LIBS) $(PLPLOT_FLAG)" cosmological="$(cosmological)"
	cd $(PLPLOT_VIEW_DIR)/ && $(MAKE) profile_1D_diag compiler="$(compiler)" CC="$(CC)"  CFLAGS="$(CFLAGS) $(CPPFLAGS) -I.." cosmological="$(cosmological)"

$(PLPLOT_VIEW_DIR)/profile_1D_line:
	cd $(PLPLOT_VIEW_DIR)/ && $(MAKE) profile_1D_line compiler="$(compiler)" CC="$(CC)"  CFLAGS="$(CFLAGS) $(CPPFLAGS) -I.." cosmological="$(cosmological)"


#$(PLPLOT_VIEW_DIR)/surf_view_multi:
#	cd $(PLPLOT_VIEW_DIR)/ && $(MAKE) surf_view_multi compiler="$(compiler)" CC="$(CC)"  CFLAGS="$(CFLAGS) $(CPPFLAGS)  -I.. $(PLPLOT_LIBS) $(PLPLOT_FLAG)" cosmological="$(cosmological)"

#$(PLPLOT_VIEW_DIR)/profile_view:
#	cd $(PLPLOT_VIEW_DIR)/ && $(MAKE) profile_view compiler="$(compiler)" CC="$(CC)"  CFLAGS="$(CFLAGS) $(CPPFLAGS) -I.. $(PLPLOT_LIBS) $(PLPLOT_FLAG)" cosmological="$(cosmological)"

#$(PLPLOT_VIEW_DIR)/surf_view_multi_min:
#	cd $(PLPLOT_VIEW_DIR)/ && $(MAKE) surf_view_multi_min compiler="$(compiler)" CC="$(CC)"  CFLAGS="$(CFLAGS) $(CPPFLAGS)  -I.. $(PLPLOT_LIBS) $(PLPLOT_FLAG)" cosmological="$(cosmological)"

#$(PLPLOT_VIEW_DIR)/profile_plane:
#	cd $(PLPLOT_VIEW_DIR)/ && $(MAKE) profile_plane compiler="$(compiler)" CC="$(CC)"  CFLAGS="$(CFLAGS) $(CPPFLAGS) -I.. $(PLPLOT_LIBS) $(PLPLOT_FLAG)" cosmological="$(cosmological)"


.SUFFIXES: .cu

.cu.o:
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -c $*.cu -o $*.o

.c.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $*.c -o $*.o

.c.s:
	$(CC) $(CFLAGS) $(CPPFLAGS) -S $*.c 

$(IC_OBJ):
	$(CC) $(IC_INC) $(CFLAGS) $(CPPFLAGS) -c $*.c -o $*.o

clean: 
	-rm -rf $(ALL_EXE) $(UTIL_EXE) $(TEST_EXE) *.o *.s
	-rm -rf test_IC/*.o  test_IC/*.~
	cd surf_view/ && $(MAKE) clean
	cd prim_chem/ && $(MAKE) clean
	cd plplot_view/ && $(MAKE) clean
	cd $(DIFFUSE_DIR)/ && $(MAKE) clean

distclean: clean
	-rm -rf *~


