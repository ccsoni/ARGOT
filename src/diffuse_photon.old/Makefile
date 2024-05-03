#use_gpu = yes
#compiler = gnu
#compiler = intel


# HA-PACS
#CUDAROOT = /opt/CUDA/5.0.35/cudatoolkit
#CUDAROOT = /opt/CUDA/4.1.28/cudatoolkit

# HPT cluster
#CUDAROOT = /usr/local/cuda


ifeq ($(compiler), gnu)
CC=mpicc
CFLAGS= -O3 -ffast-math -funroll-loops -std=c99 -mcmodel=medium -fopenmp -Iprim_chem 
AR=ar
endif

ifeq ($(compiler), intel)
CC=mpiicc
CFLAGS= -O3 -ipo -funroll-loops -std=c99 -xHost -mcmodel=medium -shared-intel -openmp -Iprim_chem 
AR=xiar
endif


#ifeq ($(use_gpu), yes)
#CPPFLAGS += -D__USE_GPU__
#ndif

# isolated boundary condition
#CPPFLAGS += -D__ISOLATED__


#command
RM   := /bin/rm

# debug mode for assign_ray_segment()
#CPPFLAGS += -D__ARGOT_ASSIGN_RAY_SEGMENT_DEBUG__ 
#CPPFLAGS += -DNDEBUG

# profiling mode 
#CPPFLAGS += -D__ARGOT_PROFILE__

# compiler flags for prim chem library.
#CPPFLAGS += -D__HEATCOOL__
#CPPFLAGS += -D__CUBA__
#CPPFLAGS += -D__HELIUM__
#CPPFLAGS += -D__HYDROGEN_MOL__
#CPPFLAGS += -D__UV_BACKGROUND__
#CPPFLAGS += -D__CASE_B__
#CPPFLAGS+ = -D__NOUV__

#healpix
#HPIX_INC := -Ihealpix/include
#HPIX_LIB := -Lhealpix/lib -lchealpix
#HPIX     := $(HPIX_INC) $(HPIX_LIB) 

#NVCC=nvcc
#NVCCFLAGS := -arch=sm_$(SM) -O3 $(CPPFLAGS) -Iprim_chem  -Xcompiler -mcmodel=medium
#NVCCFLAGS += -use_fast_math
#ifeq ($(use_gpu), yes) 
#CLIBS += -L$(CUDAROOT)/lib64 -lcudart 
#endif

COBJS := calc_diffuse_photon_radiation.o calc_rmesh_data.o ray_tracing.o \
	calc_GH_tot.o  set_angle.o set_step_func_factor.o \
	set_start_pos.o setup_diffuse.o diffuse_photon_mpi.o \
	operation_mpi_window.o init_hd_param.o 

CUOBJS := init_diffuse_gpu.o free_diffuse_gpu.o ray_tracing.o  send_diffuse_data.o \
	ray_mem_operation.o calc_rmesh_data.o calc_GH_tot.o 


ifeq ($(use_gpu), yes) 
OBJS   := $(COBJS) $(CUOBJS)
else
OBJS   := $(COBJS) 
endif

DYNMIC_LINK := link.o

BUILD_DIR := obj


ifeq ($(use_gpu), yes)

libdiffuse_photon.a : $(patsubst %,$(BUILD_DIR)/%,$(OBJS)) $(DYNMIC_LINK)
	$(AR) -r $@ $^ 

$(DYNMIC_LINK) : $(patsubst %,$(BUILD_DIR)/%,$(CUOBJS)) 
	nvcc -arch=sm_${SM} -dlink -o $@ $^ 

$(BUILD_DIR)/%.o : %.cu   $(BUILD_DIR)
	$(NVCC) -dc -o $@  $<  $(CPPFLAGS) $(NVCCFLAGS) -I..
else

libdiffuse_photon.a : $(patsubst %,$(BUILD_DIR)/%,$(OBJS))
	$(AR) -r $@ $^ 
endif



$(BUILD_DIR)/%.o : %.c  $(BUILD_DIR)
	$(CC) -c  -o $@  $<  $(CFLAGS) $(CPPFLAGS) $(HPIX) -I..



ifeq ($(use_gpu), yes)
$(BUILD_DIR)/ray_tracing.o : ray_tracing.cu $(BUILD_DIR)
	$(NVCC) -dc -o $@  $<  $(CPPFLAGS) $(NVCCFLAGS) -I..
$(BUILD_DIR)/calc_rmesh_data.o : calc_rmesh_data.cu $(BUILD_DIR)
	$(NVCC) -dc -o $@  $<  $(CPPFLAGS) $(NVCCFLAGS) -I..
$(BUILD_DIR)/calc_GH_tot.o : calc_GH_tot.cu $(BUILD_DIR)
	$(NVCC) -dc -o $@  $<  $(CPPFLAGS) $(NVCCFLAGS) -I..
else
$(BUILD_DIR)/ray_tracing.o : ray_tracing.c $(BUILD_DIR)
	$(CC) -c -o $@  $<  $(CFLAGS) $(CPPFLAGS) -I..
$(BUILD_DIR)/calc_rmesh_data.o : calc_rmesh_data.c $(BUILD_DIR)
	$(CC) -c -o $@  $<  $(CFLAGS) $(CPPFLAGS) -I..
$(BUILD_DIR)/calc_GH_tot.o : calc_GH_tot.c $(BUILD_DIR)
	$(CC) -c -o $@  $<  $(CFLAGS) $(CPPFLAGS) -I..
endif

$(BUILD_DIR) :
	mkdir -p $(BUILD_DIR)



.PHONY: clean
clean:
	$(RM) -rf   $(BUILD_DIR) libdiffuse_photon.a $(DYNMIC_LINK)  *~
