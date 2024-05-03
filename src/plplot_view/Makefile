#CC=gcc

ifeq ($(cosmological), yes)
COSM_OBJ = ../cosmology.o ../update_expansion.o
endif

SURF_VIEW_OBJ = surf_view.o ../input_data.o $(COSM_OBJ)
SURF_VIEW_DEP := $(SURF_VIEW_OBJ)

SURF_VIEW_M_OBJ = surf_view_multi.o ../input_data.o $(COSM_OBJ)
SURF_VIEW_M_DEP := $(SURF_VIEW_M_OBJ)

SURF_VIEW_M_M_OBJ = surf_view_multi_min.o ../input_data.o $(COSM_OBJ)
SURF_VIEW_M_M_DEP := $(SURF_VIEW_M_M_OBJ)

PROFILE_1D_OBJ = profile_1D.o ../input_data.o $(COSM_OBJ)
PROFILE_1D_DEP := $(PROFILE_1D_OBJ)

PROFILE_1D_DIAG_OBJ = profile_1D_diag.o ../input_data.o $(COSM_OBJ)
PROFILE_1D_DIAG_DEP := $(PROFILE_1D_DIAG_OBJ)

PROFILE_VIEW_OBJ = profile_view.o ../input_data.o $(COSM_OBJ)
PROFILE_VIEW_DEP := $(PROFILE_VIEW_OBJ)

PROFILE_PLANE_OBJ = profile_plane.o ../input_data.o $(COSM_OBJ)
PROFILE_PLANE_DEP := $(PROFILE_PLANE_OBJ)

PROFILE_1D_LINE_OBJ = profile_1D_line.o ../input_data.o $(COSM_OBJ)
#PROFILE_1D_LINE_OBJ = profile_1D_line_ave.o ../input_data.o $(COSM_OBJ)
PROFILE_1D_LINE_DEP := $(PROFILE_1D_LINE_OBJ)


#override CFLAGS += `$(PKG_CONFIG_ENV) pkg-config --cflags --libs plplotd` 
override CFLAGS += -I../diffuse_photon -lm

surf_view: $(SURF_VIEW_DEP)
	$(CC)  -o $@ $(SURF_VIEW_OBJ) $(CFLAGS)

surf_view_multi: $(SURF_VIEW_M_DEP)
	$(CC)  -o $@ $(SURF_VIEW_M_OBJ) $(CFLAGS)

surf_view_multi_frac: surf_view_multi_frac.o ../input_data.o $(COSM_OBJ)
	$(CC)  -o $@ surf_view_multi_frac.o ../input_data.o $(COSM_OBJ) $(CFLAGS)

profile_1D: $(PROFILE_1D_DEP) 
	$(CC)  -o $@ $(PROFILE_1D_OBJ) $(CFLAGS)

profile_1D_diag: $(PROFILE_1D_DIAG_DEP) 
	$(CC)  -o $@ $(PROFILE_1D_DIAG_OBJ) $(CFLAGS)

profile_1D_line: $(PROFILE_1D_LINE_DEP)
	$(CC)  -o $@ $(PROFILE_1D_LINE_OBJ) $(CFLAGS)

profile_view: $(PROFILE_VIEW_DEP) 
	$(CC)  -o $@ $(PROFILE_VIEW_OBJ) $(CFLAGS) 

profile_plane: $(PROFILE_PLANE_DEP) 
	$(CC)  -o $@ $(PROFILE_PLANE_OBJ) $(CFLAGS) 

%.o : %.c
	$(CC) -c  -o $@  $<  $(CFLAGS) 

clean: 
	-rm -rf surf_view surf_view_multi surf_view_multi_min profile_1D profile_view  *.o *~

