CC=gcc
FC=gfortran

SURF_VIEW_OBJ = surf_view.o cpgplot_map.o ../input_data.o 
SURF_VIEW_DEP := $(SURF_VIEW_OBJ)

PROFILE_1D_OBJ = profile_1D.o ../input_data.o
PROFILE_1D_DEP := $(PROFILE_1D_OBJ)

surf_view: $(SURF_VIEW_DEP)
	$(FC) $(FFLAGS) -o $@ $(SURF_VIEW_OBJ) $(CLIBS) -lm

profile_1D: $(PROFILE_1D_DEP) 
	$(CC) $(CFLAGS) -o $@ $(PROFILE_1D_OBJ) -lm

clean: 
	-rm -rf $(ALL_EXE) $(TEST_EXE) *.o

distclean: clean
	-rm -rf *~
