CC=gcc
CFLAGS=-O3 -std=c99 -ffast-math -I.. 
CFLAGS += -D__HEATCOOL__
#CFLAGS += -D__CUBA__
#CFLAGS += -D__HELIUM__
#CFLAGS += -D__HYDROGEN_MOL__
#CFLAGS += -D__UV_BACKGROUND__
#CFLAGS += -D__CASE_B__
#CFLAGS += -D__NOUV__

OBJS = calc_ioneq.o cooling_rate.o cross_section.o heatcool_rate.o \
	heating_rate.o reaction_rate.o step_heatcool.o    \
	../cosmology.o ../update_expansion.o step_reaction.o \
	heat_capacity_ratio.o

GASEVOLVEOBJ = gas_evolve_isobaric.o $(OBJS) 
COOLINGTBL_OBJ = cooling_table_ioneq.o $(OBJS)

all: $(OBJS) libprim_chem.a

libprim_chem.a: $(OBJS)
	$(AR) -r $@ $^

gas_evolve_isobaric: $(GASEVOLVEOBJ)
	$(CC) $(CFLAGS) $(GASEVOLVEOBJ) -o gas_evolve_isobaric -lm

cooling_table_ioneq: $(COOLINGTBL_OBJ)
	$(CC) $(CFLAGS) $(COOLINGTBL_OBJ) -o cooling_table_ioneq -lm

clean: 
	\rm -rf *.o libprim_chem.a

distclean: clean
	\rm -rf *~	
