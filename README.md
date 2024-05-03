# **ARGOT** #

Radiation hydrodynamics code for astrophysical objects using ARGOT
(Accelerated Radiation Transfer On Grids using Oct-Tree) scheme for
point radiating sources and ART (Authentic Ray Tracing) scheme for
diffuse radiation.

# Directory: #

`./src` : source files

`./src/prim_chem` : library for primordial chemistry network

`./src/diffuse_photon` : source files for diffuse radiation transfer 

# Compilation Procedure: #

First, you should prepare your own source code for creating an
initial condition (IC). As an example, we have **'comparison_test2_IC.c'**
to create the IC which is identical to the Test-2 in Cosmological Radiative
Transfer Comparison Project by Iliev et al (2006) and (2009).  After preparing
the source code for the IC, say **'setup_my_IC.c'**, a variable **'IC_PREFIX'**
should be set to **'setup_my_IC'** as

`IC_PREFIX = setup_my_IC`

in the 'Makefile'. Then, we can build  binaries to create ICs and to run simulations by issuing following commands:

` $ make setup_my_IC`       # build a program for setting up initial conditions

` $ make argot_mpi`         # build a program for radiation transfer simulations

` $ make argot_hydro_mpi`   # build a program for radiation hydrodynamics simulations.

Visualization programs are prepared in **surf_view** directory and can be built with

` $ make surf_view/surf_view` # build a program for simple visualization

and 

` $ make surf_view/profile_1D` # build a program for computing radial profiles.

Note that PGPLOT package should be installed to build these binaries.

## Important switches in Makefile

* switch to use GPUs

```
#!Makefile
use_gpu = yes　　
```

* switch to choose a compiler

```
#!Makefile
compiler = gnu
#compiler = intel
```

* switch to incorporate radiative transfer and diffuse radiation transfer

```
#!Makefile
radiation_transfer = yes
diffuse = yes
```

* switch to adopt the photon conservation scheme

```
#!Makefile
photon_conserving = yes
```

* switch to adopt the case-B recombination rates for chemical reactions

```
#!Makefile
case_B = yes
```

* switch to incorporate the gravitational interactions

```
#!Makefile
gravity = yes
```

* switch to adopt the isolated boundary condition for the solution of the Poisson equation

```
#!Makefile
isolated = yes
```

* switch to conduct the cosmological simulations

```
#!Makefile
cosmological = yes
```

# Execution: #

## Create initial conditions

`$ ./setup_my_IC <model_name>`

This command creates a directory `<model_name>-init` and the IC
files in it.

## Execute the program

`$ mpiexec -n 8 -f hostfile ./argot_mpi <prefix> <parameter file>`

`$ mpiexec -n 8 -f hostfile ./argot_hydro_mpi <prefix> <parameter file>`

The format of the parameter file is as follows;

```
uniform_medium_nsrc16          <-----  model_name
2                              <-----  number of divisions in ARGOT loop*
5                              <-----  number of output epochs
0.1                            <-----  list of the output epochs
0.2
0.3
0.4
0.5
```

*: In this sample of the parameter file, the ARGOT loop is divided
 into two groups, each of which is responsible for the NMESH_LOCAL/2
 mesh grids. Larger number of divisions and/or smaller number of mesh
 grids in one group requires smaller amount of memory space required
