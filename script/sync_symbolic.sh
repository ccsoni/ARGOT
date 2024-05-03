#!/bin/bash

### run in src/ dir. ###

ln -sf prim_chem/chemistry.h chemistry.h

cd prim_chem
ln -sf ../run_param.h run_param.h
ln -sf ../constants.h constants.h
ln -sf ../source.h source.h
ln -sf ../cross_section.h cross_section.h
ln -sf ../fluid.h fluid.h
ln -sf ../cosmology.h cosmology.h
cd ..
