#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
#include "run_param.h"
#include "fluid.h"

#define MESH(ix,iy,iz) (mesh[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

void update_chemistry(struct fluid_mesh *mesh, struct run_param *this_run)
{
  int ix, iy, iz;
  
  for(ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(iz=0;iz<NMESH_Z_LOCAL;iz++) {
	MESH(ix,iy,iz).prev_chem = MESH(ix,iy,iz).chem;
	MESH(ix,iy,iz).prev_uene = MESH(ix,iy,iz).uene;
	MESH(ix,iy,iz).eneg = 
	  MESH(ix,iy,iz).uene*MESH(ix,iy,iz).dens
	  + 0.5*(SQR(MESH(ix,iy,iz).momx)+
		 SQR(MESH(ix,iy,iz).momy)+
		 SQR(MESH(ix,iy,iz).momz))/MESH(ix,iy,iz).dens;
      }
    }
  }

}
