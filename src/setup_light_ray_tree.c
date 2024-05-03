#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/times.h>

#include "run_param.h"
#include "fluid.h"
#include "radiation.h"

#include "prototype.h"

#include "tree_clist.h"

static float  tree_theta_sq;
static float  base_length;
static int    tree_ncrit;
static int    tree_nleaf;
static int    tree_initflag = 1;
static int    tree_clistmask;


#define IRAY_FIRST(ix,iy,iz) (iray_first[(iz)+NMESH_Z_LOCAL*((iy)+NMESH_Y_LOCAL*(ix))])

inline void zero_out_spec(double *spec)
{
  for(int inu=0;inu<NGRID_NU;inu++) spec[inu]=0.0;
}

inline double spec_to_L(double *spec)
{
  double L;

  L=0.0;
  for(int inu=0;inu<NGRID_NU;inu++) L+= spec[inu];

  return L;
}

inline void copy_spec(double *spec1, double *spec2) 
{
  for(int inu=0;inu<NGRID_NU;inu++) spec2[inu] = spec1[inu];
}

inline void accum_spec(double *spec1, double *spec2) 
{
  for(int inu=0;inu<NGRID_NU;inu++) spec2[inu] += spec1[inu];
}


static void kawaiqsort(int a[], uint64_t key[], int lo, int up)
{
  int i, j;
  int tempa;
  uint64_t tempk;
    
  while (up > lo){
    i = lo;
    j = up;
    tempa = a[lo];
    tempk = key[lo]; 
    while (i < j) {
      for (; key[j] > tempk; j--);
      for (a[i] = a[j], key[i] = key[j]; i < j && key[i] <= tempk; i++);
      a[j] = a[i]; 
      key[j] = key[i];
    }
    a[i] = tempa;
    key[i] = tempk;
    
    if (i-lo < up-i) {
      kawaiqsort(a, key, lo, i-1);
      lo = i+1;
    }else{
      kawaiqsort(a, key, i+1, up);
      up = i-1;
    }
  }
}

static int key_to_adr(uint64_t key, struct clist_t clist[])
{
  int adr,retadr;

  adr = (tree_clistmask) & key;  
  if(clist[adr].key == key){
    retadr = adr;
  }else{
    do{
      adr = clist[adr].next;
    }while(clist[adr].key!=key);
    retadr = adr;
  }
  return retadr;
}


static int plant_tree(int ifirst, int n, uint64_t *key, int *index,
		      int key_level, int *nnode, uint64_t current_key,
		      struct clist_t *clist, int *clist_adr, int gflag,
		      double *total_L, double *total_spec, double *cL, 
		      struct radiation_src *src)
{
  int ncell[8],ifirstc[8],ic,gflagc[8];
  int flag,i,prevflag,adr,next,nextadr;
  int zeroflag,k,j;
  double ccmtmp[3];
  
  key_level -= 3;
  if(key_level<0){printf("key_level error\n");exit(1);}
  for(ic=0;ic<8;ic++) ncell[ic]=ifirstc[ic]=0;

  prevflag=-1;
  for(i=ifirst;i<(n+ifirst);i++){
    flag = (key[i]>>key_level)&0x7; 
    ncell[flag]++;
    if(flag!=prevflag) ifirstc[flag]=i;
    prevflag = flag;
  }

  for(ic=0;ic<8;ic++){
    uint64_t tmp_key;
    int iadr;
    if(ncell[ic]!=0){
      tmp_key = ((current_key<<3)|ic);  
      adr = (tree_clistmask) & tmp_key; 
      if(clist[adr].key == 0){
        clist[adr].key = tmp_key;
        clist[adr].ifirst = ifirstc[ic];
        clist[adr].key_level = key_level;
        clist[adr].n = ncell[ic];
        clist[adr].next = -1;
	//        clist[adr].length = clist[adr].l_theta*tree_theta;
	clist[adr].length = base_length * pow(2.0,-(63.0-key_level)/3.0);
        clist[adr].zeroflag = 0xff;
        if(gflag==0){
          if(ncell[ic]<=tree_ncrit){
            clist[adr].gflag = gflagc[ic] = 1;
          }else{
            clist[adr].gflag = gflagc[ic] = 0;
          }
        }else{
          clist[adr].gflag = gflagc[ic] = 2;
        }
      }else{
        iadr = (*clist_adr);
        clist[iadr].key = tmp_key;
        clist[iadr].ifirst = ifirstc[ic];
        clist[iadr].key_level = key_level;
        clist[iadr].n = ncell[ic];
        clist[iadr].length = base_length * pow(2.0,-(63.0-key_level)/3.0);
        clist[iadr].zeroflag = 0xff;
        if(gflag==0){
          if(ncell[ic]<=tree_ncrit){
            clist[iadr].gflag = gflagc[ic] = 1;
          }else{
            clist[iadr].gflag = gflagc[ic] = 0;
          }
        }else{
          clist[iadr].gflag = gflagc[ic] = 2;
        }
        clist[iadr].next = -1;
        next = adr;        
        do{
          nextadr = next;
          next = clist[nextadr].next;      
        }while(next!=-1);
        clist[nextadr].next = iadr;
        (*clist_adr)++;
      }
      (*nnode)++;
    }
  }

  ccmtmp[0] = ccmtmp[1] = ccmtmp[2] = (*total_L) = 0.0;
  zero_out_spec(total_spec);
  for(ic=0;ic<8;ic++){
    int tmpif,tmpnc,ii,tmpgflagc,tmpadr;
    uint64_t tmp_key;

    if(ncell[ic]!=0){

      tmpif = ifirstc[ic];
      tmpnc = ncell[ic];
      tmpgflagc = gflagc[ic];
      tmp_key =  current_key<<3 | ic;
      tmpadr =  key_to_adr(tmp_key,clist);

      if(ncell[ic]>tree_nleaf && key_level>0){
	double ccmass,cccm[3],spec[NGRID_NU];
         clist[tmpadr].zeroflag
	   = plant_tree(tmpif,tmpnc,key,index,key_level,nnode,tmp_key,
			clist,clist_adr,tmpgflagc,&ccmass,spec,cccm,src);
         *total_L += ccmass;
	 accum_spec(spec, total_spec);
         for(k=0;k<3;k++) ccmtmp[k] += cccm[k];

         clist[tmpadr].L = ccmass;
	 copy_spec(spec, clist[tmpadr].spec);
         for(k=0;k<3;k++) clist[tmpadr].cm[k] = cccm[k]/ccmass;

      }else{
	double totalm=0.0,cm[3]={0.0,0.0,0.0},totalminv,mass;
	double spec[NGRID_NU];
	zero_out_spec(spec);
	for(j=tmpif;j<(tmpif+tmpnc);j++){
	  ii = index[j];
	  mass = spec_to_L(src[ii].photon_rate);
	  accum_spec(src[ii].photon_rate, spec);
	  totalm += mass;
	  cm[0] += mass * src[ii].xpos;
	  cm[1] += mass * src[ii].ypos;
	  cm[2] += mass * src[ii].zpos;
	}
	//	clist[tmpadr].L = totalm;
	copy_spec(spec, clist[tmpadr].spec);
	clist[tmpadr].L = spec_to_L(spec);

	(*total_L) += totalm;
	accum_spec(spec, total_spec);
	for(k=0;k<3;k++) ccmtmp[k] += cm[k];

	totalminv = 1.0/totalm;
	for(k=0;k<3;k++) clist[tmpadr].cm[k] = cm[k]*totalminv;
      } 
    }
  }
  for(k=0;k<3;k++) cL[k] = ccmtmp[k];

  zeroflag = 0;
  for(ic=0;ic<8;ic++){
    if(ncell[ic]==0) zeroflag |= (1<<ic);
  }

  return zeroflag;

}

uint64_t count_src(float xpos, float ypos, float zpos, 
		   uint64_t current_key, struct clist_t *clist, 
		   int *index, struct run_param *this_run)
{
  int adr;
  float dx, dy, dz, r2;

  uint64_t nsrc;

  adr = key_to_adr(current_key, clist);

  dx = xpos - clist[adr].cm[0];
  dy = ypos - clist[adr].cm[1];
  dz = zpos - clist[adr].cm[2];

  r2 = dx*dx + dy*dy + dz*dz;

  nsrc = 0;
  if(r2 * tree_theta_sq  > SQR(clist[adr].length)) {
    nsrc += 1;
  }else{
    if(clist[adr].n > tree_nleaf){
      int nsrc_local=0;
      for(int ic=0;ic<8;ic++){
        int zeroflag;
        zeroflag = (clist[adr].zeroflag>>ic)&1;
        if(zeroflag != 1){ /* not empty */
          uint64_t tmpkey;
          int tmpn,tmpadr;
          tmpkey = (current_key<<3)|ic;
          tmpadr = key_to_adr(tmpkey,clist);
          tmpn = clist[tmpadr].n;

	  nsrc += count_src(xpos, ypos, zpos, tmpkey, clist, index, this_run);
	}
      }
    }else{
      nsrc += clist[adr].n;
    }
  }

  return nsrc;
}

uint64_t count_ray(struct clist_t *clist, int *index, 
		   struct run_param *this_run, uint64_t *iray_first)
{

  uint64_t nray;

  nray = 0;
  for(int ix=0;ix<NMESH_X_LOCAL;ix++) {
    for(int iy=0;iy<NMESH_Y_LOCAL;iy++) {
      for(int iz=0;iz<NMESH_Z_LOCAL;iz++) {

	uint64_t current_key;
	uint64_t nray_each_cell;

	/* center of the target cell */
	float xpos, ypos, zpos;

	xpos = this_run->xmin_local + ((float)ix+0.5)*this_run->delta_x;
	ypos = this_run->ymin_local + ((float)iy+0.5)*this_run->delta_y;
	zpos = this_run->zmin_local + ((float)iz+0.5)*this_run->delta_z;

	current_key = 1;
	nray_each_cell = count_src(xpos,ypos,zpos,current_key,
				   clist,index,this_run);

	IRAY_FIRST(ix,iy,iz) = nray;
	nray += nray_each_cell;
	
      }
    }
  }

  return nray;
}


int setup_light_ray_to(int ix, int iy, int iz,
		       struct light_ray *ray, uint64_t *iray,
		       uint64_t current_key, struct clist_t *clist, 
		       struct radiation_src *src, int *index, 
		       struct run_param *this_run)
{
  int adr;
  float xpos, ypos, zpos;
  float dx, dy, dz, r2;
  int nray;

  /* Note that ix, iy and iz are global indices */

  adr = key_to_adr(current_key, clist);

#if 0
  xpos = this_run->xmin_local + ((float)ix+0.5)*this_run->delta_x;
  ypos = this_run->ymin_local + ((float)iy+0.5)*this_run->delta_y;
  zpos = this_run->zmin_local + ((float)iz+0.5)*this_run->delta_z;  
#else
  xpos = this_run->xmin + ((float)ix+0.5)*this_run->delta_x;
  ypos = this_run->ymin + ((float)iy+0.5)*this_run->delta_y;
  zpos = this_run->zmin + ((float)iz+0.5)*this_run->delta_z;  
#endif

  dx = xpos - clist[adr].cm[0];
  dy = ypos - clist[adr].cm[1];
  dz = zpos - clist[adr].cm[2];

  r2 = SQR(dx) + SQR(dy) + SQR(dz);

  nray = 0;

  if(r2 * tree_theta_sq > SQR(clist[adr].length)) {
    ray[*iray].src.xpos = clist[adr].cm[0]; 
    ray[*iray].src.ypos = clist[adr].cm[1]; 
    ray[*iray].src.zpos = clist[adr].cm[2]; 
    for(int inu=0;inu<NGRID_NU;inu++) {ray[*iray].src.photon_rate[inu] = clist[adr].spec[inu];}
    ray[*iray].ix_target = ix;
    ray[*iray].iy_target = iy;
    ray[*iray].iz_target = iz;
    (*iray)++;
    nray++;
  }else{ /* need to further open the tree cell */
    if(clist[adr].n > tree_nleaf) {
      for(int ic=0;ic<8;ic++) {
        int zeroflag;
        zeroflag = (clist[adr].zeroflag>>ic)&1;
        if(zeroflag != 1){ /* not empty */
          uint64_t tmpkey;
          int tmpn,tmpadr;
          tmpkey = (current_key<<3)|ic;
          tmpadr = key_to_adr(tmpkey,clist);
          tmpn = clist[tmpadr].n;

	  nray += setup_light_ray_to(ix, iy, iz, ray, iray, tmpkey,
				     clist, src, index, this_run);

	}
      }
    }else{
      for(int i=clist[adr].ifirst;i<(clist[adr].ifirst+clist[adr].n);i++) {
	int isrc = index[i];
	ray[*iray].src = src[isrc];
	ray[*iray].ix_target = ix;
	ray[*iray].iy_target = iy;
	ray[*iray].iz_target = iz;
	(*iray)++;
      }
      nray += clist[adr].n;
    }
  }

  return nray;
}

void setup_light_ray_range(int im_start, int im_end, 
			   struct light_ray *ray, uint64_t current_key, 
			   struct clist_t *clist, struct radiation_src *src, 
			   int *index, struct run_param *this_run)
{
  /* index and number of light ray structure */
  uint64_t iray, nray;
    
  iray = 0;
  nray = 0;

  for(int im=im_start;im<im_end;im++) {
    
    /* 3D local and global indices */
    int ix, iy, iz;
    int ix_glb, iy_glb, iz_glb;
    
    ix = im/(NMESH_Y_LOCAL*NMESH_Z_LOCAL);
    iy = (im-ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL)/NMESH_Z_LOCAL;
    iz = im - ix*NMESH_Y_LOCAL*NMESH_Z_LOCAL - iy*NMESH_Z_LOCAL;

    ix_glb = this_run->rank_x*NMESH_X_LOCAL+ix;
    iy_glb = this_run->rank_y*NMESH_Y_LOCAL+iy;
    iz_glb = this_run->rank_z*NMESH_Z_LOCAL+iz;
    
    current_key = 1;
    
    nray += setup_light_ray_to(ix_glb, iy_glb, iz_glb, ray, &iray, current_key, 
			       clist, src, index, this_run);

  }

  fprintf(this_run->proc_file, "# nray = %llu\n", nray);
  fflush(this_run->proc_file);

  assert(nray <= MAX_NRAY_PER_TARGET);

  calc_ray_segment(ray,this_run);
}


void setup_light_ray(struct light_ray *ray, uint64_t *iray_first,
		     uint64_t current_key, struct clist_t *clist, 
		     struct radiation_src *src, int *index, 
		     struct run_param *this_run)
{
  float xpos, ypos, zpos;
  int ix, iy, iz;

  for(ix=0;ix<NMESH_X_LOCAL;ix++) {
    int ix_global = this_run->rank_x*NMESH_X_LOCAL+ix;
    for(iy=0;iy<NMESH_Y_LOCAL;iy++) {
      int iy_global = this_run->rank_y*NMESH_Y_LOCAL+iy;
      for(iz=0;iz<NMESH_Z_LOCAL;iz++) {
	int iz_global = this_run->rank_z*NMESH_Z_LOCAL+iz;
	uint64_t iray;

	iray = IRAY_FIRST(ix,iy,iz);

	setup_light_ray_to(ix_global, iy_global, iz_global, ray, &iray,
			   current_key, clist, src, index, this_run);
      }
    }
  }
  
}


void construct_tree(struct clist_t *clist, uint64_t *key, int *index, 
		    struct radiation_src *src, struct run_param *this_run)
{
  double scale;

  uint64_t ipos[3];
  uint64_t one;


  for(int i=0;i<(tree_clistmask+1);i++) clist[i].key=0;
  for(int i=0;i<this_run->nsrc;i++) index[i] = i;

  /* making keys */
  float xlength, ylength, zlength;

  xlength = this_run->xmax-this_run->xmin;
  ylength = this_run->ymax-this_run->ymin;
  zlength = this_run->zmax-this_run->zmin;
  base_length = MAX(xlength, MAX(ylength, zlength));

  scale = (double)(1<<21);
  one = ((uint64_t)1)<<63;
  
  for(int i=0;i<this_run->nsrc;i++){
    ipos[0] = (int)((src[i].xpos-this_run->xmin)/base_length * scale);
    ipos[1] = (int)((src[i].ypos-this_run->ymin)/base_length * scale);
    ipos[2] = (int)((src[i].zpos-this_run->zmin)/base_length * scale);
    key[i] = one;

    key[i] |= ((uint64_t)(((ipos[0]&0x1)<<2)     |((ipos[1]&0x1)<<1)     |((ipos[2]&0x1))));
    key[i] |= ((uint64_t)(((ipos[0]&0x2)<<2)     |((ipos[1]&0x2)<<1)     |((ipos[2]&0x2)))<<2);
    key[i] |= ((uint64_t)(((ipos[0]&0x4)<<2)     |((ipos[1]&0x4)<<1)     |((ipos[2]&0x4)))<<4);
    key[i] |= ((uint64_t)(((ipos[0]&0x8)<<2)     |((ipos[1]&0x8)<<1)     |((ipos[2]&0x8)))<<6);
    key[i] |= ((uint64_t)(((ipos[0]&0x10)<<2)    |((ipos[1]&0x10)<<1)    |((ipos[2]&0x10)))<<8);
    key[i] |= ((uint64_t)(((ipos[0]&0x20)<<2)    |((ipos[1]&0x20)<<1)    |((ipos[2]&0x20)))<<10);
    key[i] |= ((uint64_t)(((ipos[0]&0x40)<<2)    |((ipos[1]&0x40)<<1)    |((ipos[2]&0x40)))<<12);
    key[i] |= ((uint64_t)(((ipos[0]&0x80)<<2)    |((ipos[1]&0x80)<<1)    |((ipos[2]&0x80)))<<14);
    key[i] |= ((uint64_t)(((ipos[0]&0x100)<<2)   |((ipos[1]&0x100)<<1)   |((ipos[2]&0x100)))<<16);
    key[i] |= ((uint64_t)(((ipos[0]&0x200)<<2)   |((ipos[1]&0x200)<<1)   |((ipos[2]&0x200)))<<18);
    key[i] |= ((uint64_t)(((ipos[0]&0x400)<<2)   |((ipos[1]&0x400)<<1)   |((ipos[2]&0x400)))<<20);
    key[i] |= ((uint64_t)(((ipos[0]&0x800)<<2)   |((ipos[1]&0x800)<<1)   |((ipos[2]&0x800)))<<22);
    key[i] |= ((uint64_t)(((ipos[0]&0x1000)<<2)  |((ipos[1]&0x1000)<<1)  |((ipos[2]&0x1000)))<<24);
    key[i] |= ((uint64_t)(((ipos[0]&0x2000)<<2)  |((ipos[1]&0x2000)<<1)  |((ipos[2]&0x2000)))<<26);
    key[i] |= ((uint64_t)(((ipos[0]&0x4000)<<2)  |((ipos[1]&0x4000)<<1)  |((ipos[2]&0x4000)))<<28);
    key[i] |= ((uint64_t)(((ipos[0]&0x8000)<<2)  |((ipos[1]&0x8000)<<1)  |((ipos[2]&0x8000)))<<30);
    key[i] |= ((uint64_t)(((ipos[0]&0x10000)<<2) |((ipos[1]&0x10000)<<1) |((ipos[2]&0x10000)))<<32);
    key[i] |= ((uint64_t)(((ipos[0]&0x20000)<<2) |((ipos[1]&0x20000)<<1) |((ipos[2]&0x20000)))<<34);
    key[i] |= ((uint64_t)(((ipos[0]&0x40000)<<2) |((ipos[1]&0x40000)<<1) |((ipos[2]&0x40000)))<<36);
    key[i] |= ((uint64_t)(((ipos[0]&0x80000)<<2) |((ipos[1]&0x80000)<<1) |((ipos[2]&0x80000)))<<38);
    key[i] |= ((uint64_t)(((ipos[0]&0x100000)<<2)|((ipos[1]&0x100000)<<1)|((ipos[2]&0x100000)))<<40);
  }

  kawaiqsort(index, key, 0, this_run->nsrc-1);

  /* constructing the tree structure */
  int key_level, clist_adr, nnode;
  uint64_t current_key;

  current_key = 1;  
  key_level = 63;
  clist_adr = (tree_clistmask) + 1;
  nnode = 0;

  /* compute luminosity center and total luminosity of whole simulation box */
  double xcent, ycent, zcent;
  double lum, allspec[NGRID_NU];
  xcent = ycent = zcent = lum = 0.0;
  zero_out_spec(allspec);
  for(int isrc=0;isrc<this_run->nsrc;isrc++) {
    xcent += src[isrc].xpos*src[isrc].photon_rate[0];
    ycent += src[isrc].ypos*src[isrc].photon_rate[0];
    zcent += src[isrc].zpos*src[isrc].photon_rate[0];
    lum += src[isrc].photon_rate[0];
    accum_spec(src[isrc].photon_rate, allspec);
  }

  xcent /= lum;
  ycent /= lum;
  zcent /= lum;

  clist[current_key].key = current_key;
  clist[current_key].key_level = key_level;
  clist[current_key].next = -1;
  clist[current_key].n = this_run->nsrc;
  clist[current_key].ifirst = 0;
  clist[current_key].length = base_length;
#if 1
  clist[current_key].cm[0] = xcent;
  clist[current_key].cm[1] = ycent;
  clist[current_key].cm[2] = zcent;
  clist[current_key].L = spec_to_L(allspec);
  copy_spec(allspec, clist[current_key].spec);
#else
  clist[current_key].cm[0] = 0.0;
  clist[current_key].cm[1] = 0.0;
  clist[current_key].cm[2] = 0.0;
  clist[current_key].L = 0.0;
#endif
  nnode++;

  double total_L, cL[3], total_spec[NGRID_NU];

  clist[current_key].zeroflag = 
    plant_tree(0, this_run->nsrc, key,index, key_level, &nnode, current_key,
	       clist, &clist_adr, 0, &total_L, total_spec, cL, src);

}

void init_tree_param(int ncrit, struct run_param *this_run, int *nclist)
{
  tree_ncrit=60;
  tree_nleaf=1;
  tree_theta_sq=SQR(this_run->theta_crit);

  *nclist = (unsigned int) (250000*(30.0/(float)tree_nleaf)*((float)this_run->nsrc/1.0e6));
  *nclist = MAX(*nclist, 100);

  tree_clistmask = (int)(log((double)(*nclist))/log(2.0));
  tree_clistmask = (1<<(tree_clistmask-1))-1;

}

uint64_t count_ray_to(int ix, int iy, int iz, struct clist_t *clist, 
		      int *index, struct run_param *this_run)
{
  uint64_t nray;
  uint64_t current_key;

  /* center of the target cell */
  float xpos, ypos, zpos;
  
  xpos = this_run->xmin_local + ((float)ix+0.5)*this_run->delta_x;
  ypos = this_run->ymin_local + ((float)iy+0.5)*this_run->delta_y;
  zpos = this_run->zmin_local + ((float)iz+0.5)*this_run->delta_z;

  current_key = 1;

  nray = count_src(xpos, ypos, zpos, current_key, clist, index, this_run);

  return nray;
}

void setup_light_ray_tree(struct light_ray **ray, struct radiation_src *src, 
			  struct run_param *this_run)
{
  struct clist_t *clist;
  uint64_t *key;
  int *index;
  int nclist_max;

  /* first address of the ray[] array for each grid */
  static uint64_t iray_first[NMESH_X_LOCAL*NMESH_Y_LOCAL*NMESH_Z_LOCAL];

  /* setting up the tree parameters */
  tree_ncrit=60;
  tree_nleaf=1;
  tree_theta_sq=SQR(this_run->theta_crit);

  nclist_max = (unsigned int) (250000*(30.0/(float)tree_nleaf)*((float)this_run->nsrc/1.0e6));
  nclist_max = MAX(nclist_max,100);

  tree_clistmask = (int)(log((double)nclist_max)/log(2.0));
  tree_clistmask = (1<<(tree_clistmask-1))-1;

  clist = (struct clist_t *)malloc(sizeof(struct clist_t)*nclist_max);
  key = (uint64_t *)malloc(sizeof(uint64_t)*this_run->nsrc);
  index = (int *)malloc(sizeof(int)*this_run->nsrc);

  fprintf(this_run->proc_file,
	  "# ========== tree code stat. =============\n");
  fprintf(this_run->proc_file,
	  "# max number of clist :: %d \n", nclist_max);
  fprintf(this_run->proc_file,
	  "# adrmask :: %x %d\n",
	  tree_clistmask,tree_clistmask);

  construct_tree(clist, key, index, src, this_run);

  /* setting up the rays using the tree structure */
  this_run->nray = count_ray(clist, index, this_run, iray_first);
  fprintf(this_run->proc_file,"# number of rays :: %llu\n", this_run->nray);
  *ray = (struct light_ray *) malloc(sizeof(struct light_ray)*this_run->nray);

#if 0
  float xpos, ypos, zpos;
  xpos = this_run->xmin_local+0.5*this_run->delta_x;
  ypos = this_run->ymin_local+0.5*this_run->delta_y;
  zpos = this_run->zmin_local+0.5*this_run->delta_z;
  printf("count_src :: %llu\n", count_src(xpos, ypos, zpos, 1, clist, index, this_run));
#endif

  setup_light_ray(*ray, iray_first, 1, clist, src, index, this_run);

  calc_ray_segment(*ray, this_run);

#if 0
  uint64_t iray = 0;
  uint64_t nray;
  nray = setup_light_ray_to(0, 0, 0, *ray, &iray, 1, clist, src, index, this_run);
#endif

  //  printf("aa %llu\n", nray);
  //  printf("%d %d %d\n", iray_first[0], iray_first[1], iray_first[2]);

  free(clist); free(key); free(index);
}
