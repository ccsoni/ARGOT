#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef __COSMOLOGY_H__
#define __COSMOLOGY_H__

struct cosmology {
  float omega_m, omega_v, omega_b, omega_nu, hubble;
  float tend;
};

float fomega(float anow, struct cosmology cosm);
float dladt(float anow, struct cosmology cosm);
float ztotime(float znow, struct cosmology cosm);
float atotime(float znow, struct cosmology cosm);
float timetoz(float tnow, struct cosmology cosm);
float timetoa(float tnow, struct cosmology cosm);
void  funcd(double x, double *f, double *df, double tau);
float rtsafe(double x1, double x2, double xacc, double tau);

#endif /* __COSMOLOGY_H__ */

#ifdef __cplusplus
}
#endif
