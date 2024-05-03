#ifndef __MULTI_FREQ__
#define __MULTI_FREQ__

#ifdef __cplusplus
extern "C" {
#endif


#define NBIN_NU (64)

struct multi_freq_table{
  double log_tau_min, log_tau_max, dlog_tau;
  double ionization_table[NBIN_NU][2];
  double heating_table[NBIN_NU][2];
};

extern void setup_ionization_table(double, double, struct multi_freq_table*);

#endif /* __MULTI_FREQ__ */

#ifdef __cplusplus
}
#endif
