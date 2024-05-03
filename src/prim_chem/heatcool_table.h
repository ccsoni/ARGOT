#ifdef __cplusplus
extern "C" {
#endif

#ifndef __HEATCOOL_TABLE__
#define __HEATCOOL_TABLE__

#define N_NH_BIN (146)
#define N_T_BIN (266)
#define LOG_NH_MIN (-8.0)
#define LOG_NH_MAX (1.0)
#define LOG_T_MIN (0.5)
#define LOG_T_MAX (8.5)

struct heatcool_table {
  float heat_tbl[N_NH_BIN][N_T_BIN];
  float cool_tbl[N_NH_BIN][N_T_BIN];
  float wmol_tbl[N_NH_BIN][N_T_BIN];

  float lognh_tbl[N_NH_BIN];
  float logt_tbl[N_T_BIN];
  
  float dlognh, dlogt;
};

#endif

#ifdef __cplusplus
}
#endif
