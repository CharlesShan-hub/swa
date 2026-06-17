#ifndef SWA_H
#define SWA_H

#define SWA_WAVE_POINTS  512
#define SWA_FEAT_A_NUM   10
#define SWA_FEAT_TAIL    6      /* TEMP + HUMID + RPM + VPP + KURT + SKEW */
#define SWA_N_FEATURES   (SWA_FEAT_A_NUM + SWA_FEAT_TAIL)

/* 特征索引 */
#define SWA_FEAT_TEMP    (SWA_FEAT_A_NUM + 0)
#define SWA_FEAT_HUMID   (SWA_FEAT_A_NUM + 1)
#define SWA_FEAT_RPM     (SWA_FEAT_A_NUM + 2)
#define SWA_FEAT_VPP     (SWA_FEAT_A_NUM + 3)
#define SWA_FEAT_KURT    (SWA_FEAT_A_NUM + 4)
#define SWA_FEAT_SKEW    (SWA_FEAT_A_NUM + 5)

/* 一条完整的数据 */
typedef struct {
    double wave[SWA_WAVE_POINTS];   /* 512 个波形点 */
    double actual_voltage;          /* 真值电压 */
    double temp;                    /* 环境温度 */
    double humidity;                /* 环境湿度 */
    double rpm;                     /* 转速 */
} swa_input_t;

/* 提取后的 16 维特征 */
typedef struct {
    double data[SWA_N_FEATURES];    /* A1~A10, T, RH, RPM, Vpp, Kurtosis, Skewness */
} swa_features_t;

/* 预测输出 */
typedef struct {
    double voltage;                 /* 预测电压值 */
} swa_output_t;

#endif /* SWA_H */
