/**
 * @file    agile_modbus.h
 * @brief   Agile Modbus 软件包通用头文件
 * @author  马龙伟 (2544047213@qq.com)
 * @date    2022-07-28
 *
 * @attention
 *
 * <h2><center>&copy; Copyright (c) 2021 Ma Longwei.
 * All rights reserved.</center></h2>
 *
 */

#ifndef __SLAVE_H
#define __SLAVE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <rtthread.h>
#include <agile_modbus.h>
#include <agile_modbus_slave_util.h>

enum RTU_regDesc {
    RTU_REGS_START = -1,
    RTU_REGS_BUILD_TIME,        // 4b, 固件生成日期
    RTU_REGS_BUILD_TIME_2b,
    RTU_REGS_BUILD_TIME_3b,
    RTU_REGS_BUILD_TIME_4b,
    RTU_REGS_UPTIME_SEC,        // 2b, 系统运行时间，单位秒
    RTU_REGS_UPTIME_SEC_2b,
    RTU_REGS_DEVICE_STATE,      // 1b, 设备状态
    RTU_REGS_SENSOR_CNT_P1_P2,  // 1b, 传感器数量，P1+P2
    RTU_REGS_SENSOR_CNT_P3_P4,  // 1b, 传感器数量，P3+P4
    RTU_REGS_SENSOR_CNT_P5_Px,  // 1b, 传感器数量，P5+P6(保留)
    
    RTU_REGS_SLAVE_ID = 10,     // 1b, Modbus从机地址
    RTU_REGS_EXEC_CMD,          // 1b, 执行特殊命令
    RTU_REGS_RESV_BEGIN,        // 8b, 保留
    RTU_REGS_RESV_END = 19,
    // 10b, 传感器通道P1, 20
    RTU_REGS_NODE_P1_0,     // 1b, 传感器通道P1节点0
    RTU_REGS_NODE_P1_1,     // 1b, 传感器通道P1节点1
    RTU_REGS_NODE_P1_2,     // 1b, 传感器通道P1节点2
    RTU_REGS_NODE_P1_3,     // 1b, 传感器通道P1节点3
    RTU_REGS_NODE_P1_4,     // 1b, 传感器通道P1节点4
    RTU_REGS_NODE_P1_5,     // 1b, 传感器通道P1节点5
    RTU_REGS_NODE_P1_6,     // 1b, 传感器通道P1节点6
    RTU_REGS_NODE_P1_7,     // 1b, 传感器通道P1节点7
    RTU_REGS_NODE_P1_8,     // 1b, 传感器通道P1节点8
    RTU_REGS_NODE_P1_9,     // 1b, 传感器通道P1节点9
    // 10b, 传感器通道P2, 30
    RTU_REGS_NODE_P2_0,     // 1b, 传感器通道P2节点0
    RTU_REGS_NODE_P2_1,     // 1b, 传感器通道P2节点1
    RTU_REGS_NODE_P2_2,     // 1b, 传感器通道P2节点2
    RTU_REGS_NODE_P2_3,     // 1b, 传感器通道P2节点3
    RTU_REGS_NODE_P2_4,     // 1b, 传感器通道P2节点4
    RTU_REGS_NODE_P2_5,     // 1b, 传感器通道P2节点5
    RTU_REGS_NODE_P2_6,     // 1b, 传感器通道P2节点6
    RTU_REGS_NODE_P2_7,     // 1b, 传感器通道P2节点7
    RTU_REGS_NODE_P2_8,     // 1b, 传感器通道P2节点8
    RTU_REGS_NODE_P2_9,     // 1b, 传感器通道P2节点9
    // 10b, 传感器通道P3, 40
    RTU_REGS_NODE_P3_0,     // 1b, 传感器通道P3节点0
    RTU_REGS_NODE_P3_1,     // 1b, 传感器通道P3节点1
    RTU_REGS_NODE_P3_2,     // 1b, 传感器通道P3节点2
    RTU_REGS_NODE_P3_3,     // 1b, 传感器通道P3节点3
    RTU_REGS_NODE_P3_4,     // 1b, 传感器通道P3节点4
    RTU_REGS_NODE_P3_5,     // 1b, 传感器通道P3节点5
    RTU_REGS_NODE_P3_6,     // 1b, 传感器通道P3节点6
    RTU_REGS_NODE_P3_7,     // 1b, 传感器通道P3节点7
    RTU_REGS_NODE_P3_8,     // 1b, 传感器通道P3节点8
    RTU_REGS_NODE_P3_9,     // 1b, 传感器通道P3节点9
    // 10b, 传感器通道P4, 50
    RTU_REGS_NODE_P4_0,     // 1b, 传感器通道P4节点0
    RTU_REGS_NODE_P4_1,     // 1b, 传感器通道P4节点1
    RTU_REGS_NODE_P4_2,     // 1b, 传感器通道P4节点2
    RTU_REGS_NODE_P4_3,     // 1b, 传感器通道P4节点3
    RTU_REGS_NODE_P4_4,     // 1b, 传感器通道P4节点4
    RTU_REGS_NODE_P4_5,     // 1b, 传感器通道P4节点5
    RTU_REGS_NODE_P4_6,     // 1b, 传感器通道P4节点6
    RTU_REGS_NODE_P4_7,     // 1b, 传感器通道P4节点7
    RTU_REGS_NODE_P4_8,     // 1b, 传感器通道P4节点8
    RTU_REGS_NODE_P4_9,     // 1b, 传感器通道P4节点9
    // 10b, 传感器通道P5, 60
    RTU_REGS_NODE_P5_0,     // 1b, 传感器通道P5节点0
    RTU_REGS_NODE_P5_1,     // 1b, 传感器通道P5节点1
    RTU_REGS_NODE_P5_2,     // 1b, 传感器通道P5节点2
    RTU_REGS_NODE_P5_3,     // 1b, 传感器通道P5节点3
    RTU_REGS_NODE_P5_4,     // 1b, 传感器通道P5节点4
    RTU_REGS_NODE_P5_5,     // 1b, 传感器通道P5节点5
    RTU_REGS_NODE_P5_6,     // 1b, 传感器通道P5节点6
    RTU_REGS_NODE_P5_7,     // 1b, 传感器通道P5节点7
    RTU_REGS_NODE_P5_8,     // 1b, 传感器通道P5节点8
    RTU_REGS_NODE_P5_9,     // 1b, 传感器通道P5节点9
    // 10b, 传感器通道P1节点温度, 70
    RTU_REGS_TEMP_P1_0,
    RTU_REGS_TEMP_P1_1,
    RTU_REGS_TEMP_P1_2,
    RTU_REGS_TEMP_P1_3,
    RTU_REGS_TEMP_P1_4,
    RTU_REGS_TEMP_P1_5,
    RTU_REGS_TEMP_P1_6,
    RTU_REGS_TEMP_P1_7,
    RTU_REGS_TEMP_P1_8,
    RTU_REGS_TEMP_P1_9,
    // 10b, 传感器通道P1节点湿度, 80
    RTU_REGS_HUM_P1_0,
    RTU_REGS_HUM_P1_1,
    RTU_REGS_HUM_P1_2,
    RTU_REGS_HUM_P1_3,
    RTU_REGS_HUM_P1_4,
    RTU_REGS_HUM_P1_5,
    RTU_REGS_HUM_P1_6,
    RTU_REGS_HUM_P1_7,
    RTU_REGS_HUM_P1_8,
    RTU_REGS_HUM_P1_9,
    // P00波形控制，90
    RTU_REGS_P00_ROTOR_RPM,     // 1b, P00输出速度，0.1 RPM
    RTU_REGS_P00_ENV_HUMIDITY,  // 1b, P00环境湿度，0.1%
    RTU_REGS_P00_ENV_TEMP,      // 1b, P00环境温度，0.1°C
    RTU_REGS_P00_WAVE_BUSY,     // 1b, P00波形控制互斥锁
    // P00波形，100
    RTU_REGS_P00_WAVE_BEGIN,      // 512b, P00波形指针
    RTU_REGS_P00_WAVE_END = RTU_REGS_P00_WAVE_BEGIN + 512,
    RTU_REGS_COUNT,
};

#define WAV_REGS_PER_MAP                125
#define REGISTER_MAPS_COUNT             6
    
extern const agile_modbus_slave_util_t  slave_util;
extern rt_mutex_t                       lkSlvRegs;
extern const agile_modbus_slave_util_map_t register_maps[REGISTER_MAPS_COUNT];

typedef enum {
    MOD_FUNC_AND,
    MOD_FUNC_NAND,
    MOD_FUNC_OR,
    MOD_FUNC_NOR,
    MOD_FUNC_XOR,
    MOD_FUNC_NOT,
}mod_func_t;

int rtu_get_register(uint16_t index, uint16_t *pval);
int rtu_set_register(uint16_t index, uint16_t val);
int rtu_mod_register(uint16_t index, mod_func_t func, uint16_t mask);
int rtu_cpy_register(uint16_t index, uint16_t len, uint16_t *pval);
    
#ifdef __cplusplus
}
#endif

#endif /* __SLAVE_H */
