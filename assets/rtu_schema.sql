-- ============================================================
-- RTU 报文数据库表结构设计
-- 说明：列名与JSON key完全一致，方便直接入库
-- 数据库：达梦 DM8
-- ============================================================

-- ------------------------------------------------------------
-- 表1：rtu_record — 报文主记录
-- ------------------------------------------------------------
CREATE TABLE rtu_record (
    ID                              BIGINT NOT NULL IDENTITY(1,1),          -- 主键

    -- 固件/设备信息
    RTU_REGS_BUILD_TIME             INT DEFAULT NULL,                        -- 固件构建时间(秒)
    RTU_REGS_BUILD_TIME_2b          INT DEFAULT NULL,                        -- 固件构建时间2b
    RTU_REGS_BUILD_TIME_3b          INT DEFAULT NULL,                        -- 固件构建时间3b
    RTU_REGS_BUILD_TIME_4b          INT DEFAULT NULL,                        -- 固件构建时间4b
    RTU_REGS_UPTIME_SEC             INT DEFAULT NULL,                        -- 系统运行时间(秒)
    RTU_REGS_UPTIME_SEC_2b          INT DEFAULT NULL,                        -- 系统运行时间高位
    RTU_REGS_DEVICE_STATE           TINYINT DEFAULT NULL,                    -- 设备状态
    RTU_REGS_SENSOR_CNT_P1_P2       TINYINT DEFAULT NULL,                    -- P1+P2传感器数量
    RTU_REGS_SENSOR_CNT_P3_P4       TINYINT DEFAULT NULL,                    -- P3+P4传感器数量
    RTU_REGS_SENSOR_CNT_P5_Px       TINYINT DEFAULT NULL,                    -- P5+Px传感器数量

    -- 从机/命令
    RTU_REGS_SLAVE_ID               TINYINT DEFAULT NULL,                    -- Modbus从机地址
    RTU_REGS_EXEC_CMD               TINYINT DEFAULT NULL,                    -- 执行命令

    -- 传感器节点 P1
    RTU_REGS_NODE_P1_0              TINYINT DEFAULT NULL,                    -- P1通道节点0
    RTU_REGS_NODE_P1_1              TINYINT DEFAULT NULL,                    -- P1通道节点1
    RTU_REGS_NODE_P1_2              TINYINT DEFAULT NULL,                    -- P1通道节点2
    RTU_REGS_NODE_P1_3              TINYINT DEFAULT NULL,                    -- P1通道节点3
    RTU_REGS_NODE_P1_4              TINYINT DEFAULT NULL,                    -- P1通道节点4
    RTU_REGS_NODE_P1_5              TINYINT DEFAULT NULL,                    -- P1通道节点5
    RTU_REGS_NODE_P1_6              TINYINT DEFAULT NULL,                    -- P1通道节点6
    RTU_REGS_NODE_P1_7              TINYINT DEFAULT NULL,                    -- P1通道节点7
    RTU_REGS_NODE_P1_8              TINYINT DEFAULT NULL,                    -- P1通道节点8
    RTU_REGS_NODE_P1_9              TINYINT DEFAULT NULL,                    -- P1通道节点9

    -- 传感器节点 P2
    RTU_REGS_NODE_P2_0              TINYINT DEFAULT NULL,                    -- P2通道节点0
    RTU_REGS_NODE_P2_1              TINYINT DEFAULT NULL,                    -- P2通道节点1
    RTU_REGS_NODE_P2_2              TINYINT DEFAULT NULL,                    -- P2通道节点2
    RTU_REGS_NODE_P2_3              TINYINT DEFAULT NULL,                    -- P2通道节点3
    RTU_REGS_NODE_P2_4              TINYINT DEFAULT NULL,                    -- P2通道节点4
    RTU_REGS_NODE_P2_5              TINYINT DEFAULT NULL,                    -- P2通道节点5
    RTU_REGS_NODE_P2_6              TINYINT DEFAULT NULL,                    -- P2通道节点6
    RTU_REGS_NODE_P2_7              TINYINT DEFAULT NULL,                    -- P2通道节点7
    RTU_REGS_NODE_P2_8              TINYINT DEFAULT NULL,                    -- P2通道节点8
    RTU_REGS_NODE_P2_9              TINYINT DEFAULT NULL,                    -- P2通道节点9

    -- 传感器节点 P3
    RTU_REGS_NODE_P3_0              TINYINT DEFAULT NULL,                    -- P3通道节点0
    RTU_REGS_NODE_P3_1              TINYINT DEFAULT NULL,                    -- P3通道节点1
    RTU_REGS_NODE_P3_2              TINYINT DEFAULT NULL,                    -- P3通道节点2
    RTU_REGS_NODE_P3_3              TINYINT DEFAULT NULL,                    -- P3通道节点3
    RTU_REGS_NODE_P3_4              TINYINT DEFAULT NULL,                    -- P3通道节点4
    RTU_REGS_NODE_P3_5              TINYINT DEFAULT NULL,                    -- P3通道节点5
    RTU_REGS_NODE_P3_6              TINYINT DEFAULT NULL,                    -- P3通道节点6
    RTU_REGS_NODE_P3_7              TINYINT DEFAULT NULL,                    -- P3通道节点7
    RTU_REGS_NODE_P3_8              TINYINT DEFAULT NULL,                    -- P3通道节点8
    RTU_REGS_NODE_P3_9              TINYINT DEFAULT NULL,                    -- P3通道节点9

    -- 传感器节点 P4
    RTU_REGS_NODE_P4_0              TINYINT DEFAULT NULL,                    -- P4通道节点0
    RTU_REGS_NODE_P4_1              TINYINT DEFAULT NULL,                    -- P4通道节点1
    RTU_REGS_NODE_P4_2              TINYINT DEFAULT NULL,                    -- P4通道节点2
    RTU_REGS_NODE_P4_3              TINYINT DEFAULT NULL,                    -- P4通道节点3
    RTU_REGS_NODE_P4_4              TINYINT DEFAULT NULL,                    -- P4通道节点4
    RTU_REGS_NODE_P4_5              TINYINT DEFAULT NULL,                    -- P4通道节点5
    RTU_REGS_NODE_P4_6              TINYINT DEFAULT NULL,                    -- P4通道节点6
    RTU_REGS_NODE_P4_7              TINYINT DEFAULT NULL,                    -- P4通道节点7
    RTU_REGS_NODE_P4_8              TINYINT DEFAULT NULL,                    -- P4通道节点8
    RTU_REGS_NODE_P4_9              TINYINT DEFAULT NULL,                    -- P4通道节点9

    -- 传感器节点 P5
    RTU_REGS_NODE_P5_0              TINYINT DEFAULT NULL,                    -- P5通道节点0
    RTU_REGS_NODE_P5_1              TINYINT DEFAULT NULL,                    -- P5通道节点1
    RTU_REGS_NODE_P5_2              TINYINT DEFAULT NULL,                    -- P5通道节点2
    RTU_REGS_NODE_P5_3              TINYINT DEFAULT NULL,                    -- P5通道节点3
    RTU_REGS_NODE_P5_4              TINYINT DEFAULT NULL,                    -- P5通道节点4
    RTU_REGS_NODE_P5_5              TINYINT DEFAULT NULL,                    -- P5通道节点5
    RTU_REGS_NODE_P5_6              TINYINT DEFAULT NULL,                    -- P5通道节点6
    RTU_REGS_NODE_P5_7              TINYINT DEFAULT NULL,                    -- P5通道节点7
    RTU_REGS_NODE_P5_8              TINYINT DEFAULT NULL,                    -- P5通道节点8
    RTU_REGS_NODE_P5_9              TINYINT DEFAULT NULL,                    -- P5通道节点9

    -- P1 传感器温度
    RTU_REGS_TEMP_P1_0              SMALLINT DEFAULT NULL,                   -- P1通道节点0温度(0.1°C)
    RTU_REGS_TEMP_P1_1              SMALLINT DEFAULT NULL,                   -- P1通道节点1温度(0.1°C)
    RTU_REGS_TEMP_P1_2              SMALLINT DEFAULT NULL,                   -- P1通道节点2温度(0.1°C)
    RTU_REGS_TEMP_P1_3              SMALLINT DEFAULT NULL,                   -- P1通道节点3温度(0.1°C)
    RTU_REGS_TEMP_P1_4              SMALLINT DEFAULT NULL,                   -- P1通道节点4温度(0.1°C)
    RTU_REGS_TEMP_P1_5              SMALLINT DEFAULT NULL,                   -- P1通道节点5温度(0.1°C)
    RTU_REGS_TEMP_P1_6              SMALLINT DEFAULT NULL,                   -- P1通道节点6温度(0.1°C)
    RTU_REGS_TEMP_P1_7              SMALLINT DEFAULT NULL,                   -- P1通道节点7温度(0.1°C)
    RTU_REGS_TEMP_P1_8              SMALLINT DEFAULT NULL,                   -- P1通道节点8温度(0.1°C)
    RTU_REGS_TEMP_P1_9              SMALLINT DEFAULT NULL,                   -- P1通道节点9温度(0.1°C)

    -- P1 传感器湿度
    RTU_REGS_HUM_P1_0               SMALLINT DEFAULT NULL,                   -- P1通道节点0湿度(0.1%)
    RTU_REGS_HUM_P1_1               SMALLINT DEFAULT NULL,                   -- P1通道节点1湿度(0.1%)
    RTU_REGS_HUM_P1_2               SMALLINT DEFAULT NULL,                   -- P1通道节点2湿度(0.1%)
    RTU_REGS_HUM_P1_3               SMALLINT DEFAULT NULL,                   -- P1通道节点3湿度(0.1%)
    RTU_REGS_HUM_P1_4               SMALLINT DEFAULT NULL,                   -- P1通道节点4湿度(0.1%)
    RTU_REGS_HUM_P1_5               SMALLINT DEFAULT NULL,                   -- P1通道节点5湿度(0.1%)
    RTU_REGS_HUM_P1_6               SMALLINT DEFAULT NULL,                   -- P1通道节点6湿度(0.1%)
    RTU_REGS_HUM_P1_7               SMALLINT DEFAULT NULL,                   -- P1通道节点7湿度(0.1%)
    RTU_REGS_HUM_P1_8               SMALLINT DEFAULT NULL,                   -- P1通道节点8湿度(0.1%)
    RTU_REGS_HUM_P1_9               SMALLINT DEFAULT NULL,                   -- P1通道节点9湿度(0.1%)

    -- P00 参数
    RTU_REGS_P00_ROTOR_RPM          SMALLINT DEFAULT NULL,                   -- P00电机转速(0.1RPM)
    RTU_REGS_P00_ENV_HUMIDITY       SMALLINT DEFAULT NULL,                   -- P00环境湿度(0.1%)
    RTU_REGS_P00_ENV_TEMP           SMALLINT DEFAULT NULL,                   -- P00环境温度(0.1°C)
    RTU_REGS_P00_WAVE_BUSY          TINYINT DEFAULT NULL,                    -- P00波形采集忙标志

    -- 报文额外字段
    ACTUAL_VOLTAGE                  TINYINT DEFAULT NULL,                    -- 实际电压(V)
    ENV_TEMPERATURE                 TINYINT DEFAULT NULL,                    -- 环境温度(°C)
    ENV_HUMIDITY                    TINYINT DEFAULT NULL,                    -- 环境湿度(%)
    RUNNING_HOURS                   TINYINT DEFAULT NULL,                    -- 累计运行小时数
    INSTALLATION_ANGLE              TINYINT DEFAULT NULL,                    -- 安装角度(°)
    LINE_TYPE                       TINYINT DEFAULT NULL,                    -- 线路类型
    SYSTEM_TIME                     DATETIME DEFAULT NULL,                   -- 设备系统时间

    -- 记录入库时间
    CREATED_AT                      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP(), -- 记录入库时间

    PRIMARY KEY (ID)
);

CREATE INDEX IDX_RTU_RECORD_SLAVE_TIME ON rtu_record (RTU_REGS_SLAVE_ID, SYSTEM_TIME);


-- ------------------------------------------------------------
-- 表2：rtu_wave_data — P00 波形数据
-- ------------------------------------------------------------
CREATE TABLE rtu_wave_data (
    ID              BIGINT NOT NULL IDENTITY(1,1),       -- 主键
    RECORD_ID       BIGINT NOT NULL,                     -- 关联rtu_record.ID
    WAVE_DATA       TEXT DEFAULT NULL,                   -- P00波形数据(逗号分隔512个采样点)

    PRIMARY KEY (ID)
);

CREATE INDEX IDX_WAVE_RECORD ON rtu_wave_data (RECORD_ID);
ALTER TABLE rtu_wave_data ADD CONSTRAINT FK_WAVE_RECORD
    FOREIGN KEY (RECORD_ID) REFERENCES rtu_record(ID) ON DELETE CASCADE;


-- ============================================================
-- 注释
-- ============================================================
COMMENT ON TABLE  rtu_record IS 'RTU报文主记录';
COMMENT ON COLUMN rtu_record.ID IS '主键';
COMMENT ON COLUMN rtu_record.RTU_REGS_BUILD_TIME IS '固件生成日期';
COMMENT ON COLUMN rtu_record.RTU_REGS_BUILD_TIME_2b IS '固件生成日期2b';
COMMENT ON COLUMN rtu_record.RTU_REGS_BUILD_TIME_3b IS '固件生成日期3b';
COMMENT ON COLUMN rtu_record.RTU_REGS_BUILD_TIME_4b IS '固件生成日期4b';
COMMENT ON COLUMN rtu_record.RTU_REGS_UPTIME_SEC IS '系统运行时间(秒)';
COMMENT ON COLUMN rtu_record.RTU_REGS_UPTIME_SEC_2b IS '系统运行时间高位';
COMMENT ON COLUMN rtu_record.RTU_REGS_DEVICE_STATE IS '设备状态';
COMMENT ON COLUMN rtu_record.RTU_REGS_SENSOR_CNT_P1_P2 IS 'P1+P2传感器数量';
COMMENT ON COLUMN rtu_record.RTU_REGS_SENSOR_CNT_P3_P4 IS 'P3+P4传感器数量';
COMMENT ON COLUMN rtu_record.RTU_REGS_SENSOR_CNT_P5_Px IS 'P5+Px传感器数量';
COMMENT ON COLUMN rtu_record.RTU_REGS_SLAVE_ID IS 'Modbus从机地址';
COMMENT ON COLUMN rtu_record.RTU_REGS_EXEC_CMD IS '执行命令';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P1_0 IS 'P1通道节点0';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P1_1 IS 'P1通道节点1';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P1_2 IS 'P1通道节点2';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P1_3 IS 'P1通道节点3';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P1_4 IS 'P1通道节点4';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P1_5 IS 'P1通道节点5';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P1_6 IS 'P1通道节点6';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P1_7 IS 'P1通道节点7';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P1_8 IS 'P1通道节点8';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P1_9 IS 'P1通道节点9';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P2_0 IS 'P2通道节点0';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P2_1 IS 'P2通道节点1';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P2_2 IS 'P2通道节点2';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P2_3 IS 'P2通道节点3';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P2_4 IS 'P2通道节点4';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P2_5 IS 'P2通道节点5';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P2_6 IS 'P2通道节点6';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P2_7 IS 'P2通道节点7';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P2_8 IS 'P2通道节点8';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P2_9 IS 'P2通道节点9';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P3_0 IS 'P3通道节点0';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P3_1 IS 'P3通道节点1';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P3_2 IS 'P3通道节点2';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P3_3 IS 'P3通道节点3';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P3_4 IS 'P3通道节点4';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P3_5 IS 'P3通道节点5';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P3_6 IS 'P3通道节点6';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P3_7 IS 'P3通道节点7';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P3_8 IS 'P3通道节点8';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P3_9 IS 'P3通道节点9';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P4_0 IS 'P4通道节点0';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P4_1 IS 'P4通道节点1';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P4_2 IS 'P4通道节点2';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P4_3 IS 'P4通道节点3';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P4_4 IS 'P4通道节点4';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P4_5 IS 'P4通道节点5';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P4_6 IS 'P4通道节点6';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P4_7 IS 'P4通道节点7';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P4_8 IS 'P4通道节点8';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P4_9 IS 'P4通道节点9';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P5_0 IS 'P5通道节点0';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P5_1 IS 'P5通道节点1';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P5_2 IS 'P5通道节点2';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P5_3 IS 'P5通道节点3';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P5_4 IS 'P5通道节点4';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P5_5 IS 'P5通道节点5';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P5_6 IS 'P5通道节点6';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P5_7 IS 'P5通道节点7';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P5_8 IS 'P5通道节点8';
COMMENT ON COLUMN rtu_record.RTU_REGS_NODE_P5_9 IS 'P5通道节点9';
COMMENT ON COLUMN rtu_record.RTU_REGS_TEMP_P1_0 IS 'P1通道节点0温度(0.1°C)';
COMMENT ON COLUMN rtu_record.RTU_REGS_TEMP_P1_1 IS 'P1通道节点1温度(0.1°C)';
COMMENT ON COLUMN rtu_record.RTU_REGS_TEMP_P1_2 IS 'P1通道节点2温度(0.1°C)';
COMMENT ON COLUMN rtu_record.RTU_REGS_TEMP_P1_3 IS 'P1通道节点3温度(0.1°C)';
COMMENT ON COLUMN rtu_record.RTU_REGS_TEMP_P1_4 IS 'P1通道节点4温度(0.1°C)';
COMMENT ON COLUMN rtu_record.RTU_REGS_TEMP_P1_5 IS 'P1通道节点5温度(0.1°C)';
COMMENT ON COLUMN rtu_record.RTU_REGS_TEMP_P1_6 IS 'P1通道节点6温度(0.1°C)';
COMMENT ON COLUMN rtu_record.RTU_REGS_TEMP_P1_7 IS 'P1通道节点7温度(0.1°C)';
COMMENT ON COLUMN rtu_record.RTU_REGS_TEMP_P1_8 IS 'P1通道节点8温度(0.1°C)';
COMMENT ON COLUMN rtu_record.RTU_REGS_TEMP_P1_9 IS 'P1通道节点9温度(0.1°C)';
COMMENT ON COLUMN rtu_record.RTU_REGS_HUM_P1_0 IS 'P1通道节点0湿度(0.1%)';
COMMENT ON COLUMN rtu_record.RTU_REGS_HUM_P1_1 IS 'P1通道节点1湿度(0.1%)';
COMMENT ON COLUMN rtu_record.RTU_REGS_HUM_P1_2 IS 'P1通道节点2湿度(0.1%)';
COMMENT ON COLUMN rtu_record.RTU_REGS_HUM_P1_3 IS 'P1通道节点3湿度(0.1%)';
COMMENT ON COLUMN rtu_record.RTU_REGS_HUM_P1_4 IS 'P1通道节点4湿度(0.1%)';
COMMENT ON COLUMN rtu_record.RTU_REGS_HUM_P1_5 IS 'P1通道节点5湿度(0.1%)';
COMMENT ON COLUMN rtu_record.RTU_REGS_HUM_P1_6 IS 'P1通道节点6湿度(0.1%)';
COMMENT ON COLUMN rtu_record.RTU_REGS_HUM_P1_7 IS 'P1通道节点7湿度(0.1%)';
COMMENT ON COLUMN rtu_record.RTU_REGS_HUM_P1_8 IS 'P1通道节点8湿度(0.1%)';
COMMENT ON COLUMN rtu_record.RTU_REGS_HUM_P1_9 IS 'P1通道节点9湿度(0.1%)';
COMMENT ON COLUMN rtu_record.RTU_REGS_P00_ROTOR_RPM IS 'P00电机转速(0.1RPM)';
COMMENT ON COLUMN rtu_record.RTU_REGS_P00_ENV_HUMIDITY IS 'P00环境湿度(0.1%)';
COMMENT ON COLUMN rtu_record.RTU_REGS_P00_ENV_TEMP IS 'P00环境温度(0.1°C)';
COMMENT ON COLUMN rtu_record.RTU_REGS_P00_WAVE_BUSY IS 'P00波形采集忙标志';
COMMENT ON COLUMN rtu_record.ACTUAL_VOLTAGE IS '实际电压(V)';
COMMENT ON COLUMN rtu_record.ENV_TEMPERATURE IS '环境温度(°C)';
COMMENT ON COLUMN rtu_record.ENV_HUMIDITY IS '环境湿度(%)';
COMMENT ON COLUMN rtu_record.RUNNING_HOURS IS '累计运行小时数';
COMMENT ON COLUMN rtu_record.INSTALLATION_ANGLE IS '安装角度(°)';
COMMENT ON COLUMN rtu_record.LINE_TYPE IS '线路类型';
COMMENT ON COLUMN rtu_record.SYSTEM_TIME IS '设备系统时间';
COMMENT ON COLUMN rtu_record.CREATED_AT IS '记录入库时间';

COMMENT ON TABLE  rtu_wave_data IS 'P00波形数据';
COMMENT ON COLUMN rtu_wave_data.ID IS '主键';
COMMENT ON COLUMN rtu_wave_data.RECORD_ID IS '关联rtu_record.ID';
COMMENT ON COLUMN rtu_wave_data.WAVE_DATA IS 'P00波形数据(逗号分隔512个采样点)';
