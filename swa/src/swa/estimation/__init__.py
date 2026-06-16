# 电压估计算法模块
#
# 每个文件一种算法，统一接口：
#   NAME: str          — 算法名称
#   train(X, y) → model — 训练
#   predict(model, X) → y_pred — 预测
#
# 特征 X 的列顺序（由 feature_extractor 统一提取，共 18 维）：
#   [A1..A10, T, RH, RPM, Vpp, RMS, CrestFactor, Kurtosis, Skewness]
#
# 线性模型只用前 13 维（A1~RPM），XGBoost/LGBM 自动使用全部 18 维
