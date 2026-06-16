"""实验：LeNet-Hybrid 不同谐波数的效果"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from src.swa.signal_process.loader import load_jsonl
from src.swa.estimation import lenet_hybrid as module
from src.swa.config.settings import config

# 全量 38000 条，固定 100 轮，不早停
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 38000

all_records = load_jsonl("data/exported_data.jsonl")[:limit]

np.random.seed(42)
np.random.shuffle(all_records)

records = all_records
print(f"使用全部 {len(records)} 条")

train_n = int(len(records) * 0.8)
val_n = int(len(records) * 0.1)
test_n = len(records) - train_n - val_n

train_records = records[:train_n]
val_records = records[train_n:train_n + val_n]
test_records = records[train_n + val_n:train_n + val_n + test_n]

print(f"总数据: {len(records)} 条, 训练集: {len(train_records)}, 验证集: {len(val_records)}, 测试集: {len(test_records)}")

results = []
for n_fft in [6, 7, 8, 9, 10, 11, 12, 13, 14]:
    print(f"\n{'='*50}")
    print(f"测试 n_fft = {n_fft}")
    print(f"{'='*50}")
    t0 = time.time()
    # 固定 100 轮，不早停，warmup=0 取消保护
    model = module.train(train_records, val_records=val_records, test_records=test_records,
                         n_fft=n_fft, epochs=100, patience=999, warmup=0)
    # 测试评估
    (wave_t, fft_t, env_t), y_test, _ = module._build_tensors(test_records, n_fft=n_fft)
    with torch.no_grad():
        y_pred = model["model"](wave_t, fft_t, env_t).numpy()
    mae = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    max_err = float(np.max(np.abs(y_pred - y_test)))
    elapsed = time.time() - t0
    results.append((n_fft, mae, rmse, max_err))
    print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  MaxErr={max_err:.4f}  ({elapsed:.0f}s)")

print("\n" + "="*50)
print("最终结果:")
print(f"{'n_fft':>6}  {'MAE':>8}  {'RMSE':>8}  {'MaxErr':>10}")
for n_fft, mae, rmse, max_err in results:
    print(f"     {n_fft:>2}:  {mae:>8.4f}  {rmse:>8.4f}  {max_err:>10.4f}")
