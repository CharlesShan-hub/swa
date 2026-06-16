"""
LeNet-Hybrid：时域 + 频域 + 环境参数 — 三路融合
支持 Mini-Batch + 早停 + 学习率衰减。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
from scipy.fftpack import fft
from scipy.stats import kurtosis, skew

from ..config.settings import config

_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)

NAME = "LeNet-Hybrid（Mini-Batch）"


class HybridNet(nn.Module):
    def __init__(self, n_fft=10, n_env=6, init_method="xavier"):
        super().__init__()
        self.n_fft = n_fft
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(4),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 32 + n_fft + n_env, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._init_weights(init_method)

    def _init_weights(self, method="xavier"):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, wave, fft_feat, env):
        x = wave.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        combined = torch.cat([x, fft_feat, env], dim=1)
        return self.fc(combined).squeeze(1)


def _normalize(arr, mean=None, std=None):
    if mean is None:
        mean = np.mean(arr, axis=0, keepdims=True)
        std = np.std(arr, axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std, mean, std


def _extract(records: list[dict], n_fft: int = 10):
    """批量提取波形、FFT、环境三类输入"""
    wave_list, fft_list, env_list, y_list = [], [], [], []
    for rec in records:
        wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        vals = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]

        def _f(v, d=0.0):
            try: return float(v)
            except: return d

        temp = _f(rec.get("RTU_REGS_P00_ENV_TEMP"))
        humid = _f(rec.get("RTU_REGS_P00_ENV_HUMIDITY"))
        rpm = _f(rec.get("RTU_REGS_P00_ROTOR_RPM"))

        ac = vals - np.mean(vals)
        n = len(ac)
        fft_mag = np.abs(fft(ac))[:n // 2]
        harmonics = np.zeros(n_fft)
        for i in range(n_fft):
            idx = i + 1
            if idx < len(fft_mag):
                harmonics[i] = 2.0 * fft_mag[idx] / n

        # 时域特征（独立于 FFT 的信息）
        ac = vals - np.mean(vals)
        vpp = float(np.max(ac) - np.min(ac))       # 峰峰值，反映总波动
        kurt = float(kurtosis(ac, fisher=False))    # 峭度，波形尖锐度
        skewness = float(skew(ac))                  # 偏度，正负不对称

        v_str = str(rec.get("ACTUAL_VOLTAGE", "")).lower().replace("v", "").strip()
        try: voltage = float(v_str)
        except: continue

        wave_list.append(vals)
        fft_list.append(harmonics)
        env_list.append([temp, humid, rpm, vpp, kurt, skewness])
        y_list.append(voltage)

    return (np.array(wave_list), np.array(fft_list),
            np.array(env_list), np.array(y_list))


def _build_tensors(records, norm_params=None, n_fft=10):
    """从记录列表构建归一化后的张量"""
    wave_arr, fft_arr, env_arr, y_arr = _extract(records, n_fft=n_fft)

    # 波形归一化（逐条）
    wm = np.mean(wave_arr, axis=1, keepdims=True)
    ws = np.std(wave_arr, axis=1, keepdims=True) + 1e-8
    wave_norm = (wave_arr - wm) / ws

    # FFT 和环境整体归一化
    if norm_params:
        fft_norm = (fft_arr - norm_params["fft_mean"]) / norm_params["fft_std"]
        env_norm = (env_arr - norm_params["env_mean"]) / norm_params["env_std"]
    else:
        fft_norm, fm, fs = _normalize(fft_arr)
        env_norm, em, es = _normalize(env_arr)
        norm_params = {"fft_mean": fm, "fft_std": fs, "env_mean": em, "env_std": es}

    return (torch.tensor(wave_norm, dtype=torch.float32),
            torch.tensor(fft_norm, dtype=torch.float32),
            torch.tensor(env_norm, dtype=torch.float32)), y_arr, norm_params


def _eval(model, wave_t, fft_t, env_t, labels_t, loss_fn):
    """评估集上的 loss 和指标"""
    model.eval()
    with torch.no_grad():
        preds = model(wave_t, fft_t, env_t)
        loss = loss_fn(preds, labels_t).item()
        mae = torch.mean(torch.abs(preds - labels_t)).item()
        rmse = torch.sqrt(torch.mean((preds - labels_t) ** 2)).item()
        max_err = torch.max(torch.abs(preds - labels_t)).item()
    model.train()
    return loss, mae, rmse, max_err


def train(train_records, val_records=None, test_records=None, epochs=None, lr=None, batch_size=256, n_fft=10,
          patience=None, warmup=None, weight_decay=None):
    """训练 LeNet-Hybrid，支持 Mini-Batch + 早停 + 学习率衰减。
    
    Args:
        patience: 早停耐心值（覆盖配置，None则用配置值）
        warmup: 热身轮数（覆盖配置，None则用30）
        weight_decay: L2 正则化系数（None则用1e-4）
    """
    cfg = config.estimation
    if epochs is None or epochs == -1:
        epochs = cfg.max_epochs if cfg.max_epochs != -1 else 999999
    if lr is None:
        lr = 0.005
    if patience is None:
        patience = cfg.early_stop_patience
    if warmup is None:
        warmup = 30
    if weight_decay is None:
        weight_decay = 1e-4
    max_lrn = cfg.lrn

    # 构建数据
    (wave_t, fft_t, env_t), y_train, norm_params = _build_tensors(train_records, n_fft=n_fft)
    labels_t = torch.tensor(y_train, dtype=torch.float32)

    has_val = val_records is not None
    test_wave_t = test_fft_t = test_env_t = test_labels_t = None
    val_wave_t = val_fft_t = val_env_t = val_labels_t = None

    if has_val:
        (val_wave_t, val_fft_t, val_env_t), y_val, _ = _build_tensors(val_records, norm_params, n_fft=n_fft)
        val_labels_t = torch.tensor(y_val, dtype=torch.float32)
    if test_records is not None:
        (test_wave_t, test_fft_t, test_env_t), y_test, _ = _build_tensors(test_records, norm_params, n_fft=n_fft)
        test_labels_t = torch.tensor(y_test, dtype=torch.float32)

    # DataLoader
    dataset = TensorDataset(wave_t, fft_t, env_t, labels_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = HybridNet(n_fft=n_fft, n_env=env_t.size(1))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 从配置选择损失函数
    loss_fn_name = cfg.loss_fn
    if loss_fn_name == "huber":
        loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
    elif loss_fn_name == "l1":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_state = None
    bad_epochs = 0
    lrn_left = max_lrn
    epoch = 0

    while epoch < epochs:
        epoch += 1
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for wb, fb, eb, lb in loader:
            optimizer.zero_grad()
            preds = model(wb, fb, eb)
            loss = loss_fn(preds, lb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        if has_val:
            val_loss, val_mae, val_rmse, val_max = _eval(
                model, val_wave_t, val_fft_t, val_env_t, val_labels_t, loss_fn
            )

            if epoch % 10 == 0:
                msg = f"  Epoch {epoch}  train_loss={avg_train_loss:.2f}  val_loss={val_loss:.2f}"
                if test_labels_t is not None:
                    _, test_mae, test_rmse, test_max = _eval(
                        model, test_wave_t, test_fft_t, test_env_t, test_labels_t, loss_fn
                    )
                    msg += f"  test_mae={test_mae:.2f}  test_rmse={test_rmse:.2f}  test_max={test_max:.2f}"
                print(msg)

            # warmup
            if epoch <= warmup:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                if epoch == warmup:
                    best_val_loss = float("inf")
                    bad_epochs = 0
                continue

            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                if lrn_left > 0:
                    lrn_left -= 1
                    for g in optimizer.param_groups:
                        g["lr"] *= cfg.lr_decay
                    new_lr = optimizer.param_groups[0]["lr"]
                    print(f"  → Epoch {epoch}: lr 衰减至 {new_lr:.6f}（剩余: {lrn_left}）")
                    bad_epochs = 0
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        # 重置优化器动量，避免错误方向的记忆影响后续训练
                        optimizer = optim.AdamW(model.parameters(), lr=new_lr, weight_decay=weight_decay)
                else:
                    print(f"  → Epoch {epoch}: 早停（最佳 val_loss={best_val_loss:.2f}）")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
        else:
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}  loss={avg_train_loss:.2f}")

    if has_val and best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    return {"model": model, **norm_params}


def predict(model_dict: dict, record_or_X):
    """预测电压"""
    model = model_dict["model"]

    if isinstance(record_or_X, dict):
        rec = record_or_X
        wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]

        def _f(v, d=0.0):
            try: return float(v)
            except: return d
        temp = _f(rec.get("RTU_REGS_P00_ENV_TEMP"))
        humid = _f(rec.get("RTU_REGS_P00_ENV_HUMIDITY"))
        rpm = _f(rec.get("RTU_REGS_P00_ROTOR_RPM"))

        ac = wave - np.mean(wave)
        n = len(ac)
        fft_mag = np.abs(fft(ac))[:n // 2]
        harmonics = np.zeros(10)
        for i in range(10):
            idx = i + 1
            if idx < len(fft_mag):
                harmonics[i] = 2.0 * fft_mag[idx] / n

        wm = np.mean(wave); ws = np.std(wave) + 1e-8
        wave_norm = ((wave - wm) / ws).reshape(1, -1)
        fft_norm = ((harmonics - model_dict["fft_mean"]) / model_dict["fft_std"]).reshape(1, -1)

        # 时域特征
        ac = wave - np.mean(wave)
        vpp = float(np.max(ac) - np.min(ac))
        kurt = float(kurtosis(ac, fisher=False))
        skewness = float(skew(ac))
        aux = np.array([temp, humid, rpm, vpp, kurt, skewness])
        env_norm = ((aux - model_dict["env_mean"]) / model_dict["env_std"]).reshape(1, -1)

        wave_t = torch.tensor(wave_norm, dtype=torch.float32)
        fft_t = torch.tensor(fft_norm, dtype=torch.float32)
        env_t = torch.tensor(env_norm, dtype=torch.float32)
    else:
        wave_t = torch.tensor(record_or_X[0], dtype=torch.float32)
        fft_t = torch.tensor(record_or_X[1], dtype=torch.float32)
        env_t = torch.tensor(record_or_X[2], dtype=torch.float32)

    with torch.no_grad():
        return model(wave_t, fft_t, env_t).numpy()
