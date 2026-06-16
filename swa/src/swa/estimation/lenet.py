"""
LeNet-1D：直接从 512 点原始波形预测电压。

支持早停 + 学习率衰减：
- epochs=-1 表示无限轮，靠早停停止
- 验证 loss 连续 3 次上升 → lr × 0.1，最多衰减 lrn 次
- 衰减次数用尽且 loss 仍不降 → 停止训练
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from scripts.utils.device import get_device

from ..config.settings import config

_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)

_DEVICE = get_device()

NAME = "LeNet-1D"


class LeNet1D(nn.Module):
    def __init__(self, n_env=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 128 + n_env, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, wave, env=None):
        x = wave.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if env is not None:
            x = torch.cat([x, env], dim=1)
        return self.fc(x).squeeze(1)


def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val: np.ndarray = None, y_val: np.ndarray = None,
          X_test: np.ndarray = None, y_test: np.ndarray = None,
          epochs: int = None, lr: float = None):
    """
    训练 LeNet-1D，支持早停 + 学习率衰减。

    Args:
        X_train: shape=(n, 515)，训练特征
        y_train: 训练标签
        X_val, y_val: 验证集（用于早停）
        X_test, y_test: 测试集（每10轮打印指标）
        epochs: 最大轮数（None 或 -1 表示无限）
        lr: 初始学习率
    """
    # 从配置读取默认值
    cfg = config.estimation
    if epochs is None or epochs == -1:
        epochs = cfg.max_epochs if cfg.max_epochs != -1 else 999999
    if lr is None:
        lr = 0.01
    max_lrn = cfg.lrn

    # 归一化
    def _norm(X):
        w = X[:, :512]
        w_mean = np.mean(w, axis=1, keepdims=True)
        w_std = np.std(w, axis=1, keepdims=True) + 1e-8
        return (w - w_mean) / w_std, w_mean[:1], w_std[:1]

    wave_norm, wm, ws = _norm(X_train)
    wave_t = torch.tensor(wave_norm, dtype=torch.float32).to(_DEVICE)
    env_t = torch.tensor(X_train[:, 512:], dtype=torch.float32).to(_DEVICE)
    labels_t = torch.tensor(y_train, dtype=torch.float32).to(_DEVICE)

    # 验证集 & 测试集
    val_wave_t = val_env_t = val_labels_t = None
    test_wave_t = test_env_t = test_labels_t = None
    has_val = X_val is not None and y_val is not None
    if has_val:
        vn, _, _ = _norm(X_val)
        val_wave_t = torch.tensor(vn, dtype=torch.float32).to(_DEVICE)
        val_env_t = torch.tensor(X_val[:, 512:], dtype=torch.float32).to(_DEVICE)
        val_labels_t = torch.tensor(y_val, dtype=torch.float32).to(_DEVICE)
    if X_test is not None and y_test is not None:
        tn, _, _ = _norm(X_test)
        test_wave_t = torch.tensor(tn, dtype=torch.float32).to(_DEVICE)
        test_env_t = torch.tensor(X_test[:, 512:], dtype=torch.float32).to(_DEVICE)
        test_labels_t = torch.tensor(y_test, dtype=torch.float32).to(_DEVICE)

    model = LeNet1D().to(_DEVICE)
    print(f"  设备: {_DEVICE}")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_state = None
    bad_epochs = 0
    patience = cfg.early_stop_patience
    lrn_left = max_lrn
    warmup = 30  # 前 30 轮不触发早停，让模型先稳定
    epoch = 0

    model.train()
    while epoch < epochs:
        epoch += 1

        # 训练一步
        optimizer.zero_grad()
        preds = model(wave_t, env_t)
        loss = loss_fn(preds, labels_t)
        loss.backward()
        optimizer.step()

        # 验证
        if has_val:
            model.eval()
            with torch.no_grad():
                val_preds = model(val_wave_t, val_env_t)
                val_loss = loss_fn(val_preds, val_labels_t).item()
            model.train()

            if (epoch) % 10 == 0:
                msg = f"  Epoch {epoch}  train_loss={loss.item():.2f}  val_loss={val_loss:.2f}"
                # 每10轮打印测试指标
                if test_labels_t is not None:
                    model.eval()
                    with torch.no_grad():
                        test_preds = model(test_wave_t, test_env_t)
                        test_mae = torch.mean(torch.abs(test_preds - test_labels_t)).item()
                        test_rmse = torch.sqrt(torch.mean((test_preds - test_labels_t) ** 2)).item()
                        test_max = torch.max(torch.abs(test_preds - test_labels_t)).item()
                    model.train()
                    msg += f"  test_mae={test_mae:.2f}  test_rmse={test_rmse:.2f}  test_max={test_max:.2f}"
                print(msg)

            # 前 warmup 轮只记录最佳，不触发早停
            if epoch <= warmup:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                # warmup 结束时重置，重新计数
                if epoch == warmup:
                    best_val_loss = float("inf")
                    bad_epochs = 0
                continue

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                bad_epochs = 0
            else:
                bad_epochs += 1

            # 连续 patience 次未改善 → 衰减学习率
            if bad_epochs >= patience:
                if lrn_left > 0:
                    lrn_left -= 1
                    for g in optimizer.param_groups:
                        g["lr"] *= cfg.lr_decay
                    new_lr = optimizer.param_groups[0]["lr"]
                    print(f"  → Epoch {epoch}: lr 衰减至 {new_lr:.6f}（剩余衰减次数: {lrn_left}）")
                    bad_epochs = 0
                    # 恢复到最佳模型 + 重置优化器动量
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        optimizer = optim.AdamW(model.parameters(), lr=new_lr, weight_decay=1e-4)
                else:
                    print(f"  → Epoch {epoch}: 衰减次数用尽，早停（最佳 val_loss={best_val_loss:.2f}）")
                    # 加载最佳模型
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
        else:
            if (epoch) % 10 == 0:
                print(f"  Epoch {epoch}  loss={loss.item():.2f}")

    # 无验证集时取最后状态
    if has_val and best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    return {"model": model, "wave_mean": wm, "wave_std": ws}


def predict(model_dict: dict, X_raw: np.ndarray) -> np.ndarray:
    """预测电压"""
    model = model_dict["model"]
    wave_raw = X_raw[:, :512]
    if "wave_mean" in model_dict:
        wm = model_dict["wave_mean"]
        ws = model_dict["wave_std"]
    else:
        wm = np.mean(wave_raw, axis=1, keepdims=True)
        ws = np.std(wave_raw, axis=1, keepdims=True) + 1e-8
    wave_norm = (wave_raw - wm) / ws
    wave = torch.tensor(wave_norm, dtype=torch.float32).to(_DEVICE)
    env = torch.tensor(X_raw[:, 512:], dtype=torch.float32).to(_DEVICE)
    with torch.no_grad():
        return model(wave, env).cpu().numpy()
