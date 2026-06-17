"""
LeNet-BiPath v2：a+b+c+d+e 五模块架构（b 独立于 a）。

设计思路：
  输入 → a（共享浅层卷积）→ c（绝对值头）
       → b（独立微型卷积）→ b（符号头）    ← b 完全不依赖 a
         a → conv3~4（更深卷积）→ d（精细绝对值头）
              └── e（整合层：融合 c 和 d → 最终绝对值）
              └── final = sign(b) × e(c, d)

三阶段训练：
  Phase 1 (epoch 1~N): 只训练 b（符号），b 独立路径不影响 a
  Phase 2 (epoch N+~M): 训练 a+c（绝对值）
  Phase 3 (epoch M+):   冻结 a/c，训练 conv_d + d + e 全链条
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
from scipy.fftpack import fft

from scripts.utils.device import get_device

from ..config.settings import config

_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)

_DEVICE = get_device()

NAME = "LeNet-BiPath v2（a+b+c+d+e）"


class BiPathNetV2(nn.Module):
    def __init__(self, n_fft=10, n_env=6, wave_len=512, init_method="xavier"):
        super().__init__()
        self.n_fft = n_fft
        self.n_env = n_env
        self.wave_len = wave_len

        # a: 共享浅层 Conv1d×2 → wave_len // 16
        conv_out = wave_len // 16
        self.conv_a = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(4),
        )
        conv_a_flat = 16 * conv_out

        # b: 独立符号预测头 — 不从 conv_a 取特征，用自己的微型网络
        self.b_conv = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.AvgPool1d(4),  # 512 → 128
        )
        b_conv_flat = 4 * (wave_len // 4)  # 4 × 128 = 512
        self.b_fc = nn.Sequential(
            nn.Linear(b_conv_flat + n_fft + n_env, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        # c: 绝对值预测头（粗略）
        self.c_fc = nn.Sequential(
            nn.Linear(conv_a_flat + n_fft + n_env, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

        # d 的深层卷积
        self.conv_d = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(4),
        )
        deep_out = wave_len // 256
        conv_d_flat = 64 * deep_out

        # d: 精细绝对值头（从深层特征预测）
        self.d_fc = nn.Sequential(
            nn.Linear(conv_d_flat + n_fft + n_env, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

        # e: 整合层 — 融合 c（粗略）和 d（精细）为最终绝对值
        self.e_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU(),
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
        x = wave.unsqueeze(1)               # (B, 1, 512)

        # a: 共享浅层卷积 → c 头用
        a_out = self.conv_a(x)               # (B, 16, 32)
        a_flat = a_out.view(a_out.size(0), -1)
        combined_a = torch.cat([a_flat, fft_feat, env], dim=1)

        # b: 独立符号路径 — 用自己的微型卷积，完全不依赖 conv_a
        b_feat = self.b_conv(x)               # (B, 4, 128)
        b_flat = b_feat.view(b_feat.size(0), -1)
        combined_b = torch.cat([b_flat, fft_feat, env], dim=1)
        b_out = self.b_fc(combined_b).squeeze(1)

        # c: 粗略绝对值（依赖 conv_a）
        c_out = self.c_fc(combined_a).squeeze(1)

        # d: 深层特征 → 精细绝对值
        d_feat = self.conv_d(a_out)          # (B, 64, 2)
        d_flat = d_feat.view(d_feat.size(0), -1)
        combined_d = torch.cat([d_flat, fft_feat, env], dim=1)
        d_out = self.d_fc(combined_d).squeeze(1)

        # e: 整合 c 和 d
        e_in = torch.stack([c_out, d_out], dim=1)  # (B, 2)
        e_out = self.e_fc(e_in).squeeze(1)

        # 最终输出 = 符号 × 整合后的绝对值
        final = torch.sign(b_out) * e_out

        return b_out, c_out, d_out, e_out, final


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
        vals = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)

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

        # 从 record 中读取预提取的时域特征
        vpp = _f(rec.get("vpp"))
        kurt = _f(rec.get("kurtosis"))
        skewness = _f(rec.get("skewness"))

        v_str = str(rec.get("ACTUAL_VOLTAGE", "")).lower().replace("v", "").strip()
        try:
            voltage = float(v_str)
        except:
            continue

        wave_list.append(vals)
        fft_list.append(harmonics)
        env_list.append([temp, humid, rpm, vpp, kurt, skewness])
        y_list.append(voltage)

    return (np.array(wave_list), np.array(fft_list),
            np.array(env_list), np.array(y_list))


def _build_tensors(records, norm_params=None, n_fft=10):
    """从记录列表构建归一化后的张量"""
    wave_arr, fft_arr, env_arr, y_arr = _extract(records, n_fft=n_fft)

    wm = np.mean(wave_arr, axis=1, keepdims=True)
    ws = np.std(wave_arr, axis=1, keepdims=True) + 1e-8
    wave_norm = (wave_arr - wm) / ws

    if norm_params:
        fft_norm = (fft_arr - norm_params["fft_mean"]) / norm_params["fft_std"]
        env_norm = (env_arr - norm_params["env_mean"]) / norm_params["env_std"]
    else:
        fft_norm, fm, fs = _normalize(fft_arr)
        env_norm, em, es = _normalize(env_arr)
        norm_params = {"fft_mean": fm, "fft_std": fs, "env_mean": em, "env_std": es}

    return (torch.tensor(wave_norm, dtype=torch.float32).to(_DEVICE),
            torch.tensor(fft_norm, dtype=torch.float32).to(_DEVICE),
            torch.tensor(env_norm, dtype=torch.float32).to(_DEVICE)), y_arr, norm_params


def _eval(model, wave_t, fft_t, env_t, labels_t, loss_fn):
    """评估集上的完整指标，返回 (loss, mae, rmse, max_err, sign_acc, c_mae, sign_errors, n_total)"""
    model.eval()
    with torch.no_grad():
        b, c, d, e, final = model(wave_t, fft_t, env_t)
        loss = loss_fn(final, labels_t).item()
        mae = torch.mean(torch.abs(final - labels_t)).item()
        rmse = torch.sqrt(torch.mean((final - labels_t) ** 2)).item()
        max_err = torch.max(torch.abs(final - labels_t)).item()
        # 符号准确率 & 错误数
        sign_pred = torch.sign(b)
        sign_true = torch.sign(labels_t)
        sign_correct = (sign_pred == sign_true)
        sign_acc = sign_correct.float().mean().item()
        n_total = labels_t.size(0)
        n_sign_errors = n_total - sign_correct.sum().item()
        # c 头绝对值 MAE
        c_mae = torch.mean(torch.abs(c - torch.abs(labels_t))).item()
    model.train()
    return loss, mae, rmse, max_err, sign_acc, c_mae, int(n_sign_errors), n_total


def train(train_records, val_records=None, test_records=None, epochs=None, lr=None, batch_size=256, n_fft=10,
          patience=None, warmup=None, weight_decay=None,
          symbol_epochs=5, phase1_epochs=40, phase1_only=False):
    """三阶段训练 LeNet-BiPath v2（b 独立于 a）。

    Phase 1 (符号):     训练 b（独立微型路径），不依赖 conv_a
    Phase 2 (绝对值):   训练 a+c，b 不受影响
    Phase 3 (全链条):   冻结 a/c，训练 conv_d + d_fc + e_fc，基于 final 输出

    Args:
        patience: 早停耐心值
        warmup: 热身轮数
        weight_decay: L2 正则化系数
        symbol_epochs: Phase 1 轮数（默认 5，99%+ 就够）
        phase1_epochs: Phase 1 + Phase 2 总轮数（默认 40）
        phase1_only: 仅执行 Phase 1+2，不进入 Phase 3
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

    # ── 构建数据 ──
    (wave_t, fft_t, env_t), y_train, norm_params = _build_tensors(train_records, n_fft=n_fft)
    labels_t = torch.tensor(y_train, dtype=torch.float32).to(_DEVICE)

    has_val = val_records is not None
    test_wave_t = test_fft_t = test_env_t = test_labels_t = None
    val_wave_t = val_fft_t = val_env_t = val_labels_t = None

    if has_val:
        (val_wave_t, val_fft_t, val_env_t), y_val, _ = _build_tensors(val_records, norm_params, n_fft=n_fft)
        val_labels_t = torch.tensor(y_val, dtype=torch.float32).to(_DEVICE)
    if test_records is not None:
        (test_wave_t, test_fft_t, test_env_t), y_test, _ = _build_tensors(test_records, norm_params, n_fft=n_fft)
        test_labels_t = torch.tensor(y_test, dtype=torch.float32).to(_DEVICE)

    _generator = torch.Generator().manual_seed(_SEED)
    dataset = TensorDataset(wave_t, fft_t, env_t, labels_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=_generator)

    model = BiPathNetV2(n_fft=n_fft, n_env=env_t.size(1), wave_len=wave_t.size(1)).to(_DEVICE)
    print(f"  设备: {_DEVICE}")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_state = None
    bad_epochs = 0
    lrn_left = max_lrn
    epoch = 0

    # ── 当前阶段 ──
    def _current_phase(e):
        if e <= symbol_epochs:
            return 1  # Phase 1: 符号
        elif e <= phase1_epochs or phase1_only:
            return 2  # Phase 2: 绝对值
        else:
            return 3  # Phase 3: 全链条

    # ── 训练循环 ──
    while epoch < epochs:
        epoch += 1
        phase = _current_phase(epoch)

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for wb, fb, eb, lb in loader:
            optimizer.zero_grad()
            b_out, c_out, d_out, e_out, final = model(wb, fb, eb)

            if phase == 1:
                # Phase 1: 训练 b（符号）- 独立路径，不依赖 conv_a
                loss = loss_fn(b_out, torch.sign(lb))
            elif phase == 2:
                # Phase 2: 只训练 a+c（绝对值）
                loss = loss_fn(c_out, torch.abs(lb))
            else:
                # Phase 3: 全链条 final
                loss = loss_fn(final, lb)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # ── 阶段切换：冻结/解冻 ──
        if epoch == symbol_epochs and phase1_epochs > symbol_epochs:
            # Phase 1 → 2: b 已独立，无需冻结，直接进入 a+c 训练
            val_n_err, val_n_tot = 0, 0
            if has_val:
                _, _, _, _, _, _, val_n_err, val_n_tot = _eval(
                    model, val_wave_t, val_fft_t, val_env_t, val_labels_t, loss_fn
                )
            print(f"  → Phase 1→2: b 已独立（{val_n_err}/{val_n_tot} 符号错误），开始 a+c 绝对值训练")

        if epoch == phase1_epochs and not phase1_only and phase1_epochs > symbol_epochs:
            # Phase 2 → 3: 冻结 a+c，训练 conv_d + d + e
            # b 独立不需要冻结
            for name, param in model.named_parameters():
                if name.startswith("conv_a") or name.startswith("c_fc"):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=weight_decay
            )
            print(f"  → Phase 2→3: 冻结 a/c，全链条微调（conv_d + d + e）")

        # ── 评估 ──
        if has_val:
            val_loss, val_mae, val_rmse, val_max, val_sign_acc, val_c_mae, val_n_err, val_n_tot = _eval(
                model, val_wave_t, val_fft_t, val_env_t, val_labels_t, loss_fn
            )

            if epoch % 5 == 0 or epoch == 1 or epoch == symbol_epochs or epoch == phase1_epochs:
                msg = f"  [Phase{phase}] Epoch {epoch}"

                if phase == 1:
                    msg += f"  sign={val_sign_acc*100:.1f}% ({val_n_err}/{val_n_tot})  c_mae={val_c_mae:.2f}V"
                elif phase == 2:
                    msg += f"  c_mae={val_c_mae:.2f}V  sign={val_sign_acc*100:.1f}% ({val_n_err}/{val_n_tot})"
                else:
                    msg += f"  val_mae={val_mae:.2f}  val_rmse={val_rmse:.2f}  val_max={val_max:.2f}"

                if test_labels_t is not None:
                    _, test_mae, test_rmse, test_max, test_sign_acc, test_c_mae, test_n_err, test_n_tot = _eval(
                        model, test_wave_t, test_fft_t, test_env_t, test_labels_t, loss_fn
                    )
                    if phase == 1:
                        msg += f"  test_sign={test_sign_acc*100:.1f}% ({test_n_err}/{test_n_tot})  test_c_mae={test_c_mae:.2f}V"
                    elif phase == 2:
                        msg += f"  test_c_mae={test_c_mae:.2f}V  test_sign={test_sign_acc*100:.1f}% ({test_n_err}/{test_n_tot})"
                    else:
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

            # early stopping
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
                        optimizer = optim.AdamW(
                            [p for p in model.parameters() if p.requires_grad],
                            lr=new_lr, weight_decay=weight_decay
                        )
                else:
                    print(f"  → Epoch {epoch}: 早停（最佳 val_loss={best_val_loss:.2f}）")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
        else:
            if epoch % 5 == 0 or epoch == 1:
                print(f"  [Phase{phase}] Epoch {epoch}  loss={avg_train_loss:.2f}")

    if has_val and best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    return {"model": model, **norm_params}


def predict(model_dict: dict, record_or_X):
    """预测电压"""
    model = model_dict["model"]

    n_env_pred = model.n_env if hasattr(model, 'n_env') else 6
    wave_len_pred = model.wave_len if hasattr(model, 'wave_len') else 512

    if isinstance(record_or_X, dict):
        rec = record_or_X
        wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:wave_len_pred]

        def _f(v, d=0.0):
            try: return float(v)
            except: return d
        temp = _f(rec.get("RTU_REGS_P00_ENV_TEMP"))
        humid = _f(rec.get("RTU_REGS_P00_ENV_HUMIDITY"))
        rpm = _f(rec.get("RTU_REGS_P00_ROTOR_RPM"))

        ac = wave - np.mean(wave)
        n = len(ac)
        fft_mag = np.abs(fft(ac))[:n // 2]

        n_fft_pred = model.n_fft if hasattr(model, 'n_fft') else 10
        harmonics = np.zeros(n_fft_pred)
        for i in range(n_fft_pred):
            idx = i + 1
            if idx < len(fft_mag):
                harmonics[i] = 2.0 * fft_mag[idx] / n

        wm = np.mean(wave)
        ws = np.std(wave) + 1e-8
        wave_norm = ((wave - wm) / ws).reshape(1, -1)
        fft_norm = ((harmonics - model_dict["fft_mean"]) / model_dict["fft_std"]).reshape(1, -1)

        # 从 record 中读取预提取的时域特征
        vpp = _f(rec.get("vpp"))
        kurt = _f(rec.get("kurtosis"))
        skewness = _f(rec.get("skewness"))
        aux = np.array([temp, humid, rpm, vpp, kurt, skewness])
        env_norm = ((aux - model_dict["env_mean"]) / model_dict["env_std"]).reshape(1, -1)

        wave_t = torch.tensor(wave_norm, dtype=torch.float32).to(_DEVICE)
        fft_t = torch.tensor(fft_norm, dtype=torch.float32).to(_DEVICE)
        env_t = torch.tensor(env_norm, dtype=torch.float32).to(_DEVICE)
    else:
        wave_t = torch.tensor(record_or_X[0], dtype=torch.float32).to(_DEVICE)
        fft_t = torch.tensor(record_or_X[1], dtype=torch.float32).to(_DEVICE)
        env_t = torch.tensor(record_or_X[2], dtype=torch.float32).to(_DEVICE)

    with torch.no_grad():
        _, _, _, _, final = model(wave_t, fft_t, env_t)
        return final.cpu().numpy()
