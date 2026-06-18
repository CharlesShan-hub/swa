"""
深度学习训练脚本：lenet / lenet_hybrid / lenet_bipath

用法：
    uv run python scripts/train_dl.py --algorithm lenet_hybrid
    uv run python scripts/train_dl.py --algorithm lenet_bipath
    uv run python scripts/train_dl.py --algorithm lenet_bipath --phase1-only
"""

import argparse
import json
import sys
import os
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

_SEED = 42
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)

from src.swa.config.settings import config
from scripts.utils.loader import load_jsonl, split_jsonl, get_dataset_model_path


def _print_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    max_err = np.max(np.abs(y_pred - y_true))
    print(f"\n评估结果:")
    print(f"  MAE:  {mae:.4f} V")
    print(f"  RMSE: {rmse:.4f} V")
    print(f"  最大误差: {max_err:.4f} V")


def _save_model(model, algorithm, base, n_fft=None):
    model_path = f"{base}.pth"
    torch.save(model["model"].state_dict(), model_path)
    save_data = {"algorithm": algorithm, "model_path": model_path}
    if n_fft is not None:
        save_data["n_fft"] = n_fft
    for k in ["wave_mean", "wave_std", "fft_mean", "fft_std", "env_mean", "env_std"]:
        if k in model:
            save_data[k] = model[k].tolist()
    with open(f"{base}.json", "w") as f:
        json.dump(save_data, f)
    print(f"\n模型已保存: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="训练深度学习电压估算模型")
    parser.add_argument("--data", default=config.data_source.local_path,
                        help=f"训练数据 JSONL 路径 (默认: {config.data_source.local_path})")
    parser.add_argument("--algorithm",
                        choices=["lenet", "lenet_hybrid", "lenet_bipath"],
                        default=config.estimation.algorithm, help="算法 (默认: settings.py 中的配置)")
    parser.add_argument("--no-full-dataset", action="store_true", dest="full_dataset", default=True,
                        help="不使用全量数据，配合 --limit 使用")
    parser.add_argument("--limit", type=int, default=0,
                        help="非全量时限制条数 (默认: 全部)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="训练集比例 (默认: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="验证集比例 (默认: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="测试集比例 (默认: 0.1)")
    parser.add_argument("--output", default=config.estimation.model_path,
                        help=f"模型参数输出路径 (默认: {config.estimation.model_path})")
    parser.add_argument("--n_fft", "--n-fft", type=int, default=10,
                        help="FFT 谐波数量 (默认: 10)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="训练批次大小 (默认: 256)")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率 (默认: lenet=0.01, lenet_hybrid/lenet_bipath=0.005)")
    parser.add_argument("--symbol-epochs", type=int, default=5,
                        help="Phase 1 纯符号训练轮数，仅对 lenet_bipath 有效 (默认: 5)")
    parser.add_argument("--phase1-epochs", type=int, default=40,
                        help="Phase 1+2 总轮数，仅对 lenet_bipath 有效 (默认: 40)")
    parser.add_argument("--phase1-only", action="store_true",
                        help="仅执行 Phase 1+2，不进入 Phase 3")

    args = parser.parse_args()

    module = importlib.import_module(f"src.swa.estimation.{args.algorithm}")
    print(f"算法: {module.NAME}")

    # 学习率自动缩放
    base_batch = 256
    if args.lr is None:
        if args.algorithm in ["lenet_hybrid", "lenet_bipath"]:
            args.lr = 0.005 * (args.batch_size / base_batch)
        elif args.algorithm == "lenet":
            args.lr = 0.01 * (args.batch_size / base_batch)
    print(f"  batch_size={args.batch_size}, lr={args.lr:.6f}")

    # 加载 & 划分数据
    print(f"加载数据: {args.data}")
    records = load_jsonl(args.data)
    print(f"共 {len(records)} 条，使用{'全量' if args.full_dataset else f'前 {args.limit} 条'}，拆分比例 {args.train_ratio}:{args.val_ratio}:{args.test_ratio}")

    train_records, val_records, test_records = split_jsonl(
        records,
        full_dataset=args.full_dataset,
        limit=args.limit,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=_SEED
    )
    print(f"训练集: {len(train_records)}, 验证集: {len(val_records)}, 测试集: {len(test_records)}")

    # 确定输出路径：如果有对应的数据集目录，就用那个
    output_base = get_dataset_model_path(args.data, args.output)
    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    base, ext = os.path.splitext(output_base)
    print(f"模型将保存到: {base}_*.model")

    # ── 训练 ──
    if args.algorithm == "lenet_bipath":
        model = module.train(train_records, val_records=val_records, test_records=test_records,
                             n_fft=args.n_fft, batch_size=args.batch_size, lr=args.lr,
                             symbol_epochs=args.symbol_epochs,
                             phase1_epochs=args.phase1_epochs, phase1_only=args.phase1_only)
    else:
        model = module.train(train_records, val_records=val_records, test_records=test_records,
                             n_fft=args.n_fft, batch_size=args.batch_size, lr=args.lr)

    # ── 最终测试评估 ──
    (wave_t, fft_t, env_t), y_test, _ = module._build_tensors(test_records, n_fft=args.n_fft)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():
        if args.algorithm == "lenet_hybrid":
            y_pred = model["model"](wave_t, fft_t, env_t).cpu().numpy()
        elif args.algorithm == "lenet":
            y_pred = model["model"](wave_t, fft_t, env_t).cpu().numpy()
        else:
            # lenet_bipath: forward returns (b, c, d, e, final)
            b, c, d, e, final = model["model"](wave_t, fft_t, env_t)
            b = b.cpu()
            c = c.cpu()
            final = final.cpu()
            y_pred = final.numpy()

            sign_pred = torch.sign(b)
            sign_true = torch.sign(y_test_t)
            sign_acc = (sign_pred == sign_true).float().mean().item()
            c_mae = torch.mean(torch.abs(c - torch.abs(y_test_t))).item()
            d_mae = torch.mean(torch.abs(d - torch.abs(y_test_t))).item()

            sign_err_mask = (sign_pred != sign_true).squeeze()
            n_sign_err = sign_err_mask.sum().item()
            print(f"\n符号预测准确率: {sign_acc*100:.1f}% ({n_sign_err}/{len(y_test_t)} 个符号错误)")
            print(f"c 头（绝对值）MAE: {c_mae:.2f} V")
            if not args.phase1_only:
                print(f"d 头（精细绝对值）MAE: {d_mae:.2f} V")

            if n_sign_err > 0:
                print(f"\n符号错误样本（展示前 10 条）:")
                true_sign_np = sign_true.squeeze().numpy()
                pred_sign_np = sign_pred.squeeze().numpy()
                err_idx = np.where(true_sign_np != pred_sign_np)[0][:10]
                print(f"{'True':>8} {'Pred':>8} {'SignTrue':>8} {'SignPred':>8}")
                for i in err_idx:
                    print(f"{y_test[i]:>8.1f} {y_pred[i]:>8.1f} {true_sign_np[i]:>8.0f} {pred_sign_np[i]:>8.0f}")

    _print_metrics(y_test, y_pred)
    _save_model(model, args.algorithm, base, n_fft=args.n_fft)


if __name__ == "__main__":
    main()
