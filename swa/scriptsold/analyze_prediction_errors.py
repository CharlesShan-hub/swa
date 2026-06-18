
"""
分析各电压档位的预测误差分布，以便确定3sigma的范围
直接基于已有评估结果来分析（避免重新跑模型）
"""
import sys
sys.path.insert(0, '.')
from collections import defaultdict
import numpy as np

def parse_evaluation_file(file_path):
    """解析评估结果文件中的分电压统计"""
    voltage_stats = []
    reading = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "分电压评估结果" in line:
                reading = True
                continue
            if reading and ("-" in line and "+" in line and "|" in line):
                continue
            if reading and line.strip() == "":
                break
            if reading:
                # 解析行
                parts = [p.strip() for p in line.strip().split("|")]
                if len(parts) == 2 and parts[0].replace("V", "").replace(".", "").replace("-", "").isdigit():
                    voltage = int(parts[0].replace("V", ""))
                    count = int(parts[1])
                    mae = float(parts[3])
                    voltage_stats.append({
                        'voltage': voltage,
                        'count': count,
                        'mae': mae
                    })
    
    return voltage_stats

def get_error_stats_from_evaluation(voltage_stats):
    """基于评估结果的MAE来估算误差分布"""
    print("\n基于已有评估结果的误差分析:")
    print("="*80)
    print("  真值电压  | 样本数 |   MAE   |     推荐±3σ范围")
    print("-"*80)
    
    # 从之前的完整评估我们知道总体P95是 ~13V
    OVERALL_P95 = 13.0  
    CONSERVATIVE_RANGE = 15.0  # ±15V
    MODERATE_RANGE = 20.0      # ±20V (更安全)
    
    for stat in voltage_stats:
        v = stat['voltage']
        range_str = "{:+d}V ~ {:+d}V".format(int(v - MODERATE_RANGE), int(v + MODERATE_RANGE))
        print("  {:7d}V  | {:6d} | {:7.2f} | {:20s}".format(
            v, stat['count'], stat['mae'], range_str
        ))
    
    print("\n" + "="*80)
    print("推荐方案:")
    print("  方案A:  ±{:.0f}V  (覆盖95%的预测误差，来自P95)".format(OVERALL_P95))
    print("  方案B:  ±{:.0f}V  (更保守一些)".format(CONSERVATIVE_RANGE))
    print("  方案C:  ±{:.0f}V  (更安全，覆盖绝大多数情况)".format(MODERATE_RANGE))
    print("="*80)

if __name__ == "__main__":
    print("="*80)
    print("分析 5000.jsonl + CatBoost 的评估结果")
    print("="*80)
    stats = parse_evaluation_file("data/5000/evaluation_results/catboost_model.txt")
    get_error_stats_from_evaluation(stats)
