
"""
分析数据集中各个真值电压的分布情况
"""
import json
from collections import defaultdict
import numpy as np

def analyze_data(file_path):
    # 加载数据
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    print(f"总样本数: {len(records)}")
    
    # 按真值电压分组
    voltage_groups = defaultdict(list)
    for rec in records:
        # 解析真值电压
        actual_voltage = rec.get('ACTUAL_VOLTAGE', None)
        if actual_voltage is None:
            continue
            
        # 从波形提取的特征值（或者其他特征）
        signal = rec.get('SIGNAL', [])
        
        # 只保留有完整波形的数据
        if len(signal) == 512:
            voltage_groups[actual_voltage].append(np.array(signal))
    
    print("\n各电压档位样本数:")
    for v in sorted(voltage_groups.keys()):
        print(f"  {v}V: {len(voltage_groups[v])} samples")
    
    # 分析每个电压档位的波形统计
    print("\n各电压档位的波形统计特征:")
    print("  电压档位 | 样本数 |    均值    |   标准差   |   最小值   |   最大值  ")
    print("-" * 70)
    
    all_voltage_info = []
    for v in sorted(voltage_groups.keys()):
        signals = voltage_groups[v]
        # 计算每个样本的波形统计
        means = [np.mean(s) for s in signals]
        stds = [np.std(s) for s in signals]
        
        mean_of_means = np.mean(means)
        std_of_means = np.std(means)
        min_mean = np.min(means)
        max_mean = np.max(means)
        
        all_voltage_info.append({
            'voltage': v,
            'count': len(signals),
            'mean': mean_of_means,
            'std': std_of_means,
            'min': min_mean,
            'max': max_mean
        })
        
        print(f"{v:&gt;8}V | {len(signals):&gt;6} | {mean_of_means:&gt;10.2f} | {std_of_means:&gt;10.2f} | {min_mean:&gt;10.2f} | {max_mean:&gt;10.2f}")
    
    return all_voltage_info

if __name__ == "__main__":
    print("=" * 70)
    print("分析 exported_data.jsonl")
    print("=" * 70)
    info1 = analyze_data("data/exported_data.jsonl")
    
    print("\n" + "=" * 70)
    print("分析 5000.jsonl")
    print("=" * 70)
    info2 = analyze_data("data/5000.jsonl")
