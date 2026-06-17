
"""
简单演示一下基于电压绝对值1/3的容错范围策略
"""

STANDARD_VOLTAGES = [-110, -90, -70, -50, -40, -30, -20, -10, 0, 10, 30, 50, 70, 80, 90, 100, 110]


def get_tolerance(voltage):
    """
    根据电压大小得到容错范围：电压绝对值的 1/3，最小 5V
    """
    if voltage == 0:
        return 5.0  # 0V 特殊处理，容错 5V
    return max(abs(voltage) / 3.0, 5.0)


def classify_voltage(predicted_v):
    """
    使用动态容错范围策略
    """
    best_v = None
    min_dist = float('inf')
    
    for v in STANDARD_VOLTAGES:
        tol = get_tolerance(v)
        dist = abs(predicted_v - v)
        if dist <= tol and dist < min_dist:
            min_dist = dist
            best_v = v
    
    if best_v is None:
        # 没找到，返回最近的
        best_v = min(STANDARD_VOLTAGES, key=lambda x: abs(predicted_v - x))
    
    return best_v


def demo():
    print("="*80)
    print("演示：基于电压绝对值 1/3 的容错范围策略")
    print("="*80)
    
    print("\n各标准电压的容错范围：")
    for v in STANDARD_VOLTAGES:
        tol = get_tolerance(v)
        print("  {:4d}V  容错范围 ±{:4.1f}V  (即 {:5.1f}V 至 {:5.1f}V)".format(
            v, tol, v - tol, v + tol
        ))
    
    print("\n测试一些边界预测值：")
    test_cases = [
        # 30V 边界
        35, 30, 25, 20,
        # 50V 边界
        60, 50, 40,
        # 110V
        120, 110, 100,
        # 负数
        -35, -30, -25,
        # 0V附近
        8, 0, -8,
    ]
    
    print("\n  预测值    分类结果")
    print("-"*40)
    for pred in test_cases:
        cls = classify_voltage(pred)
        print("  {:6.1f}V    {:4d}V".format(pred, cls))


if __name__ == "__main__":
    demo()
