import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_csv_data(file_path):
    """
    加载CSV文件中的512点采样数据
    :param file_path: CSV文件路径
    :return: 采样数据数组（numpy.ndarray）
    """
    # 读取CSV文件，处理不同格式的情况
    try:
        # 尝试以无表头方式读取
        df = pd.read_csv(file_path, header=None)
        # 提取所有数值数据并展平为一维数组
        data = df.values.flatten()
        # 过滤非数值数据和空值
        data = [x for x in data if isinstance(x, (int, float)) and not np.isnan(x)]
        
        # 确保数据长度为512点
        if len(data) >= 512:
            data = data[:512]
        else:
            raise ValueError(f"数据长度不足512点，实际长度：{len(data)}")
            
        return np.array(data, dtype=np.float64)
        
    except Exception as e:
        # 尝试其他读取方式
        try:
            with open(file_path, 'r') as f:
                content = f.read().replace('\n', ',').split(',')
                data = []
                for item in content:
                    item = item.strip()
                    if item:
                        try:
                            data.append(float(item))
                        except:
                            continue
            if len(data) >= 512:
                return np.array(data[:512], dtype=np.float64)
            else:
                raise ValueError(f"数据长度不足512点，实际长度：{len(data)}")
        except Exception as e2:
            raise Exception(f"数据加载失败：{str(e2)}")

def sine_wave_model(t, amplitude, frequency, phase, dc_offset):
    """
    正弦波数学模型：y = A*sin(2πft + φ) + B
    :param t: 时间数组
    :param amplitude: 振幅 (A)
    :param frequency: 频率 (f)
    :param phase: 初始相位 (φ，单位：rad)
    :param dc_offset: 直流偏移 (B)
    :return: 正弦波计算值
    """
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + dc_offset

def calculate_sine_confidence(fft_magnitude, dominant_freq_index, fft_size):
    """
    计算正弦波置信度（主频率能量占总能量的比例）
    :param fft_magnitude: FFT幅度谱
    :param dominant_freq_index: 主频率索引
    :param fft_size: FFT点数
    :return: 置信度（0-1）
    """
    # 只考虑正频率部分
    half_size = fft_size // 2
    fft_pos = fft_magnitude[:half_size]
    
    # 计算总能量和主频率能量（能量与幅度平方成正比）
    total_energy = np.sum(np.square(fft_pos[1:]))  # 排除直流分量
    dominant_energy = np.square(fft_pos[dominant_freq_index])
    
    # 计算置信度，确保在0-1范围内
    confidence = dominant_energy / total_energy if total_energy > 1e-9 else 0.0
    return np.clip(confidence, 0.0, 1.0)

def analyze_sine_wave(signal_data, sample_freq=1000.0):
    """
    分析信号是否为正弦波并计算关键参数
    :param signal_data: 采样数据数组（1024点）
    :param sample_freq: 采样频率（Hz），默认1000Hz
    :return: 分析结果字典
    """
    # 基本参数计算
    n_points = len(signal_data)
    time = np.arange(n_points) / sample_freq  # 时间数组
    dt = 1.0 / sample_freq  # 时间步长
    
    # 1. 计算RMS值（有效值）
    rms_value = np.sqrt(np.mean(np.square(signal_data)))
    
    # 2. FFT分析
    # 去除直流分量
    signal_dc_removed = signal_data - np.mean(signal_data)
    # 执行FFT
    fft_result = fft(signal_dc_removed)
    # 计算频率轴
    freq_axis = np.fft.fftfreq(n_points, d=dt)
    # 计算幅度谱
    fft_magnitude = np.abs(fft_result)
    
    # 3. 找到主频率
    # 只考虑正频率
    positive_mask = freq_axis >= 0
    freq_pos = freq_axis[positive_mask]
    fft_pos_mag = fft_magnitude[positive_mask]
    
    # 排除直流分量（0Hz），找到幅度最大的频率
    dominant_freq_index = np.argmax(fft_pos_mag[1:]) + 1  # +1是因为跳过了索引0（直流）
    dominant_freq = freq_pos[dominant_freq_index]
    
    # 4. 计算正弦波置信度
    confidence = calculate_sine_confidence(fft_magnitude, dominant_freq_index, n_points)
    
    # 5. 正弦波拟合以获取更精确的相位和振幅
    # 初始参数猜测
    dc_offset_guess = np.mean(signal_data)
    amplitude_guess = (np.max(signal_data) - np.min(signal_data)) / 2.0
    phase_guess = 0.0  # 初始相位猜测为0
    
    initial_guess = [amplitude_guess, dominant_freq, phase_guess, dc_offset_guess]
    print(f"初始拟合参数猜测: 振幅={amplitude_guess:.4f}, 频率={dominant_freq:.4f} Hz, 相位={phase_guess:.4f} rad, 直流偏移={dc_offset_guess:.4f}")
    # 拟合正弦波模型
    try:
        popt, _ = curve_fit(
            sine_wave_model, 
            time, 
            signal_data, 
            p0=initial_guess,
            maxfev=10000,
            bounds=(
                [0, 0, -np.pi, -np.inf],  # 下限：振幅≥0，频率≥0，相位≥-π，直流偏移无下限
                [np.inf, sample_freq/2, np.pi, np.inf]  # 上限：频率≤奈奎斯特频率
            )
        )
        fit_amplitude, fit_freq, fit_phase, fit_dc = popt
        
        # 调整相位到0-2π范围
        fit_phase = fit_phase % (2 * np.pi)
        
        # 生成拟合曲线
        fitted_signal = sine_wave_model(time, fit_amplitude, fit_freq, fit_phase, fit_dc)
        
    except Exception as e:
        print(f"正弦波拟合警告：{str(e)}，使用FFT结果估算相位")
        # 如果拟合失败，使用FFT结果估算相位
        fit_freq = dominant_freq
        fit_amplitude = fft_pos_mag[dominant_freq_index] / (n_points / 2)  # 幅度归一化
        fit_phase = np.angle(fft_result[dominant_freq_index]) % (2 * np.pi)  # 相位调整到0-2π
        fit_dc = np.mean(signal_data)
        fitted_signal = sine_wave_model(time, fit_amplitude, fit_freq, fit_phase, fit_dc)
    
    # 6. 计算拟合误差（用于辅助判断）
    fit_error = np.mean(np.abs(signal_data - fitted_signal))
    
    # 返回分析结果
    return {
        'confidence': confidence,          # 正弦波置信度（0-1）
        'frequency': fit_freq,             # 频率（Hz）
        'rms_value': rms_value,            # RMS值（有效值）
        'phase': fit_phase,                # 初始相位（rad）
        'phase_deg': np.degrees(fit_phase),# 初始相位（度）
        'amplitude': fit_amplitude,        # 振幅
        'dc_offset': fit_dc,               # 直流偏移
        'time': time,                      # 时间数组
        'original_signal': signal_data,    # 原始信号
        'fitted_signal': fitted_signal,    # 拟合信号
        'freq_axis': freq_pos,             # 频率轴（正频率）
        'fft_magnitude': fft_pos_mag       # FFT幅度谱（正频率）
    }

def plot_analysis_results(analysis_result, save_path='sine_wave_analysis.png'):
    """
    绘制信号分析结果图表
    :param analysis_result: 分析结果字典
    :param save_path: 图表保存路径
    """
    # 创建2x2的子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('正弦波信号分析结果', fontsize=16, fontweight='bold')
    
    # 1. 原始信号与拟合信号对比
    ax1.plot(analysis_result['time'], analysis_result['original_signal'], 
             'b-', linewidth=1.5, label='原始信号', alpha=0.8)
    ax1.plot(analysis_result['time'], analysis_result['fitted_signal'], 
             'r--', linewidth=2, label='正弦波拟合', alpha=0.9)
    ax1.set_title('原始信号与正弦波拟合对比', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('信号幅值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 信号局部放大图（前100个点）
    n_zoom = min(100, len(analysis_result['time']))
    ax2.plot(analysis_result['time'][:n_zoom], analysis_result['original_signal'][:n_zoom], 
             'b-', linewidth=2, label='原始信号', alpha=0.8)
    ax2.plot(analysis_result['time'][:n_zoom], analysis_result['fitted_signal'][:n_zoom], 
             'r--', linewidth=2.5, label='正弦波拟合', alpha=0.9)
    ax2.set_title(f'信号局部放大（前{n_zoom}个采样点）', fontsize=12, fontweight='bold')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('信号幅值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. FFT频谱图
    # 找到主频率位置用于标注
    dominant_freq = analysis_result['frequency']
    print(f"主频率: {dominant_freq:.4f} Hz")
    dominant_freq_idx = np.argmin(np.abs(analysis_result['freq_axis'] - dominant_freq))
    ax3.plot(analysis_result['freq_axis'], analysis_result['fft_magnitude'], 
             'g-', linewidth=1.5, label='FFT幅度谱')
    ax3.scatter(dominant_freq, analysis_result['fft_magnitude'][dominant_freq_idx], 
                color='red', s=80, zorder=5, label=f'主频率: {dominant_freq:.2f} Hz')
    ax3.set_title('信号FFT频谱分析', fontsize=12, fontweight='bold')
    ax3.set_xlabel('频率 (Hz)')
    ax3.set_ylabel('FFT幅度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # 只显示主要频率范围（0到采样频率的1/5）
    ax3.set_xlim(0, analysis_result['freq_axis'].max() / 5)
    
    # 4. 分析结果文本显示
    ax4.axis('off')  # 关闭坐标轴
    # 准备结果文本
    result_text = f"""
    正弦波信号分析结果
    
    1. 正弦波置信度: {analysis_result['confidence']:.4f}
       （{analysis_result['confidence']*100:.1f}%，≥0.5判定为正弦波）
    
    2. 信号频率: {analysis_result['frequency']:.4f} Hz
    
    3. 有效值 (RMS): {analysis_result['rms_value']:.6f}
    
    4. 初始相位: {analysis_result['phase']:.4f} rad
       （{analysis_result['phase_deg']:.2f}°）
    
    5. 振幅: {analysis_result['amplitude']:.6f}
    
    6. 直流偏移: {analysis_result['dc_offset']:.6f}
    
    7. 采样参数:
       - 采样点数: {len(analysis_result['original_signal'])}
       - 采样频率: {1/(analysis_result['time'][1]-analysis_result['time'][0]):.1f} Hz
       - 总采样时间: {analysis_result['time'][-1]:.3f} s
    """
    # 显示文本
    ax4.text(0.1, 0.9, result_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 调整子图间距
    plt.tight_layout()
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"分析图表已保存至: {save_path}")

def main():
    """
    主函数：加载数据、分析信号、绘制结果
    """
    # 1. 配置参数
    csv_file_path = 'raw.csv'  # CSV文件路径
    sample_frequency = 15873       # 采样频率（Hz），可根据实际情况修改
    
    try:
        # 2. 加载CSV数据
        print(f"正在加载CSV文件: {csv_file_path}")
        signal_data = load_csv_data(csv_file_path)
        print(f"成功加载 {len(signal_data)} 点采样数据")
        
        # 3. 分析正弦波信号
        print(f"\n正在分析信号（采样频率: {sample_frequency} Hz）...")
        analysis_result = analyze_sine_wave(signal_data, sample_frequency)
        
        # 4. 打印分析结果
        print("\n" + "="*60)
        print("正弦波信号分析结果")
        print("="*60)
        print(f"1. 正弦波置信度: {analysis_result['confidence']:.4f}")
        print(f"   信号类型判定: {'正弦波' if analysis_result['confidence'] >= 0.5 else '非正弦波'}")
        print(f"2. 信号频率:     {analysis_result['frequency']:.4f} Hz")
        print(f"3. 有效值 (RMS): {analysis_result['rms_value']:.6f}")
        print(f"4. 初始相位:     {analysis_result['phase']:.4f} rad ({analysis_result['phase_deg']:.2f}°)")
        print(f"5. 振幅:         {analysis_result['amplitude']:.6f}")
        print(f"6. 直流偏移:     {analysis_result['dc_offset']:.6f}")
        print("="*60)
        
        # 5. 绘制分析结果图表
        print(f"\n正在生成分析图表...")
        plot_analysis_results(analysis_result)
        
        print("\n信号分析完成！")
        
    except Exception as e:
        print(f"\n分析过程出错: {str(e)}")

if __name__ == "__main__":
    main()