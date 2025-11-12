% =========================================================================
% High-Fidelity Physiological Signal Simulator (MATLAB Version)
% Author: Gemini (based on user's Python version)
% Date: 2025-10-13
% =========================================================================
clear; clc; close all;

%% --- 参数设置 ---
SIMULATION_DURATION = 60;  % 仿真时长 (秒)
RADAR_FS = 100;            % 模拟雷达的采样率 (Hz)
MEAN_HR = 75;              % 平均心率 (BPM)

% --- 切换这里来观察不同状态 ---
% PHYSIOLOGICAL_STATE = 'normal';
PHYSIOLOGICAL_STATE = 'stress';
% PHYSIOLOGICAL_STATE = 'deep_sleep';


%% --- 生成信号 ---
fprintf('正在生成状态为 ''%s'' 的信号...\n', PHYSIOLOGICAL_STATE);

% 1. 生成心跳时间点序列
heartbeats = simulate_rr_intervals_v2( ...
    SIMULATION_DURATION, ...
    PHYSIOLOGICAL_STATE, ...
    MEAN_HR ...
);

% 2. 生成模拟的雷达胸腔位移信号
[time, radar_signal] = generate_chest_displacement_signal( ...
    heartbeats, ...
    SIMULATION_DURATION, ...
    RADAR_FS, ...
    PHYSIOLOGICAL_STATE ...
);

fprintf('信号生成完毕。\n');


%% --- 可视化 ---
figure('Position', [100, 100, 1200, 800]); % 创建一个大一点的图形窗口
sgtitle(sprintf('Simulated Physiological Signal (State: %s) - MATLAB Version', PHYSIOLOGICAL_STATE), 'FontSize', 16);

% --- 图1: RR间期序列 (展示HRV) ---
subplot(3, 1, 1);
rr_intervals_ms = diff(heartbeats) * 1000;
if length(rr_intervals_ms) > 1
    plot(heartbeats(2:end), rr_intervals_ms, 'o-', 'MarkerSize', 4, 'LineWidth', 1.5, 'DisplayName', 'RR Intervals');
    ylim([mean(rr_intervals_ms) - 80, mean(rr_intervals_ms) + 80]);
end
title('Heart Rate Variability (RR Interval Tachogram)');
ylabel('RR Interval (ms)');
grid on;
legend;
xlim([0, SIMULATION_DURATION]);

% --- 图2: 生成的模拟雷达信号 ---
subplot(3, 1, 2);
plot(time, radar_signal, 'LineWidth', 1.5, 'DisplayName', 'Simulated Chest Displacement');
title('Simulated Radar Signal (Chest Displacement)');
ylabel('Displacement (a.u.)');
xlabel('Time (s)');
grid on;
legend;
xlim([0, SIMULATION_DURATION]);

% --- 图3: 信号的频谱图 ---
subplot(3, 1, 3);
if length(rr_intervals_ms) > 10
    fs_tachogram = 1 / (mean(rr_intervals_ms) / 1000);
    
    % 【核心修正】: 明确指定窗函数和窗长
    % 使用'hann'窗，窗长设为整个信号的长度，以获得最高的分辨率
    % 这与Python版本中使用尽可能长的nperseg的行为是一致的
    window_length = length(rr_intervals_ms);
    [Pxx, f] = pwelch(rr_intervals_ms - mean(rr_intervals_ms), hann(window_length), [], window_length, fs_tachogram);
    
    semilogy(f, Pxx, 'LineWidth', 1.5, 'DisplayName', 'RR Power Spectrum'); % 使用半对数坐标，并修正图例名称
    hold on;
    
    % 绘制LF和HF频段的背景区域
    yl = get(gca, 'YLim');
    patch([0.04 0.15 0.15 0.04], [yl(1) yl(1) yl(2) yl(2)], 'cyan', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'LF band (0.04-0.15 Hz)');
    patch([0.15 0.4 0.4 0.15], [yl(1) yl(1) yl(2) yl(2)], 'green', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'HF band (0.15-0.4 Hz)');
    
    % 重新绘制信号线，确保它在最上层
    semilogy(f, Pxx, 'LineWidth', 1.5, 'HandleVisibility','off'); 

    hold off;
    title('Power Spectral Density of RR Intervals (Corrected)');
    xlabel('Frequency (Hz)');
    ylabel('Power (ms^2/Hz)');
    xlim([0, 0.5]);
    grid on;
    legend;
end


%% ========== 函数定义 ==========

% --------------------------------------------------------------------------
% 核心任务一 (V2.0 - 稳健版): 根据状态生成心跳节律 (RR Intervals)
% --------------------------------------------------------------------------
function heartbeats_time = simulate_rr_intervals_v2(duration, state, mean_hr)
    % 1. 定义不同状态的HRV参数
    if strcmp(state, 'stress')
        rmssd_ms = 25;
        lf_hf_ratio = 4.0;
    elseif strcmp(state, 'deep_sleep')
        rmssd_ms = 80;
        lf_hf_ratio = 0.5;
    else % 'normal'
        rmssd_ms = 50;
        lf_hf_ratio = 1.5;
    end

    % 2. 准备调制信号
    internal_fs = 100;
    time_mod = 0:(1/internal_fs):(duration - 1/internal_fs);
    
    hf_amplitude = rmssd_ms / sqrt(2);
    hf_freq = 0.25;
    hf_component = hf_amplitude * sin(2 * pi * hf_freq * time_mod);
    
    lf_amplitude = hf_amplitude * sqrt(lf_hf_ratio);
    lf_freq = 0.1;
    lf_component = lf_amplitude * sin(2 * pi * lf_freq * time_mod);
    
    mean_rr = 60000 / mean_hr;
    noise = 10 * randn(1, length(time_mod));
    
    rr_modulation = mean_rr + hf_component + lf_component + noise;

    % 3. 迭代生成心跳事件
    heartbeats_time = [0.0];
    while heartbeats_time(end) < duration
        current_time = heartbeats_time(end);
        
        current_rr_idx = round(current_time * internal_fs) + 1; % MATLAB索引从1开始
        if current_rr_idx > length(rr_modulation)
            break;
        end
        current_rr = rr_modulation(current_rr_idx);
        
        next_beat = current_time + max(current_rr, 200) / 1000; % 转换为秒
        
        if next_beat >= duration
            break;
        end
        
        heartbeats_time = [heartbeats_time, next_beat];
    end
end

% --------------------------------------------------------------------------
% 核心任务二：生成连续的生理信号
% --------------------------------------------------------------------------
function [time_axis, final_signal] = generate_chest_displacement_signal(heartbeats_time, duration, fs, state)
    n_samples = duration * fs;
    time_axis = linspace(0, duration, n_samples);
    
    if strcmp(state, 'stress')
        breathing_freq = 0.3;
        breathing_amplitude = 0.8;
    else
        breathing_freq = 0.25;
        breathing_amplitude = 1.0;
    end
    
    breathing_signal = breathing_amplitude * sin(2 * pi * breathing_freq * time_axis);
    
    heartbeat_signal = zeros(1, n_samples);
    heartbeat_amplitude = 0.2;
    
    for i = 1:length(heartbeats_time)
        t = heartbeats_time(i);
        idx = round(t * fs) + 1; % MATLAB索引从1开始
        
        if idx <= n_samples
            pulse_width = round(0.1 * fs);
            window = gausswin(pulse_width)'; % 使用gausswin并转置为行向量
            
            start_idx = max(1, idx - floor(pulse_width/2));
            end_idx = min(n_samples, start_idx + pulse_width - 1);
            
            actual_len = end_idx - start_idx + 1;
            
            heartbeat_signal(start_idx:end_idx) = heartbeat_signal(start_idx:end_idx) + heartbeat_amplitude * window(1:actual_len);
        end
    end
    
    combined_signal = breathing_signal + heartbeat_signal;
    
    baseline_wander = 0.1 * sin(2 * pi * 0.05 * time_axis);
    white_noise = 0.05 * randn(1, n_samples);
    
    final_signal = combined_signal + baseline_wander + white_noise;
end