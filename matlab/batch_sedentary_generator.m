% =========================================================================
% Batch Signal Generator for Sedentary States Dataset (Production Version - V3 Corrected)
% Author: Gemini
% Date: 2025-10-13
% Description: This script generates the final, production-level dataset
%              with a robust check to prevent floating-point index errors.
% =========================================================================
clear; clc; close all;

%% --- 1. 配置 (Configuration) ---
% -------------------------------------------------------------------------
OUTPUT_DIR = 'sedentary_dataset';
STATES = {'relaxed', 'flow', 'stress', 'drowsiness'};
SIMULATION_DURATION = 180;
SAMPLES_PER_STATE = 300;
RADAR_FS = 100;
HR_RANGE = [60, 80];

%% --- 2. 数据集生成主循环 (Main Generation Loop) ---
% -------------------------------------------------------------------------
fprintf('开始生成最终的“实战级”久坐状态数据集...\n');
fprintf('仿真时长: %d秒/样本, 每个状态 %d 个样本\n', SIMULATION_DURATION, SAMPLES_PER_STATE);
if ~exist(OUTPUT_DIR, 'dir'), mkdir(OUTPUT_DIR); end
total_files_generated = 0;
for state_idx = 1:length(STATES)
    current_state = STATES{state_idx};
    state_dir = fullfile(OUTPUT_DIR, current_state);
    if ~exist(state_dir, 'dir'), mkdir(state_dir); end
    fprintf('\n--- 正在生成状态: %s ---\n', current_state);
    for sample_num = 1:SAMPLES_PER_STATE
        mean_hr = randi(HR_RANGE);
        heartbeats = simulate_rr_intervals_sedentary(SIMULATION_DURATION, current_state, mean_hr);
        [time_axis, radar_signal] = generate_chest_displacement_sedentary(heartbeats, SIMULATION_DURATION, RADAR_FS, current_state);
        label = current_state;
        rr_intervals_ms = diff(heartbeats) * 1000;
        filename = sprintf('%s_hr%d_sample%03d.mat', current_state, mean_hr, sample_num);
        filepath = fullfile(state_dir, filename);
        save(filepath, 'radar_signal', 'time_axis', 'label', 'rr_intervals_ms', 'heartbeats');
        total_files_generated = total_files_generated + 1;
        if mod(sample_num, 50) == 0, fprintf('已生成 %d / %d 个样本\n', sample_num, SAMPLES_PER_STATE); end
    end
end
fprintf('\n数据集生成完毕！总共生成了 %d 个文件。\n', total_files_generated);

%% ========== 函数定义 ==========
function heartbeats_time = simulate_rr_intervals_sedentary(duration, state, mean_hr)
    if strcmp(state, 'relaxed'), rmssd_ms = 70 + randn()*10; lf_hf_ratio = 0.6 + randn()*0.2;
    elseif strcmp(state, 'flow'), rmssd_ms = 45 + randn()*8; lf_hf_ratio = 2.0 + randn()*0.4;
    elseif strcmp(state, 'stress'), rmssd_ms = 20 + randn()*5; lf_hf_ratio = 4.5 + randn()*0.5;
    elseif strcmp(state, 'drowsiness'), rmssd_ms = 30 + randn()*5; lf_hf_ratio = 3.5 + randn()*0.5;
    end
    internal_fs = 100;
    time_mod = 0:(1/internal_fs):(duration - 1/internal_fs);
    hf_amplitude = rmssd_ms / sqrt(2); hf_freq = 0.25;
    hf_component = hf_amplitude * sin(2 * pi * hf_freq * time_mod);
    lf_amplitude = hf_amplitude * sqrt(lf_hf_ratio); lf_freq = 0.1;
    lf_component = lf_amplitude * sin(2 * pi * lf_freq * time_mod);
    mean_rr = 60000 / mean_hr;
    noise = 10 * randn(1, length(time_mod));
    rr_modulation = mean_rr + hf_component + lf_component + noise;
    heartbeats_time = [0.0];
    while heartbeats_time(end) < duration
        current_time = heartbeats_time(end);
        current_rr_idx = round(current_time * internal_fs) + 1;
        
        % 【核心修正】增加一个更稳健的、防御性的安全检查，防止任何形式的索引越界
        if current_rr_idx < 1 || current_rr_idx > length(rr_modulation)
            break; % 如果索引无效 (小于1或大于数组长度)，则立即安全退出循环
        end
        
        current_rr = rr_modulation(current_rr_idx);
        next_beat = current_time + max(current_rr, 200) / 1000;
        if next_beat >= duration, break; end
        heartbeats_time = [heartbeats_time, next_beat];
    end
end
function [time_axis, final_signal] = generate_chest_displacement_sedentary(heartbeats_time, duration, fs, state)
    n_samples = duration * fs;
    time_axis = linspace(0, duration, n_samples);
    if strcmp(state, 'relaxed') || strcmp(state, 'drowsiness'), breathing_freq = 0.2; breathing_amplitude = 1.2;
    elseif strcmp(state, 'stress'), breathing_freq = 0.33; breathing_amplitude = 0.7;
    else, breathing_freq = 0.25; breathing_amplitude = 1.0;
    end
    breathing_signal = breathing_amplitude * sin(2 * pi * breathing_freq * time_axis);
    heartbeat_signal = zeros(1, n_samples);
    heartbeat_amplitude = 0.2;
    for i = 1:length(heartbeats_time)
        t = heartbeats_time(i); idx = round(t * fs) + 1;
        if idx <= n_samples
            pulse_width = round(0.1 * fs); window = gausswin(pulse_width)';
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