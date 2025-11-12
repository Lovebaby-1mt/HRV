% =========================================================================
% Signal Processing Pipeline (IIR-Filter with Post-Processing Gold Standard)
% Author: Gemini
% Date: 2025-10-13
% Description: This Gold Standard version combines the stable IIR filter for
%              signal separation with the crucial RR interval post-processing
%              step, creating a robust and reliable feature extractor.
% =========================================================================
clear; clc; close all;

%% --- 1. 配置 (Configuration) ---
% -------------------------------------------------------------------------
DATASET_DIR = 'sedentary_dataset';
STATE = 'stress'; 
FILENAME = 'stress_hr63_sample040.mat'; 

filePath = fullfile(DATASET_DIR, STATE, FILENAME);
% -------------------------------------------------------------------------

%% --- 2. 加载数据 (Load Data) ---
% -------------------------------------------------------------------------
if ~exist(filePath, 'file')
    error('文件不存在: %s', filePath);
end
fprintf('正在加载并处理文件: %s\n', filePath);
data = load(filePath);

radar_signal = data.radar_signal;
time_axis = data.time_axis;
fs = 1 / (time_axis(2) - time_axis(1));

true_label = data.label;
true_heartbeat_times = data.heartbeats;
true_rr_intervals_ms = data.rr_intervals_ms;


%% --- 3. 信号分离 (Signal Separation using IIR Filters) ---
% -------------------------------------------------------------------------
fprintf('正在使用IIR滤波器进行信号分离...\n');

lpFilt = designfilt('lowpassiir', 'FilterOrder', 4, 'HalfPowerFrequency', 0.8, 'SampleRate', fs);
breathing_signal = filtfilt(lpFilt, radar_signal);

bpFilt = designfilt('bandpassiir', 'FilterOrder', 4, 'HalfPowerFrequency1', 0.9, 'HalfPowerFrequency2', 2.5, 'SampleRate', fs);
heartbeat_signal = filtfilt(bpFilt, radar_signal);


%% --- 4. 特征提取 (Feature Extraction - Basic but Robust) ---
% -------------------------------------------------------------------------
% 呼吸率提取
[pks_breath, locs_breath] = findpeaks(breathing_signal, 'MinPeakDistance', fs*2);
if length(locs_breath) > 1, breathing_rate_rpm = 60 / (mean(diff(locs_breath)) / fs); else, breathing_rate_rpm = NaN; end

% RR间期提取 (基础峰值检测)
heartbeat_signal_squared = heartbeat_signal .^ 2;
% 使用 'MinPeakProminence' 依然是比 'MinPeakHeight' 更稳健的选择
prominence_threshold = 1.5 * std(heartbeat_signal_squared);
[~, locs_heart] = findpeaks(heartbeat_signal_squared, 'MinPeakProminence', prominence_threshold, 'MinPeakDistance', fs*0.5);
pks_heart = heartbeat_signal(locs_heart); 
extracted_heartbeat_times = locs_heart / fs;
rr_intervals_raw = diff(extracted_heartbeat_times) * 1000; % 得到原始RR序列


%% --- 5. RR间期后处理 (Outlier Correction) - 【核心模块】---
% -------------------------------------------------------------------------
fprintf('正在对RR间期序列进行后处理...\n');
if ~isempty(rr_intervals_raw)
    physio_min_ms = 60000 / 180; 
    physio_max_ms = 60000 / 40;
    outlier_indices = find(rr_intervals_raw < physio_min_ms | rr_intervals_raw > physio_max_ms);
    rr_intervals_cleaned = rr_intervals_raw;
    for i = 1:length(outlier_indices)
        idx = outlier_indices(i);
        local_start = max(1, idx - 2);
        local_end = min(length(rr_intervals_cleaned), idx + 2);
        local_indices = setdiff(local_start:local_end, outlier_indices);
        if ~isempty(local_indices)
            rr_intervals_cleaned(idx) = mean(rr_intervals_cleaned(local_indices));
        else
            global_good_indices = setdiff(1:length(rr_intervals_cleaned), outlier_indices);
            if ~isempty(global_good_indices)
                rr_intervals_cleaned(idx) = mean(rr_intervals_cleaned(global_good_indices));
            end
        end
    end
else
    rr_intervals_cleaned = [];
end


%% --- 6. 性能验证与HRV分析 (基于清理后的数据) ---
% -------------------------------------------------------------------------
if ~isempty(rr_intervals_cleaned) && ~isempty(true_rr_intervals_ms)
    compare_len = min(length(true_rr_intervals_ms), length(rr_intervals_cleaned));
    rr_mae_ms = mean(abs(true_rr_intervals_ms(1:compare_len) - rr_intervals_cleaned(1:compare_len)));
else
    rr_mae_ms = Inf;
end
if length(rr_intervals_cleaned) > 10
    rmssd = sqrt(mean(diff(rr_intervals_cleaned).^2));
    sdnn = std(rr_intervals_cleaned);
    fs_tachogram = 1 / (mean(rr_intervals_cleaned) / 1000);
    [Pxx, f] = pwelch(rr_intervals_cleaned - mean(rr_intervals_cleaned), hann(length(rr_intervals_cleaned)), [], [], fs_tachogram);
    lf_band = (f >= 0.04 & f < 0.15);
    hf_band = (f >= 0.15 & f <= 0.4);
    lf_power = trapz(f(lf_band), Pxx(lf_band));
    hf_power = trapz(f(hf_band), Pxx(hf_band));
    if hf_power > 0, lf_hf_ratio = lf_power / hf_power; else, lf_hf_ratio = Inf; end
else
    rmssd = NaN; sdnn = NaN; lf_power = NaN; hf_power = NaN; lf_hf_ratio = NaN;
end


%% --- 7. 结果总结与可视化 (Results & Visualization) ---
% -------------------------------------------------------------------------
fprintf('\n--- 分析结果 (IIR + Post-Processing Version) ---\n');
fprintf('真实标签: %s\n', true_label);
fprintf('提取的呼吸率: %.2f 次/分钟\n', breathing_rate_rpm);
if ~isempty(rr_intervals_cleaned), fprintf('提取的平均心率 (清理后): %.2f BPM\n', 60000 / mean(rr_intervals_cleaned)); end
fprintf('\n--- 性能验证 ---\n');
fprintf('RR间期 MAE (清理后 vs. 真实值): %.2f ms\n', rr_mae_ms);
fprintf('心跳检测率: %.1f%% (%d/%d)\n', length(locs_heart)/length(true_heartbeat_times)*100, length(locs_heart), length(true_heartbeat_times));
fprintf('\n--- 提取的HRV指标 (清理后) ---\n');
fprintf('RMSSD: %.2f ms\n', rmssd);
fprintf('SDNN: %.2f ms\n', sdnn);
fprintf('LF Power: %.2f ms^2\n', lf_power);
fprintf('HF Power: %.2f ms^2\n', hf_power);
fprintf('LF/HF Ratio: %.2f\n', lf_hf_ratio);
fprintf('----------------\n');

% --- 绘图 ---
figure('Position', [50, 50, 1400, 900]);
sgtitle(sprintf('信号处理流程 (IIR + Post-Processing): %s', FILENAME), 'FontSize', 16, 'Interpreter', 'none');
% 左侧图: 信号分离
subplot(3, 2, 1); plot(time_axis, radar_signal); title('1. 原始雷达信号'); grid on; xlabel('Time (s)'); ylabel('Displacement (a.u.)');
subplot(3, 2, 3); plot(time_axis, breathing_signal, 'g'); hold on; plot(locs_breath/fs, pks_breath, 'ro'); title('2. 分离的呼吸信号'); grid on; xlabel('Time (s)'); ylabel('Amplitude');
subplot(3, 2, 5); plot(time_axis, heartbeat_signal, 'r'); hold on; plot(locs_heart/fs, pks_heart, 'bo'); title('3. 分离的心跳信号与峰值检测'); grid on; xlabel('Time (s)'); ylabel('Amplitude');

% 右侧图: RR间期分析与HRV
% 图4: 原始 vs. 清理后的RR间期对比
subplot(3, 2, 2);
if ~isempty(rr_intervals_raw)
    plot(extracted_heartbeat_times(2:end), rr_intervals_raw, 'o-', 'DisplayName', '原始提取的RR');
    hold on;
    plot(extracted_heartbeat_times(2:end), rr_intervals_cleaned, 's-', 'Color', 'm', 'LineWidth', 2, 'DisplayName', '清理后的RR');
    title('4. RR间期后处理: 原始 vs. 清理后'); xlabel('Time (s)'); ylabel('RR Interval (ms)'); grid on; legend;
end

% 图5: 最终结果 vs. 真实值
subplot(3, 2, 4);
if ~isempty(rr_intervals_cleaned)
    plot(extracted_heartbeat_times(2:end), rr_intervals_cleaned, 'o-', 'DisplayName', '最终RR序列');
    hold on;
    plot(true_heartbeat_times(2:end), true_rr_intervals_ms, 's--', 'DisplayName', '真实的RR (金标准)');
    title('5. 最终结果 vs. 真实值'); xlabel('Time (s)'); ylabel('RR Interval (ms)'); grid on; legend;
end

% 图6: HRV功率谱 (基于清理后的数据)
subplot(3, 2, 6);
if length(rr_intervals_cleaned) > 10
    semilogy(f, Pxx, 'LineWidth', 1.5, 'DisplayName', 'HRV Power');
    hold on;
    yl = get(gca, 'YLim');
    patch([0.04 0.15 0.15 0.04], [yl(1) yl(1) yl(2) yl(2)], 'cyan', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'LF band');
    patch([0.15 0.4 0.4 0.15], [yl(1) yl(1) yl(2) yl(2)], 'green', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'HF band');
    semilogy(f, Pxx, 'LineWidth', 1.5, 'HandleVisibility','off');
    hold off;
    title('6. 清理后的HRV功率谱密度'); xlabel('Frequency (Hz)'); ylabel('Power (ms^2/Hz)'); xlim([0, 0.5]); grid on; legend('Location', 'northeast');
end