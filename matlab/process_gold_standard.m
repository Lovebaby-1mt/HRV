function [breathing_signal, heartbeat_signal, rr_intervals_cleaned, Pxx, f] = process_gold_standard(radar_signal, fs)
% =========================================================================
% process_gold_standard (函数版本)
%
% 描述: 
% 这是一个重构后的函数版本。它接受原始雷达信号和采样率，
% 执行信号分离、RR间期后处理，并返回绘制“生理指纹图”(Plan C)
% 所需的中间信号。
%
% 输入:
%   radar_signal - (double vector) 原始雷达信号
%   fs           - (double) 采样频率 (Hz)
%
% 输出:
%   breathing_signal     - (double vector) 分离的呼吸波形
%   heartbeat_signal     - (double vector) 分离的心跳波形
%   rr_intervals_cleaned - (double vector) 后处理后的RR间期 (ms)
%   Pxx                  - (double vector) RR间期序列的功率谱密度
%   f                    - (double vector) Pxx 对应的频率向量
% =========================================================================

%% --- 3. 信号分离 (Signal Separation using IIR Filters) ---
% -------------------------------------------------------------------------
% (来自原始脚本的第 3 节)
lpFilt = designfilt('lowpassiir', 'FilterOrder', 4, 'HalfPowerFrequency', 0.8, 'SampleRate', fs);
breathing_signal = filtfilt(lpFilt, radar_signal);

bpFilt = designfilt('bandpassiir', 'FilterOrder', 4, 'HalfPowerFrequency1', 0.9, 'HalfPowerFrequency2', 2.5, 'SampleRate', fs);
heartbeat_signal = filtfilt(bpFilt, radar_signal);

%% --- 4. 特征提取 (Feature Extraction - Basic but Robust) ---
% -------------------------------------------------------------------------
% (来自原始脚本的第 4 节, 移除了未使用的 breathing_rate_rpm)
heartbeat_signal_squared = heartbeat_signal .^ 2;
prominence_threshold = 1.5 * std(heartbeat_signal_squared);
[~, locs_heart] = findpeaks(heartbeat_signal_squared, 'MinPeakProminence', prominence_threshold, 'MinPeakDistance', fs*0.5);
extracted_heartbeat_times = locs_heart / fs;
rr_intervals_raw = diff(extracted_heartbeat_times) * 1000; % 得到原始RR序列

%% --- 5. RR间期后处理 (Outlier Correction) - 【核心模块】---
% -------------------------------------------------------------------------
% (来自原始脚本的第 5 节)
rr_intervals_cleaned = []; % 首先初始化
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
end

%% --- 6. HRV分析 (基于清理后的数据) ---
% -------------------------------------------------------------------------
% (来自原始脚本的第 6 节, 简化后只为获取 Pxx 和 f)

% 初始化输出
Pxx = [];
f = [];

if length(rr_intervals_cleaned) > 10 % 必须有足够的数据点才能计算PSD
    try
        fs_tachogram = 1 / (mean(rr_intervals_cleaned) / 1000);
        [Pxx, f] = pwelch(rr_intervals_cleaned - mean(rr_intervals_cleaned), hann(length(rr_intervals_cleaned)), [], [], fs_tachogram);
    catch ME
        % 防御性编程：如果 pwelch 失败 (例如 fs_tachogram 为 NaN)
        fprintf('警告: 无法计算 Pwelch。文件: %s. 错误: %s\n', ME.message);
        Pxx = [];
        f = [];
    end
end

end