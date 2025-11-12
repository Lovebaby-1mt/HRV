% =========================================================================
% Build Medical Model Prototype (V3 - Robust Validation Version)
% Author: Gemini
% Date: 2025-10-13
% Description: This script uses an expanded dataset and implements 10-fold
%              cross-validation for a robust and reliable assessment of the
%              final model's performance. It also trains a final model on
%              all data for feature importance analysis.
% =========================================================================

clear; clc; close all;

%% --- 1. 批量特征提取 (与之前版本相同) ---
% -------------------------------------------------------------------------
% 假设您已经生成了更大的数据集
fprintf('开始批量处理扩增后的数据集...\n');
DATASET_DIR = 'sedentary_dataset';
STATES = {'relaxed', 'flow', 'stress', 'drowsiness'};
feature_table_advanced = table();

% ... (此部分特征提取循环代码与之前完全相同) ...
for state_idx = 1:length(STATES)
    current_state = STATES{state_idx};
    state_dir = fullfile(DATASET_DIR, current_state);
    if ~exist(state_dir, 'dir'), warning('文件夹不存在: %s', state_dir); continue; end
    files = dir(fullfile(state_dir, '*.mat'));
    fprintf('正在处理状态: %s (%d个文件)\n', current_state, length(files));
    for file_idx = 1:length(files)
        filePath = fullfile(state_dir, files(file_idx).name);
        data = load(filePath);
        features = extract_features_from_signal_advanced(data.radar_signal, 100);
        if ~isempty(features)
            features.Label = categorical({current_state});
            feature_table_advanced = [feature_table_advanced; features];
        end
    end
end
fprintf('\n高级特征提取完成！总共提取了 %d 个有效样本。\n', height(feature_table_advanced));
if height(feature_table_advanced) < 50, error('有效样本过少，无法进行模型训练。'); end

%% --- 2. 【核心升级】使用10折交叉验证进行模型评估 ---
% -------------------------------------------------------------------------
fprintf('\n开始执行10折交叉验证...\n');

% 定义分类器模板。我们在这里定义好随机森林的参数。
t = templateTree('MaxNumSplits', height(feature_table_advanced)); % 允许树充分生长
ensemble_template = fitcensemble(feature_table_advanced, 'Label', 'Method', 'Bag', 'Learners', t, 'NumLearningCycles', 100);

% 使用 crossval 函数自动执行10折交叉验证
% 'KFold', 10 表示使用10折
cv_model = crossval(ensemble_template, 'KFold', 10);

% 计算交叉验证的平均损失率 (错误率)
cv_loss = kfoldLoss(cv_model);

% 计算最终的、稳健的准确率
robust_accuracy = (1 - cv_loss) * 100;

fprintf('----------------------------------------------------------\n');
fprintf(' 10折交叉验证完成！\n');
fprintf(' 模型的稳健平均准确率: %.2f%%\n', robust_accuracy);
fprintf('----------------------------------------------------------\n');
fprintf('这个准确率是对模型在未知数据上表现的更可靠的估计。\n');


%% --- 3. 在全部数据上训练最终模型，并分析特征重要性 ---
% -------------------------------------------------------------------------
fprintf('\n为了分析特征重要性，将在全部数据上训练一个最终模型...\n');

% 使用所有数据进行训练
final_mdl = fitcensemble(feature_table_advanced, 'Label', 'Method', 'Bag', 'NumLearningCycles', 100);

% 【兼容性修正】: 使用通用的 predictorImportance 函数
importance = predictorImportance(final_mdl);
feature_names = final_mdl.PredictorNames;

% 创建条形图来可视化
figure;
bar(importance);
title('最终模型特征重要性评估 (基于全部数据)');
ylabel('重要性得分');
xtickangle(45);
set(gca, 'xtick', 1:length(feature_names), 'xticklabel', feature_names);
grid on;
fprintf('特征重要性分析完成。请查看新生成的图表。\n');


%% ========== 核心特征提取与辅助函数 (保持不变) ==========
% ... (将之前版本的所有函数粘贴在这里) ...
function features = extract_features_from_signal_advanced(radar_signal, fs)
    % ... (代码与V2版本完全相同) ...
    lpFilt = designfilt('lowpassiir', 'FilterOrder', 4, 'HalfPowerFrequency', 0.8, 'SampleRate', fs);
    breathing_signal = filtfilt(lpFilt, radar_signal);
    bpFilt = designfilt('bandpassiir', 'FilterOrder', 4, 'HalfPowerFrequency1', 0.9, 'HalfPowerFrequency2', 2.5, 'SampleRate', fs);
    heartbeat_signal = filtfilt(bpFilt, radar_signal);
    [~, locs_breath] = findpeaks(breathing_signal, 'MinPeakDistance', fs*2);
    if length(locs_breath) > 1, breathing_rate_rpm = 60 / (mean(diff(locs_breath)) / fs); else, features = []; return; end
    heartbeat_signal_squared = heartbeat_signal .^ 2;
    prominence_threshold = 1.5 * std(heartbeat_signal_squared);
    [~, locs_heart] = findpeaks(heartbeat_signal_squared, 'MinPeakProminence', prominence_threshold, 'MinPeakDistance', fs*0.5);
    rr_intervals_raw = diff(locs_heart / fs) * 1000;
    if length(rr_intervals_raw) < 15, features = []; return; end
    physio_min_ms = 60000 / 180; physio_max_ms = 60000 / 40;
    outlier_indices = find(rr_intervals_raw < physio_min_ms | rr_intervals_raw > physio_max_ms);
    rr_intervals_cleaned = rr_intervals_raw;
    for i = 1:length(outlier_indices)
        idx = outlier_indices(i);
        local_start = max(1, idx - 2); local_end = min(length(rr_intervals_cleaned), idx + 2);
        local_indices = setdiff(local_start:local_end, outlier_indices);
        if ~isempty(local_indices), rr_intervals_cleaned(idx) = mean(rr_intervals_cleaned(local_indices));
        else
            global_good_indices = setdiff(1:length(rr_intervals_cleaned), outlier_indices);
            if ~isempty(global_good_indices), rr_intervals_cleaned(idx) = mean(rr_intervals_cleaned(global_good_indices)); end
        end
    end
    if length(rr_intervals_cleaned) < 15, features = []; return; end
    rr_intervals = rr_intervals_cleaned;
    rmssd = sqrt(mean(diff(rr_intervals).^2));
    sdnn = std(rr_intervals);
    fs_tachogram = 1 / (mean(rr_intervals) / 1000);
    [Pxx, f] = pwelch(rr_intervals - mean(rr_intervals), hann(length(rr_intervals)), [], [], fs_tachogram);
    lf_band = (f >= 0.04 & f < 0.15); hf_band = (f >= 0.15 & f <= 0.4);
    lf_power = trapz(f(lf_band), Pxx(lf_band)); hf_power = trapz(f(hf_band), Pxx(hf_band));
    if hf_power > 0, lf_hf_ratio = lf_power / hf_power; else, lf_hf_ratio = Inf; end
    [sd1, sd2, sd_ratio] = calculate_poincare(rr_intervals);
    sampen = calculate_sampen(rr_intervals, 2, 0.2 * std(rr_intervals));
    dfa_alpha1 = calculate_dfa_alpha1(rr_intervals);
    features = table(breathing_rate_rpm, rmssd, sdnn, lf_power, hf_power, lf_hf_ratio, sd1, sd2, sd_ratio, sampen, dfa_alpha1);
end
function [sd1, sd2, sd_ratio] = calculate_poincare(rr)
    rr_n = rr(1:end-1); rr_n_plus_1 = rr(2:end);
    sd1 = std(rr_n - rr_n_plus_1) / sqrt(2); sd2 = std(rr_n + rr_n_plus_1) / sqrt(2);
    if sd2 > 0, sd_ratio = sd1 / sd2; else, sd_ratio = NaN; end
end
function sampen = calculate_sampen(data, m, r)
    N = length(data); correls = zeros(1, 2);
    for k = 1:2
        m_k = m + k - 1;
        if N > m_k
            patterns = zeros(m_k, N - m_k);
            for i = 1:m_k, patterns(i, :) = data(i:N-m_k+i-1); end
            count = 0;
            for i = 1:(N - m_k)
                temp_patterns = patterns; temp_patterns(:, i) = [];
                distances = max(abs(temp_patterns - repmat(patterns(:, i), 1, N-m_k-1)));
                count = count + sum(distances < r);
            end
            correls(k) = count;
        end
    end
    if correls(2) > 0 && correls(1) > 0, sampen = -log(correls(2) / correls(1)); else, sampen = NaN; end
end
function alpha1 = calculate_dfa_alpha1(rr)
    data = cumsum(rr - mean(rr)); N = length(data);
    min_box_size = 4; max_box_size = 16;
    box_sizes = unique(round(logspace(log10(min_box_size), log10(max_box_size), 10)));
    fluctuations = zeros(size(box_sizes));
    for i = 1:length(box_sizes)
        n = box_sizes(i); num_segments = floor(N / n);
        rms_vals = zeros(1, num_segments);
        for v = 1:num_segments
            idx_start = (v - 1) * n + 1; idx_end = v * n;
            segment = idx_start:idx_end;
            p = polyfit(segment, data(segment), 1); fit = polyval(p, segment);
            rms_vals(v) = sqrt(mean((data(segment) - fit).^2));
        end
        fluctuations(i) = sqrt(mean(rms_vals.^2));
    end
    valid_indices = ~isinf(log10(box_sizes)) & ~isinf(log10(fluctuations)) & fluctuations > 0;
    if sum(valid_indices) > 1
        p = polyfit(log10(box_sizes(valid_indices)), log10(fluctuations(valid_indices)), 1);
        alpha1 = p(1);
    else, alpha1 = NaN; end
end