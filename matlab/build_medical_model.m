% =========================================================================
% Build Medical Model Prototype for Sedentary State Classification
% Author: Gemini
% Date: 2025-10-13
% Description: This script batch-processes the entire dataset using the
%              robust filter-based feature extractor, and then trains a
%              machine learning classifier to identify the physiological state.
% =========================================================================
clear; clc; close all;

%% --- 1. 批量特征提取 (Batch Feature Extraction) ---
% -------------------------------------------------------------------------
fprintf('开始批量处理数据集并提取特征...\n');
DATASET_DIR = 'sedentary_dataset';
STATES = {'relaxed', 'flow', 'stress', 'drowsiness'};

feature_table = table(); % 创建一个空的table来存储特征和标签

for state_idx = 1:length(STATES)
    current_state = STATES{state_idx};
    state_dir = fullfile(DATASET_DIR, current_state);
    files = dir(fullfile(state_dir, '*.mat'));
    
    fprintf('正在处理状态: %s (%d个文件)\n', current_state, length(files));
    
    for file_idx = 1:length(files)
        filePath = fullfile(state_dir, files(file_idx).name);
        data = load(filePath);
        
        % 调用我们的核心特征提取函数
        features = extract_features_from_signal(data.radar_signal, 100);
        
        if ~isempty(features)
            % 将特征和标签添加到table中
            features.Label = categorical({current_state}); % 标签必须是categorical类型
            feature_table = [feature_table; features];
        end
    end
end

fprintf('\n特征提取完成！总共提取了 %d 个有效样本。\n', height(feature_table));
save('feature_dataset.mat', 'feature_table'); % 保存特征数据集

%% --- 2. 模型训练与评估 (Model Training & Evaluation) ---
% -------------------------------------------------------------------------
fprintf('正在训练分类模型...\n');

% 划分数据集为训练集和测试集 (例如, 80%训练, 20%测试)
cv = cvpartition(height(feature_table), 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);
training_data = feature_table(idxTrain, :);
testing_data = feature_table(idxTest, :);

% 训练一个分类树模型 (这是一个很好的起点)
% 您也可以尝试 'fitcsvm' (SVM) 或 'fitcensemble' (随机森林)
mdl = fitctree(training_data, 'Label');

% 在测试集上进行预测
predictions = predict(mdl, testing_data);
true_labels = testing_data.Label;

% 评估模型性能
accuracy = sum(predictions == true_labels) / length(true_labels);
fprintf('模型在测试集上的准确率: %.2f%%\n', accuracy * 100);

% 显示混淆矩阵，直观地看分类效果
figure;
confusionchart(true_labels, predictions);
title(sprintf('分类混淆矩阵 (准确率: %.2f%%)', accuracy * 100));

%% ========== 核心特征提取函数 ==========
function features = extract_features_from_signal(radar_signal, fs)
    % 这是一个封装了我们之前脚本核心逻辑的函数
    
    % 1. 信号分离
    lpFilt = designfilt('lowpassiir', 'FilterOrder', 4, 'HalfPowerFrequency', 0.8, 'SampleRate', fs);
    breathing_signal = filtfilt(lpFilt, radar_signal);
    bpFilt = designfilt('bandpassiir', 'FilterOrder', 4, 'HalfPowerFrequency1', 0.9, 'HalfPowerFrequency2', 2.5, 'SampleRate', fs);
    heartbeat_signal = filtfilt(bpFilt, radar_signal);
    
    % 2. 呼吸率
    [~, locs_breath] = findpeaks(breathing_signal, 'MinPeakDistance', fs*2);
    if length(locs_breath) > 1, breathing_rate_rpm = 60 / (mean(diff(locs_breath)) / fs); else, features = []; return; end
    
    % 3. RR间期
    heartbeat_signal_normalized = heartbeat_signal / max(abs(heartbeat_signal));
    heartbeat_signal_squared = heartbeat_signal_normalized .^ 2;
    prominence_threshold = 0.1;
    [~, locs_heart] = findpeaks(heartbeat_signal_squared, 'MinPeakProminence', prominence_threshold, 'MinPeakDistance', fs*0.5);
    extracted_rr_intervals_ms = diff(locs_heart / fs) * 1000;
    
    if length(extracted_rr_intervals_ms) < 10, features = []; return; end % 如果提取的RR点太少，则放弃此样本
        
    % 4. HRV分析
    rmssd = sqrt(mean(diff(extracted_rr_intervals_ms).^2));
    sdnn = std(extracted_rr_intervals_ms);
    fs_tachogram = 1 / (mean(extracted_rr_intervals_ms) / 1000);
    [Pxx, f] = pwelch(extracted_rr_intervals_ms - mean(extracted_rr_intervals_ms), hann(length(extracted_rr_intervals_ms)), [], [], fs_tachogram);
    lf_band = (f >= 0.04 & f < 0.15);
    hf_band = (f >= 0.15 & f <= 0.4);
    lf_power = trapz(f(lf_band), Pxx(lf_band));
    hf_power = trapz(f(hf_band), Pxx(hf_band));
    if hf_power > 0, lf_hf_ratio = lf_power / hf_power; else, lf_hf_ratio = Inf; end
    
    % 5. 返回特征表
    features = table(breathing_rate_rpm, rmssd, sdnn, lf_power, hf_power, lf_hf_ratio);
end