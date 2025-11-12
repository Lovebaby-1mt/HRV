%% create_fingerprint_dataset.m
% -------------------------------------------------------------------------
% 目标: 
% 1. 遍历 /sedentary_dataset/ 中的所有 .mat 文件。
% 2. 调用 process_gold_standard.m 获取中间信号。
% 3. 将4个中间信号绘制为2x2的“生理指纹图”。
% 4. 将图像保存到 /sedentary_images_dataset/ 下的对应分类文件夹。
%
% Plan C 核心脚本
% -------------------------------------------------------------------------

clear; clc; close all;

%% 1. 设置路径与参数
% 输入: 包含 'relaxed', 'flow', 'stress', 'drowsiness' 子文件夹的 .mat 数据集
inputBaseDir = 'sedentary_dataset'; 

% 输出: CNN 训练用的图像数据集
outputBaseDir = 'sedentary_images_dataset';

% 状态类别
categories = {'relaxed', 'flow', 'stress', 'drowsiness'};

% 图像参数
outputSize = [256, 256]; % 输出图像像素 (宽, 高)
outputResolution = 72;   % 图像分辨率

%% 2. 循环处理数据
disp('开始生成生理指纹图数据集 (Plan C)...');

for c = 1:length(categories)
    currentCategory = categories{c};
    disp(['--- 处理类别: ', currentCategory, ' ---']);
    
    % 创建输出子文件夹
    outputCatDir = fullfile(outputBaseDir, currentCategory);
    if ~exist(outputCatDir, 'dir')
        mkdir(outputCatDir);
    end
    
    % 获取该类别下的所有 .mat 文件
    inputCatDir = fullfile(inputBaseDir, currentCategory);
    matFiles = dir(fullfile(inputCatDir, '*.mat'));
    
    for i = 1:length(matFiles)
        matFileName = matFiles[i].name;
        fullMatPath = fullfile(inputCatDir, matFileName);
        
        fprintf('处理文件: %s\n', matFileName);
        
        try
            %% 3. 加载数据并调用“金标准”函数
            
            % 加载数据
            data = load(fullMatPath);
            fs = 1 / (data.time_axis(2) - data.time_axis(1));
            
            % -------------------------------------------------------------
            % **核心步骤: 调用我们重构好的函数**
            % -------------------------------------------------------------
            [breathing_signal, heartbeat_signal, rr_intervals_cleaned, Pxx, f] = ...
                 process_gold_standard(data.radar_signal, fs);

            
            %% 4. 核心绘图逻辑：创建 2x2 指纹图
            
            % 创建一个不可见的 figure
            fig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [0 0 outputSize(1) outputSize(2)]);
            
            % 使用 tiledlayout 实现“零”间距
            t = tiledlayout(2, 2, 'TileSpacing', 'none', 'Padding', 'none');
            
            % (1,1) 左上角: 呼吸波形图
            % ---------------------------------
            nexttile(1);
            if ~isempty(breathing_signal)
                % 使用 Z-Score 标准化信号，让CNN只关注形态，忽略绝对幅度
                plot(zscore(breathing_signal));
            end
            axis off; % 关闭所有UI元素：坐标轴、刻度、标签
            
            % (1,2) 右上角: 心跳波形图
            % ---------------------------------
            nexttile(2);
            if ~isempty(heartbeat_signal)
                plot(zscore(heartbeat_signal));
            end
            axis off;
            
            % (2,1) 左下角: RR间期序列图
            % ---------------------------------
            nexttile(3);
            if ~isempty(rr_intervals_cleaned)
                % 绘制带标记的折线图，以显示“野点”和趋势
                plot(rr_intervals_cleaned, 'o-'); 
            end
            axis off;
            
            % (2,2) 右下角: HRV功率谱密度图
            % ---------------------------------
            nexttile(4);
            if ~isempty(Pxx) && ~isempty(f)
                plot(f, Pxx);
                % 关键: 固定X轴范围，使所有PSD图具有可比性
                xlim([0 0.5]); 
            end
            axis off;
            
            
            %% 5. 保存图像
            outputImageName = strrep(matFileName, '.mat', '.png');
            outputImagePath = fullfile(outputCatDir, outputImageName);
            
            % 使用 exportgraphics 保存“干净”的图像
            exportgraphics(fig, outputImagePath, 'Resolution', outputResolution);
            
            close(fig); % 关闭 figure 释放内存
            
        catch ME
            fprintf('!! 处理失败: %s. 错误: %s (Line: %d)\n', matFileName, ME.message, ME.stack(1).line);
            if exist('fig', 'var') && ishandle(fig)
                 close(fig); % 确保出错时关闭 figure
            end
        end
    end
end

disp('--- 生理指纹图数据集生成完毕! ---');