# 基于“生理指纹图”与CNN的生理状态鲁棒感知

本项目旨在开发一套新颖且鲁棒的算法，用于对四种微妙的久坐生理状态（放松、专注、压力、困倦）进行分类。该算法基于非接触式的毫米波雷达信号，通过引入“生理指纹图”的概念，并结合卷积神经网络（CNN），以克服传统方法的局限性。

## 核心理念：“生理指纹图”

传统方法严重依赖于手动提取的HRV（心率变异性）数值特征，这类特征对于信号处理中的微小误差极为敏感。我们的方法将研究范式从**“计算脆弱的数值”**转向**“学习鲁棒的形态”**。我们将生理信号快照可视化为一张2D图像——“生理指纹图”，并训练CNN来学习与各生理状态相关的全局形态特征。

### 用于消融实验的指纹图版本

为了探究哪种生理信息的组合最为有效，我们设计了四种不同版本的指纹图进行对比实验：

-   **V1 (全信息版):** 一个2x2的图像，包含呼吸波形、心跳波形、RR间期序列图和HRV功率谱密度（PSD）。
-   **V2 (纯HRV版):** 一个1x2的图像，仅包含RR间期序列图和HRV功率谱。
-   **V3 (纯波形版):** 一个1x2的图像，仅包含呼吸波形和心跳波形。
-   **V4 (极简版):** 一个1x1的图像，仅包含信息最浓缩的HRV功率谱。

## 代码库结构

本项目是一个端到端的Python流水线，覆盖了从数据仿真、模型训练到对比分析的全过程。

-   `data_generator.py`: 仿真并生成四种生理状态的高保真`.npz`数据文件。
-   `signal_processor.py`: 包含从原始数据中提取呼吸、心跳、RR间期和PSD等中间信号的核心逻辑。
-   `fingerprint_generator.py`: 一个参数化的脚本（使用`--version`），用于生成V1-V4不同版本的指纹图。
-   `feature_extractor.py`: 提取传统的数值特征，用于训练基线模型。
-   `train_control_model.py`: 使用分层K折交叉验证，训练并评估一个强大的基线模型（随机森林）。
-   `train_cnn.py`: 一个参数化的脚本（使用`--dataset_dir`和`--output_file`），用于在任意版本的指纹图数据集上训练和评估CNN。
-   `compare_versions.py`: 一个专门的脚本，可自动加载所有版本的CNN实验结果，并生成最终的横向对比可视化图表。
-   `requirements.txt`: 一个灵活的Python依赖包列表，用于复现项目环境。

## 如何运行完整的多版本对比实验

请遵循以下步骤，以复现完整的对比实验流程。

**第一步：配置环境**
```bash
pip install -r requirements.txt
```

**第二步：生成基础原始数据**
```bash
python data_generator.py
```

**第三步：生成所有版本的指纹图数据集**
```bash
# 依次生成每个版本
python fingerprint_generator.py --version V1
python fingerprint_generator.py --version V2
python fingerprint_generator.py --version V3
python fingerprint_generator.py --version V4
```

**第四步：为每个版本的指纹图训练CNN模型**
这是一个计算密集型步骤，**强烈建议在GPU环境（如Google Colab）中运行**。
```bash
# 依次为每个数据集运行训练
python train_cnn.py --dataset_dir sedentary_images_dataset_v1 --output_file cnn_results_v1.json
python train_cnn.py --dataset_dir sedentary_images_dataset_v2 --output_file cnn_results_v2.json
python train_cnn.py --dataset_dir sedentary_images_dataset_v3 --output_file cnn_results_v3.json
python train_cnn.py --dataset_dir sedentary_images_dataset_v4 --output_file cnn_results_v4.json
```

**第五步：生成最终的横向对比可视化图表**
该脚本会自动查找所有`cnn_results_v*.json`文件并进行比较。
```bash
python compare_versions.py
```
运行后，将在控制台打印一个总结表格，并保存一张可视化的对比图到`version_accuracy_comparison.png`。
