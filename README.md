# EfficientSAM 公开数据定量复现实验

## 文件说明

- `evaluate_pet.py`：20张图像、两个官方模型、中心正点提示的评估代码；输出`results_pet`。
- `ablation_pet.py`：同样20张图像、三种提示条件、两个官方模型的消融代码；输出`results_ablation`。
- `requirements.txt`：Mac上已报告的软件版本。建议在新虚拟环境安装；版本可依硬件调整。

**来源说明：** 两份Python脚本根据本次聊天中已发送给用户的命令重新整理，并非直接从用户Mac硬盘导出的原始脚本。用户提供的实际运行证据是CSV、总结文件及PNG。代码重新运行会因为环境、权重或计时波动产生差异。

## 数据与模型

论文：Xiong et al., EfficientSAM, CVPR 2024。
官方代码： https://github.com/yformer/EfficientSAM
数据：Oxford-IIIT Pet，测试集3669张；实验固定`random.Random(42)`抽20张。
不附带数据集原始压缩包和官方模型权重。请使用原项目及公开数据集下载。

## 在 Mac 复跑

需要 Python 3.10 或更新版本（推荐 3.11）。先下载本仓库，把下列四个文件放入同一个文件夹。

```bash
git clone https://github.com/yformer/EfficientSAM.git
cp evaluate_pet.py ablation_pet.py requirements.txt EfficientSAM/
cd EfficientSAM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
unzip -o weights/efficient_sam_vits.pt.zip -d weights
python -c 'from torchvision.datasets import OxfordIIITPet; OxfordIIITPet(root="data", split="test", target_types="segmentation", download=True)'
python evaluate_pet.py
python ablation_pet.py
```

官方模型及权重来源：https://github.com/yformer/EfficientSAM 。请确认 `weights/efficient_sam_vitt.pt` 和解压后的 `weights/efficient_sam_vits.pt` 存在。
数据集下载需要额外磁盘空间，数据与权重均不上传到此仓库。

输出分别位于 `results_pet/` 与 `results_ablation/`，包括 CSV、汇总文本和可视化图片。

本次测试强制 `device=cpu`，即使 Mac 支持 MPS 也未用 MPS 测速。20 张图像来自 GT 辅助生成正负提示点；Oxford-IIIT Pet 标签 3（边界）被忽略，不是无提示分割或论文原训练复现。评价精度与论文 COCO/LVIS 上的 AP 不可直接比较。

## 依赖和核验状态

原附件将 OpenCV 版本写为 `4.13.0.0`，该版本的 PyPI 查询返回 404；此上传副本改用 `opencv-python>=4.10,<5`。其余版本保留原附件记录。此依赖列表不是经过完整重跑验证的锁定环境。

已检查两份脚本的 Python 语法；本次未下载数据、模型或重新运行完整实验。原始 CSV、结果图及运行截图保留在 AT3 作业附件中，不包含于此代码仓库。原附件说明中的“上一层运行数据”不适用于本仓库布局。
