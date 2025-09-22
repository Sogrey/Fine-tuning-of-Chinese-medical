# Fine-tuning-of-Chinese-medical
中文医疗模型微调

算力平台：[AutoDL算力云 ](https://www.autodl.com)

选择配置：

- **镜像**：PyTorch 2.5.1  Python 3.12(ubuntu22.04)  CUDA 12.4
- **GPU**：RTX 4090(24GB) * 1
- **CPU**：16 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
- **内存**：120GB
- **硬盘**：系统盘:30 GB  数据盘:免费:50GB  付费:0GB

基于 Qwen3:4B 模型为基座模型，以[Chinese medical dialogue data 中文医疗问答数据集](https://github.com/Toyhom/Chinese-medical-dialogue-data)为训练数据。

``` bash
git clone --recursive https://github.com/Sogrey/Fine-tuning-of-Chinese-medical.git
```

