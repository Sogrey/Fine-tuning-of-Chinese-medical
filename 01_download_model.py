
# pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# https://modelscope.cn/
# Step1，下载 qwen3-4B
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-4B', cache_dir='./models')

model_dir = snapshot_download('Sogrey/Chinese-medical-lora', cache_dir='./models')