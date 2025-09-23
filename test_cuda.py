import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 测试GPU计算
    x = torch.rand(5, 3).cuda()
    print(f"张量在设备: {x.device}")
    print("GPU测试成功!")
else:
    print("CUDA不可用，请检查PyTorch安装和CUDA配置")
    
    # 提供更多诊断信息
    import sys
    import os
    import platform
    
    print("\n===== 系统信息 =====")
    print(f"操作系统: {platform.system()} {platform.version()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"Python路径: {sys.executable}")
    
    print("\n===== PyTorch信息 =====")
    print(f"PyTorch安装路径: {torch.__file__}")
    
    print("\n===== CUDA环境变量 =====")
    cuda_path = os.environ.get("CUDA_PATH", "未设置")
    print(f"CUDA_PATH: {cuda_path}")
    
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "未设置")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # 检查是否有CUDA相关路径在PATH中
    path_env = os.environ.get("PATH", "")
    cuda_in_path = any("cuda" in p.lower() for p in path_env.split(os.pathsep))
    print(f"PATH中包含CUDA路径: {cuda_in_path}")
    
    # 尝试导入其他CUDA相关包
    try:
        import torch.cuda
        print("\n尝试获取CUDA属性:")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
    except Exception as e:
        print(f"导入torch.cuda时出错: {str(e)}")