"""
这个文件的存在是因为我本地是8G GPU,目标是需要16G GPU。
如果你的环境有16G GPU，可以直接运行这个文件：02_test_model_GPU16G.py
"""

"""
测试用例：

=== 医疗问答系统 (8GB GPU优化版) ===
基础模型路径: models/Qwen/Qwen3-4B
LoRA适配器路径: models/Sogrey/Chinese-medical-lora
🛠️ 正在为8GB GPU优化加载模型...
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████| 3/3 [00:27<00:00,  9.24s/it]
✅ 模型加载完成！当前显存使用：2.61GB

患者提问: 请问慢性肾炎患者能吃豆腐吗？

AI医生思考中...请问慢性肾炎患者能吃豆腐吗？

### 回答：
你好，肾脏疾病是临床常见的病症，其主要表现为蛋白尿、血尿、水肿、高血压等。一般需要通过饮食治疗和药物治疗相结合的方法 进行治疗的。建议平时注意休息，避免劳累，禁烟酒，低盐低脂饮食，高热量优质蛋白质饮食，忌食辛辣刺激性食物。，对于肾炎疾 病的出现，患者朋友们应该做到积极对症治疗，因为早期的肾炎是容易得到控制的。患者们不要错过治疗的好时机。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
import os
import gc
from datetime import datetime

# 模型路径配置
BASE_MODEL = "models/Qwen/Qwen3-4B"
LORA_MODEL = "models/Sogrey/Chinese-medical-lora"

class MedicalQA_8GB_Optimized:
    def __init__(self):
        """初始化医疗QA系统（8GB GPU优化版）"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.offload_dir = f"./offload_{datetime.now().strftime('%Y%m%d')}"
        self._setup_environment()
        
        # 提前初始化tokenizer（使用本地路径）
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            pad_token='<|extra_0|>'  # 设置明确的pad_token
        )
        self.model = None
        self._load_models()

    def _setup_environment(self):
        """配置运行环境"""
        os.makedirs(self.offload_dir, exist_ok=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_models(self):
        """加载量化模型和LoRA适配器"""
        print("🛠️ 正在为8GB GPU优化加载模型...")
        
        try:
            # 量化配置（兼容8GB显存）
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            # 显存分配策略（关键配置）
            memory_mapping = {
                0: "6GiB",    # 主GPU保留6GB
                "cpu": "24GiB" # 系统内存分配24GB
            }

            # 1. 加载基础模型（4位量化）
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                quantization_config=bnb_config,
                offload_folder=self.offload_dir,
                max_memory=memory_mapping,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            # 2. 加载LoRA适配器
            self.model = PeftModel.from_pretrained(
                base_model,
                LORA_MODEL,
                device_map="auto",
                offload_folder=self.offload_dir,
                max_memory=memory_mapping
            ).eval()

            print(f"✅ 模型加载完成！当前显存使用：{torch.cuda.memory_allocated()/1024**3:.2f}GB")
            
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def stream_answer(self, question, max_length=512):
        """流式生成回答"""
        try:
            inputs = self.tokenizer(question, return_tensors="pt", padding=True).to(self.device)
            
            # 显式设置Qwen模型特定的生成配置
            generation_config = GenerationConfig(
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=20,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=151643,  # Qwen特定的BOS token
                forced_eos_token_id=151645  # Qwen特定的EOS token
            )
            
            # 创建生成器时禁止调试信息
            with torch.no_grad():
                output_stream = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    streamer=None,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # 模拟流式输出
            generated_text = ""
            for seq in output_stream.sequences:
                new_text = self.tokenizer.decode(seq, skip_special_tokens=True)
                if new_text.startswith(question):
                    new_text = new_text[len(question):].strip()
                if new_text and new_text != generated_text:
                    yield new_text[len(generated_text):]
                    generated_text = new_text
                    
        except Exception as e:
            raise RuntimeError(f"生成回答时出错: {str(e)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _cleanup(self):
        """安全的资源清理"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

    def __del__(self):
        self._cleanup()

# 使用示例
if __name__ == "__main__":
    try:
        print("=== 医疗问答系统 (8GB GPU优化版) ===")
        print(f"基础模型路径: {BASE_MODEL}")
        print(f"LoRA适配器路径: {LORA_MODEL}")
        
        qa = MedicalQA_8GB_Optimized()
        
        while True:
            question = input("\n患者提问: ").strip()
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            print("\nAI医生思考中...\n", end="", flush=True)
            
            try:
                # 流式输出
                full_answer = ""
                for chunk in qa.stream_answer(question):
                    print(chunk, end="", flush=True)
                    full_answer += chunk
                
                print("\n")  # 回答结束后换行
                
            except Exception as e:
                print(f"\n❌ 生成错误: {str(e)}")
            
    except KeyboardInterrupt:
        print("\n👋 已退出")
    except Exception as e:
        print(f"\n❌ 系统错误: {str(e)}")
    finally:
        if 'qa' in locals():
            qa._cleanup()
