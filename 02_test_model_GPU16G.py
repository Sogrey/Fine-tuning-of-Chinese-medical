import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import gc
from datetime import datetime

# 模型路径配置
BASE_MODEL = "models/Qwen/Qwen3-4B"
LORA_MODEL = "models/Sogrey/Chinese-medical-lora"

class MedicalQA_16GB_Optimized:
    def __init__(self):
        """初始化医疗QA系统（16GB GPU优化版）"""
        assert torch.cuda.is_available(), "需要NVIDIA GPU支持"
        assert torch.cuda.get_device_properties(0).total_memory >= 16*1024**3, "需要至少16GB显存"
        
        self.device = "cuda"
        self._setup_environment()
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            pad_token='<|extra_0|>'
        )
        self.model = None
        self._load_models()

    def _setup_environment(self):
        """配置运行环境"""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)
        gc.collect()
        torch.cuda.empty_cache()

    def _load_models(self):
        """加载完整精度模型和LoRA适配器"""
        print("🦾 正在为16GB GPU加载模型...")
        
        try:
            # 1. 加载基础模型（全精度）
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

            # 2. 加载LoRA适配器（合并到基础模型）
            self.model = PeftModel.from_pretrained(
                base_model,
                LORA_MODEL,
                device_map="auto"
            ).eval()

            print(f"✅ 模型加载完成！当前显存使用：{torch.cuda.memory_allocated()/1024**3:.2f}GB")
            
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def stream_answer(self, question, max_length=1024):
        """流式生成回答（支持更长上下文）"""
        try:
            inputs = self.tokenizer(question, return_tensors="pt", padding=True).to(self.device)
            
            # 优化生成配置（利用更大显存）
            generation_config = GenerationConfig(
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=20,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=151643
            )
            
            # 高性能生成（使用CUDA图优化）
            with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
                output_stream = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    streamer=None,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True  # 充分利用大显存
                )
            
            # 流式输出处理
            full_text = ""
            for seq in output_stream.sequences:
                decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
                new_text = decoded[len(question):].strip() if decoded.startswith(question) else decoded
                if new_text and new_text != full_text:
                    yield new_text[len(full_text):]
                    full_text = new_text
                    
        except Exception as e:
            raise RuntimeError(f"生成回答时出错: {str(e)}")
        finally:
            torch.cuda.empty_cache()

    def _cleanup(self):
        """资源清理"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

    def __del__(self):
        self._cleanup()

# 使用示例
if __name__ == "__main__":
    qa = None  # 先初始化为None
    try:
        print("=== 医疗问答系统 (16GB GPU高性能版) ===")
        print(f"基础模型路径: {BASE_MODEL}")
        print(f"LoRA适配器路径: {LORA_MODEL}")
        
        qa = MedicalQA_16GB_Optimized()
        
        while True:
            question = input("\n患者提问: ").strip()
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            print("\nAI医生思考中...", end="", flush=True)
            
            try:
                # 流式输出
                for chunk in qa.stream_answer(question):
                    print(chunk, end="", flush=True)
                print("\n")  # 回答结束后换行
                
            except Exception as e:
                print(f"\n❌ 生成错误: {str(e)}")
            
    except KeyboardInterrupt:
        print("\n👋 已退出")
    except Exception as e:
        print(f"\n❌ 系统错误: {str(e)}")
    finally:
        if qa is not None:  # 更安全的检查方式
            qa._cleanup()
