from typing import Optional, Literal
from pydantic import BaseModel, Field


# reference: ChatGLM3/finetune_chatmodel_demo/arguments.py
# reference: ChatGLM3/finetune_basemodel_demo/arguments.py

class ModelArguments(BaseModel):
    base_model_path: str = "THUDM/chatglm3-6b"
    ptuning_checkpoint: Optional[str] = None
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None  # 下载 HuggingFace 模型的缓存路径
    use_fast_tokenizer: Optional[bool] = True
    revision: Optional[str] = "main"
    use_auth_token: Optional[bool] = False  # 用于下载 HuggingFace 模型的 token
    resize_position_embeddings: Optional[bool] = None
    quantization_bit: Optional[int] = None  # 量化模型的位数
    pre_seq_len: Optional[int] = None
    prefix_projection: Optional[bool] = False
    # LORA Arguments: 只有在 LORA 微调时才会被使用
    lora_rank: Optional[int] = 8  # 平衡模型的复杂度和灵活度，更大 rank 值可以使模型适配更好，但会消耗更多计算资源
    lora_alpha: Optional[int] = 32  # 更大值意味着调整权重更大，但有过拟合的风险
    lora_dropout: Optional[float] = 0.1  # 提高模型泛化的参数


class DataTrainingArguments(BaseModel):
    train_file: str
    eval_file: Optional[str] = None
    train_format: Literal["multi-turn", "input-output"]  # 微调的数据格式
    finetune_type: Literal["p_tuning", "lora_tuning"] = "p_tuning"
    max_seq_length: Optional[int] = 1024  # 输入序列的最大长度，超过则会被截断，少于则会被填充
    max_source_length: Optional[int] = 1024  # 输入序列的最大长度，超过则会被截断，少于则会被填充
    max_target_length: Optional[int] = 128  # 输入序列的最大长度，超过则会被截断，少于则会被填充
    overwrite_cache: Optional[bool] = False  # 覆盖缓存的训练和验证数据集
    preprocessing_num_workers: Optional[int] = 1  # 预处理所使用的进程数
    pad_to_max_length: Optional[bool] = False  # 是否将输入序列填充到最大长度，为 False 时则在每个 batch 会动态填充
    max_train_samples: Optional[int] = None



class InferenceArguments(BaseModel):
    base_model_path: str = "THUDM/chatglm3-6b"
    load_in_8bit: Optional[bool] = False
    max_new_tokens: Optional[int] = 128
    inference_type: Literal["p_tuning", "lora_tuning"] = "p_tuning"
    pt_checkpoint: Optional[str] = None
    pre_seq_len: Optional[int] = 128
    lora_checkpoint: Optional[str] = None
    lora_rank: Optional[int] = 8  # 平衡模型的复杂度和灵活度，更大 rank 值可以使模型适配更好，但会消耗更多计算资源
    lora_alpha: Optional[int] = 32  # 更大值意味着调整权重更大，但有过拟合的风险
    lora_dropout: Optional[float] = 0.1  # 提高模型泛化的参数
    device: Literal["cpu", "cuda"] = "cuda"

