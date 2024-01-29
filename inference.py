import os, yaml
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from lib.arguments import ModelArguments, DataTrainingArguments, InferenceArguments


def run(config_path: str = "config.yml"):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_args = ModelArguments(**config["model_args"])
    data_args = DataTrainingArguments(**config["data_args"])
    train_args = Seq2SeqTrainingArguments(**config["train_args"])
    infer_args = InferenceArguments(**config["train_args"])

    tokenizer = AutoTokenizer.from_pretrained(infer_args.base_model_path, trust_remote_code=True)


    if infer_args.inference_type == "p_tuning":
        config = AutoConfig.from_pretrained(
            infer_args.base_model_path, trust_remote_code=True, pre_seq_len=infer_args.pt_pre_seq_len
        )
        model = AutoModel.from_pretrained(
            infer_args.base_model_path, config=config, trust_remote_code=True
        ).cuda()
        prefix_state_dict = torch.load(os.path.join(infer_args.pt_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(
            infer_args.base_model_path, load_in_8bit=False, trust_remote_code=True, device_map="auto"
        )

    model = model.to(infer_args.device)

    if infer_args.inference_type == "lora_tuning":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=True,
            target_modules=['query_key_value'],
            r=infer_args.lora_rank,
            lora_alpha=infer_args.lora_alpha,
            lora_dropout=infer_args.lora_dropout
        )
        model = get_peft_model(model, peft_config).to(infer_args.device)

        if os.path.exists(infer_args.lora_path):
            model.load_state_dict(torch.load(infer_args.lora_path), strict=False)

    # 第一种调用方式，另外一种则是直接使用 model.chat
    prompt1 = "对应微调任务的文本"
    inputs = tokenizer(prompt1, return_tensors="pt").to(infer_args.device)
    response1 = model.generate(
        input_ids=inputs["input_ids"],
        max_length=inputs["input_ids"].shape[-1] + infer_args.max_new_tokens
    )
    response1 = response1[0, inputs["input_ids"].shape[-1]:]
    print("Response:", tokenizer.decode(response1, skip_special_tokens=True))


if __name__ == "__main__":
    run()
