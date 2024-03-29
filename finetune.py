import os
import sys
import json
import yaml
import torch
import logging
from sklearn.model_selection import train_test_split
from art.art import tprint
from transformers.utils import logging as tf_logging  # type: ignore
from transformers import AutoConfig, AutoTokenizer, AutoModel, set_seed  # type: ignore
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

from lib.arguments import ModelArguments, DataTrainingArguments
from lib.preprocess import sanity_check, InputOutputDataset, MultiTurnDataset
from lib.trainer import PrefixTrainer, LoRATrainer


tprint("ChatGLM3", "tarty1")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    )


def run(config_path: str = "config.yml"):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_args = ModelArguments(**config["model_args"])
    data_args = DataTrainingArguments(**config["data_args"])
    train_args = Seq2SeqTrainingArguments(**config["train_args"])

    if train_args.should_log:
        tf_logging.set_verbosity_info()

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    tf_logging.set_verbosity(log_level)
    tf_logging.enable_default_handler()
    tf_logging.enable_explicit_format()

    logger.warning(f"""
        Process rank: {train_args.local_rank}
        device: {train_args.device}
        n_gpu: {train_args.n_gpu}
        distributed training: {bool(train_args.local_rank != -1)}
        16-bits training: {train_args.fp16}
        """)
    logger.info(f"Training/Evaluation parameters \n{train_args}")
    set_seed(train_args.seed)

    config = AutoConfig.from_pretrained(model_args.base_model_path, trust_remote_code=True)
    config.use_cache = False
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model_path, trust_remote_code=True)

    logger.info(f"Finetuning Type: {data_args.finetune_type}")
    logger.info(f"Data Format: {data_args.train_format}")

    if data_args.finetune_type == "p_tuning" and model_args.ptuning_checkpoint is not None:
        model = AutoModel.from_pretrained(model_args.base_model_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}

        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v

        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_args.base_model_path, config=config, trust_remote_code=True)


    if model_args.quantization_bit is not None:
        logger.info(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    if data_args.finetune_type == "p_tuning":
        if model_args.pre_seq_len is not None:
            # P-tuning v2
            model = model.half()
            model.transformer.prefix_encoder.float()
        else:
            # Finetune
            model = model.float()

    with open(data_args.train_file, "r", encoding="utf-8") as f:
        if data_args.train_file.endswith(".json"):
            train_data = json.load(f)
        elif data_args.train_file.endswith(".jsonl"):
            train_data = [json.loads(line) for line in f]
        else:
            raise ValueError("train_file parameter must be a json file!")


    if data_args.eval_file is None:
        _, eval_data = train_test_split(
            train_data, test_size=0.3, random_state=42
        )
    else:
        if data_args.eval_file.endswith(".json"):
            eval_data = json.load(f)
        elif data_args.eval_file.endswith(".jsonl"):
            eval_data = [json.loads(line) for line in f]
        else:
            raise ValueError("eval_file parameter must be a json file!")


    if data_args.train_format == "multi-turn":
        train_dataset = MultiTurnDataset(
            train_data,
            tokenizer,
            data_args.max_seq_length,
        )
        eval_dataset = MultiTurnDataset(
            eval_data,
            tokenizer,
            data_args.max_seq_length,
        )
    elif data_args.train_format == "input-output":
        train_dataset = InputOutputDataset(
            train_data,
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
        )
        eval_dataset = InputOutputDataset(
            eval_data,
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
        )
    else:
        raise ValueError(f"Unknown train format: {data_args.train_format}")

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    sanity_check(train_dataset[0]['input_ids'], train_dataset[0]['labels'], tokenizer)

    if data_args.finetune_type == "lora_tuning":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_rank,
            target_modules=['query_key_value'],
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config).to("cuda")


    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    if data_args.finetune_type == "lora_tuning":
        trainer = LoRATrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        trainer = PrefixTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            save_changed=model_args.pre_seq_len is not None
        )

    output_dir = train_args.output_dir
    dirlist = os.listdir(output_dir)
    checkpoints_num = 0

    for checkpoint_str in dirlist:
        if checkpoint_str.find("checkpoint") > 0:
            checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
            if checkpoint > checkpoints_num:
                checkpoints_num = checkpoint

    if train_args.resume_from_checkpoint is not None:
        is_auto_resume_from_checkpoint = True
    else:
        is_auto_resume_from_checkpoint = False

    if is_auto_resume_from_checkpoint and checkpoints_num > 0:
        # If there is a breakpoint, continue training at the breakpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        checkpoint_dir = os.path.join(output_dir, "checkpoint-" + str(checkpoints_num))
        logger.info(checkpoint_dir)
        trainer.train(resume_from_checkpoint=checkpoint_dir)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()
    else:
        # Train normally without breakpoints
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()


if __name__ == "__main__":
    run(config_path="config.yml")
