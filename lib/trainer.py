import os
import torch
from typing import Optional
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import logging


logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class LoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # TODO: 需要解构参数，添加 TrainArguments

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model_to_save = unwrap_model(self.model)

        # Create a state_dict for saving, similar to the PrefixTrainer approach
        if isinstance(model_to_save, PreTrainedModel):
            state_dict = {k: v.to("cpu") for k, v in model_to_save.named_parameters() if v.requires_grad}
            # Using Hugging Face's save_pretrained instead of PyTorch's torch.save
            model_to_save.save_pretrained(
                output_dir, state_dict=state_dict, save_function=torch.save, safe_serialization=False
            )
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

        # Save tokenizer and training arguments as usual
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        logger.info(self.args)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME,))


class PrefixTrainer(Trainer):
    def __init__(self, *args, save_changed=False, **kwargs):
        self.save_changed = save_changed
        super().__init__(*args, **kwargs)  # TODO: 需要解构参数，添加 TrainArguments

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")


        if not isinstance(self.model, PreTrainedModel):
            model_to_save = unwrap_model(self.model)

            if isinstance(model_to_save, PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                # Since transformers 4.35.2,safe_serialization are set `True`,
                # which will save model as `safetensors` format.
                model_to_save.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            if self.save_changed:
                logger.info("Saving PrefixEncoder")
                state_dict = self.model.state_dict()
                filtered_state_dict = {}

                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        filtered_state_dict[k] = state_dict[k]

                self.model.save_pretrained(output_dir, state_dict=filtered_state_dict, safe_serialization=False)
            else:
                logger.info("Saving the whole model")
                self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        logger.info(self.args)
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
