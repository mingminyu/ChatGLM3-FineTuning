from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

from .tokenization_chatglm import ChatGLMTokenizer


"""
reference: finetune_chatmodel_demo/preprocess_utils.py
reference: finetune_basemodel_demo/preprocess_utils.py

TODO: 代码中还有对工具调用的数据格式的调整，这里暂时用不上，后续补充
"""


def sanity_check(
    tokens: List[int],
    target: List[int],
    tokenizer: ChatGLMTokenizer
) -> None:
    print("<<<<<<<<<<<<< Sanity Check >>>>>>>>>>>>>")
    for t, m in zip(tokens, target):
        decoded = (
            tokenizer.tokenizer.index_special_tokens[t]
            if t in tokenizer.tokenizer.index_special_tokens else tokenizer.decode([t])
            )

        if t != 0:
            print("%20s: %6d -> %6d" % (repr(decoded), t, m))

    print("<<<<<<<<<<<<< Sanity Check >>>>>>>>>>>>>")
    assert len(tokens) == len(target), f"length mismatch: {len(tokens)} vs {len(target)}"


class InputOutputDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        tokenizer: ChatGLMTokenizer,
        max_source_length: int,
        max_target_length: int
    ):
        super(InputOutputDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, list]:
        data_item = self.data[i]
        # 这里 Base 模型需要的是 context/target，但是我们都按照 prompt/response 进行处理
        a_ids = self.tokenizer.encode(
            text=data_item['prompt'],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_length
            )
        b_ids = self.tokenizer.encode(
            text=data_item['response'],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_length
            )

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels += [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"
        return {"input_ids": input_ids, "labels": labels}


def format_conversation(
        item: Dict,
        tokenizer: ChatGLMTokenizer,
        conversation_key: Optional[str] = "conversations",
) -> Tuple:
    """格式化对话数据"""
    conversations = deepcopy(item[conversation_key])

    # Note: `loss_mask` here means whether *the prediction* of the token should take loss
    tokens, loss_masks = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")], [0, 0]

    def _update(_tokens: List[int], value: int = 1):
        value = int(value)
        tokens.extend(_tokens)
        loss_masks.extend([value] * len(_tokens))


    for idx, conv in enumerate(conversations):
        loss = conv.get("loss", True)

        if conv['role'] in {'system', 'user'}:
            loss = False

        text = tokenizer.build_single_message(conv['role'], "", conv["content"])
        _update(text, loss)

    _update([tokenizer.eos_token_id], False)

    assert len(tokens) == len(loss_masks), f"length mismatch: {len(tokens)} vs {len(loss_masks)}"
    return tokens, loss_masks


class MultiTurnDataset(Dataset):
    def __init__(
            self,
            data: List[dict],
            tokenizer: ChatGLMTokenizer,
            max_seq_length: int
    ):
        super(MultiTurnDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, list]:
        data_item = self.data[i]
        tokens, loss_masks = format_conversation(data_item, self.tokenizer, "conversations")

        # labels are used inside the model
        target_based_loss_mask = [False] + loss_masks[:-1]
        labels = [(t if m else -100) for t, m in zip(tokens, target_based_loss_mask)]

        tokens = tokens[:self.max_seq_length]
        labels = labels[:self.max_seq_length]
        tokens += [self.tokenizer.pad_token_id] * (self.max_seq_length - len(tokens))
        labels += [-100] * (self.max_seq_length - len(labels))

        assert len(tokens) == len(labels), f"length mismatch: {len(tokens)} vs {len(labels)}"

        return {"input_ids": tokens, "labels": labels}
