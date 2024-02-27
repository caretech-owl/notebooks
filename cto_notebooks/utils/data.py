from typing import Dict, Generator, List

import torch
from transformers import PreTrainedTokenizer


def split_chunks(
    arr: List[int], size: int, step: int
) -> Generator[List[int], None, None]:
    for i in range(0, len(arr), step):
        yield arr[i : i + size]


def encode(
    text: str, add_bos_token: bool, tokenizer: PreTrainedTokenizer, cutoff_len: int
) -> List[int]:
    result = tokenizer.encode(text, truncation=True, max_length=cutoff_len)
    # Check if the first two tokens are BOS
    if len(result) >= 2 and result[:2] == [
        tokenizer.bos_token_id,
        tokenizer.bos_token_id,
    ]:
        result = result[1:]

    if not add_bos_token and result[0] == tokenizer.bos_token_id:
        result = result[1:]
    return result


def tokenize(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    cutoff_len: int,
    append_eos_token: bool = False,
) -> Dict[str, List]:
    input_ids = encode(prompt, True, tokenizer, cutoff_len)

    if (
        append_eos_token
        and input_ids[-1] != tokenizer.eos_token_id
        and len(input_ids) < cutoff_len
    ):
        input_ids.append(tokenizer.eos_token_id)

    # TODO: Is is reasonable to set padding(ton) token to eos_token when not set?
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = [tokenizer.pad_token_id] * (cutoff_len - len(input_ids)) + input_ids
    labels = [1] * len(input_ids)

    input_ids = torch.tensor(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id),
    }
