import copy

import torch

from transformers import PreTrainedTokenizer
from datasets import load_dataset

from openlm.utils.register import REGISTRY

from torchdata.datapipes.iter import Mapper, Shuffler, Batcher, Collator, ShardingFilter
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence


def _tokenize(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
):
    return tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
    )["input_ids"][0]


IGNORE_INDEX = -100  # default ignore index in cross entropy loss
PROMPT_DICT = {
    "prompt_input":
        ("Below is an instruction that describes a task, paired with an input that provides further context. "
         "Write a response that appropriately completes the request.\n\n"
         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
    "prompt_no_input": ("Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"),
}


def instrction_process(
    sample: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
):
    # we have a template to wrap the raw instruction
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    format_instruction = prompt_input.format_map(sample) \
        if sample.get("input", "") != "" else prompt_no_input.format_map(sample)
    targets = f"{sample['output']}{tokenizer.eos_token}"

    input_ids: torch.Tensor = _tokenize(format_instruction+targets, tokenizer=tokenizer, max_length=max_length)
    format_instruction_ids: torch.Tensor = _tokenize(format_instruction, tokenizer=tokenizer, max_length=max_length)
    n_format_instruction_ids = format_instruction_ids.ne(tokenizer.pad_token_id).sum().item()

    # different from the pretraining LM task, we only calculate loss on the reponse but ignore the instruction
    labels = copy.deepcopy(input_ids)
    labels[:n_format_instruction_ids] = IGNORE_INDEX  # we ignore the cross entropy loss for format_instructions

    sample["input_ids"] = input_ids
    sample["labels"] = labels
    return sample


def collate_fn(
    samples,
    tokenizer: PreTrainedTokenizer,
):
    # import ipdb; ipdb.set_trace()
    input_ids, labels = tuple([sample[key] for sample in samples] for key in ("input_ids", "labels"))
    input_ids = pad_sequence(input_ids,
                             batch_first=True,
                             padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


@REGISTRY.register
def instruct_json_dataset(
    filepath: str,
    tokenizer: PreTrainedTokenizer,
    max_token_length: int,
):
    dataset = load_dataset(
        "json",
        data_files=filepath,
        split="train",
        streaming=True  # for more efficient data loading
    )
    dataset.shuffle
    dataset = dataset.map(lambda sample: instrction_process(sample, tokenizer, max_token_length), batched=False)
    dataset = dataset.with_format("torch")

    dataset = Shuffler(dataset, buffer_size=1000)
    # dataset = Mapper(dataset, fn=lambda sample: (sample["input_ids"], sample["labels"]))
    dataset = Batcher(dataset, batch_size=64)
    dataset = Collator(dataset, collate_fn=lambda samples: collate_fn(samples, tokenizer))

    loader = DataLoader(dataset, num_workers=0)

    return loader
