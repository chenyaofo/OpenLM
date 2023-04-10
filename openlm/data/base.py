import json
import copy
import typing

import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from openlm.utils.register import REGISTRY


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


def _generate_instrcution_response_from_sample(
    sample: dict,
    tokenizer: PreTrainedTokenizer,
):
    # if json has different layout, just modify this function

    # we have a template to wrap the raw instruction
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    format_instruction = prompt_input.format_map(sample) \
        if sample.get("input", "") != "" else prompt_no_input.format_map(sample)
    responses = f"{sample['output']}{tokenizer.eos_token}"

    return format_instruction, responses


def instrction_process(
    sample: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
):
    instruction, responses = _generate_instrcution_response_from_sample(
        sample=sample,
        tokenizer=tokenizer
    )

    input_ids: torch.Tensor = _tokenize(instruction+responses, tokenizer=tokenizer, max_length=max_length)
    instruction_ids: torch.Tensor = _tokenize(instruction, tokenizer=tokenizer, max_length=max_length)
    n_instruction_ids = instruction_ids.ne(tokenizer.pad_token_id).sum().item()

    # different from the pretraining LM task, we only calculate loss on the reponse but ignore the instruction
    labels = copy.deepcopy(input_ids)
    labels[:n_instruction_ids] = IGNORE_INDEX  # we ignore the cross entropy loss for format_instructions

    sample["input_ids"] = input_ids
    sample["labels"] = labels
    return sample


def collate_fn(
    samples,
    tokenizer: PreTrainedTokenizer,
):
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


class InstructionDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        transforms: typing.Callable,
    ):
        with open(filepath, "r") as f:
            self.raw_samples = json.load(f)
        self.transforms = transforms

    def __getitem__(self, index) -> dict:
        return self.transforms(self.raw_samples[index])

    def __len__(self):
        return len(self.raw_samples)


@REGISTRY.register
def instruct_json_dataset(
    filepath: str,
    tokenizer: PreTrainedTokenizer,
    max_token_length: int,
    batch_size: int,
    num_workers: int,
):
    # json file should be a list, each item in the list should contains three fields:
    #   instruction
    #   input
    #   output
    dataset = InstructionDataset(
        filepath=filepath,
        transforms=lambda sample: instrction_process(sample, tokenizer, max_token_length)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda samples: collate_fn(samples, tokenizer)
    )

    return loader
