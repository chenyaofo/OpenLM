'''
Modified from https://raw.githubusercontent.com/hpcaitech/ColossalAI/main/applications/Chat/coati/dataset/sft_dataset.py

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
'''
import io
import copy
import json
import logging
import typing
import pathlib

import torch
import transformers

from torch.utils.data import Dataset

_logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input":
        ("Below is an instruction that describes a task, paired with an input that provides further context. "
         "Write a response that appropriately completes the request.\n\n"
         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
    "prompt_no_input": ("Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"),
}


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def _tokenize_fn(texts: typing.Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> typing.Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ) for text in texts
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: typing.Sequence[str],
    targets: typing.Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
) -> typing.Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, max_length) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        cache_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_datasets_size: int = None,
        max_length: int = 512
    ):
        super(SupervisedDataset, self).__init__()

        data_path: pathlib.Path = pathlib.Path(data_path)
        cache_path: pathlib.Path = pathlib.Path(cache_path)

        if cache_path.exists():
            _logger.info(f"Loading data from cache {cache_path} with {n_samples} samples")
            # the dataset to finetuned may be much smalle than the pretraining data of LLM
            # hence we just simply use torch pickle to cache to tokens
            # if the dataset is very large, tfrecord is more preferred
            cache_tokens = torch.load(cache_path, map_location="cpu")
            self.input_ids = cache_tokens["input_ids"]
            self.label_ids = cache_tokens["label_ids"]
            n_samples = len(self.label_ids)

            if max_datasets_size is not None:
                _logger.info(f"Limiting dataset to {max_datasets_size} examples.")
                self.input_ids = self.input_ids[:max_datasets_size]
                self.label_ids = self.label_ids[:max_datasets_size]
        else:
            raw_data: typing.List[typing.Dict[str, str]] = jload(data_path)
            n_samples = len(raw_data)
            _logger.info(f"Loading data from json file {data_path} with {n_samples} samples")

            if max_datasets_size is not None:
                _logger.info(f"Limiting dataset to {max_datasets_size} examples.")
                raw_data = raw_data[:max_datasets_size]

            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example)
                if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in raw_data
            ]

            targets = [f"{example['output']}{tokenizer.eos_token}" for example in raw_data]

            _logger.info("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer, max_length)

            self.input_ids = data_dict["input_ids"]
            self.label_ids = data_dict["labels"]

            if cache_path.exists():
                _logger.info(f"Saving preprocssed tokens to cache {cache_path}")
                torch.save(data_dict, cache_path)


    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i) -> typing.Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.label_ids[i])
