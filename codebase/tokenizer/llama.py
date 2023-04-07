from codebase.utils.register import REGISTRY
from transformers import AutoTokenizer


@REGISTRY.register
def llama_tokenizer(name_or_path, **kwargs):
    raise NotImplementedError("Some tokens are not processed correctly, waiting for fixing.")
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        padding_side="right",
        use_fast=False,
    )
    # this tokenizer does not have pad token, solve it by erferring to
    # https://github.com/huggingface/transformers/issues/3859
    tokenizer.eos_token = '<\s>'
    tokenizer.pad_token = tokenizer.eos_token
    # TODO
    # it may be required to ersize the embeding layers
    # refer to https://github.com/hpcaitech/ColossalAI/blob/main/applications/Chat/coati/utils/tokenizer_utils.py
    return tokenizer
