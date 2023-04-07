from codebase.utils.register import REGISTRY
from transformers import AutoTokenizer

@REGISTRY.register
def opt_tokenizer(name_or_path, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    # this tokenizer does not have pad token, solve it by erferring to
    # https://github.com/huggingface/transformers/issues/3859
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer