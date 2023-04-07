from codebase.utils.register import REGISTRY
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

@REGISTRY.register
def gpt2_tokenizer(name_or_path, **kwargs):
    if name_or_path is None:
        name_or_path = "gpt2" # it would be downloaded from huggingface.co
    tokenizer = GPT2Tokenizer.from_pretrained(name_or_path)
    # this tokenizer does not have pad token, solve it by erferring to
    # https://github.com/huggingface/transformers/issues/3859
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer