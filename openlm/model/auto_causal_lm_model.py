import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from openlm.utils.register import REGISTRY
from openlm.utils.common import get_n_trainable_parameters

_logger = logging.getLogger(__name__)


@REGISTRY.register
def causallm_model_with_tokenizer(
    model_name_or_path: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path
    )

    if use_lora:
        model = get_peft_model(
            model=model,
            peft_config=LoraConfig(
                peft_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
        )

    n_all_param, n_trainable_params = get_n_trainable_parameters(model)
    _logger.info(f"#all params= {n_all_param/1e6:.2f}M, #trainable params= {n_trainable_params/1e6:.2f}M, "
                 f"%trainable params= {n_trainable_params/n_all_param:.4f}%")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # this tokenizer does not have pad token, solve it by erferring to
    # https://github.com/huggingface/transformers/issues/3859
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer