import logging

from transformers.deepspeed import HfDeepSpeedConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from openlm.utils.register import REGISTRY
from openlm.utils.common import get_n_trainable_parameters

_logger = logging.getLogger(__name__)


def is_zero3(ds_config: dict):
    return ds_config is not None and ds_config["zero_optimization"]["stage"] == 3



@REGISTRY.register
def causallm_model_with_tokenizer(
    model_name_or_path: str,
    gradient_checkpointing_enable:bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    ds_config: dict,
):
    if is_zero3(ds_config=ds_config):
        _logger.info("Since zero stage is set to 3, the model would be loaded in shards mode")
        hf_ds_config = HfDeepSpeedConfig(ds_config) if is_zero3(ds_config) else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
    )

    if gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()

    if use_lora:
        if gradient_checkpointing_enable:
            model.enable_input_require_grads()
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
    # import ipdb; ipdb.set_trace()
    # n_all_param, n_trainable_params = get_n_trainable_parameters(model)
    # _logger.info(f"#all params={n_all_param/1e6:.2f}M, #trainable params={n_trainable_params/1e6:.2f}M, "
    #              f"%trainable params={n_trainable_params/n_all_param*100.0:.4f}%")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # this tokenizer does not have pad token, solve it by erferring to
    # https://github.com/huggingface/transformers/issues/3859
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
