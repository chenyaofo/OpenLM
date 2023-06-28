import logging

from transformers.deepspeed import HfDeepSpeedConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from torch4x.register import REGISTRY


from deepspeed.utils import logger as _logger
from deepspeed.utils import log_dist

from .lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters


def is_zero3(ds_config: dict):
    return ds_config is not None and ds_config["zero_optimization"]["stage"] == 3


@REGISTRY.register
def causallm_model_with_tokenizer(
    model_name_or_path: str,
    gradient_checkpointing_enable: bool,
    use_lora: bool,
    ds_config: dict,
    lora_module_name: str,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
):
    if is_zero3(ds_config=ds_config):
        log_dist("Since zero stage is set to 3, the model would be loaded in shards mode", ranks=[0])
        hf_ds_config = HfDeepSpeedConfig(ds_config) if is_zero3(ds_config) else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )

    # if gradient_checkpointing_enable:
    #     model.gradient_checkpointing_enable()

    if use_lora:
        if gradient_checkpointing_enable:
            raise RuntimeError("Gradient Checkpointing and LoRA cannot be enabled at the same time.")
        model = convert_linear_layer_to_lora(
            model,
            part_module_name=lora_module_name,
            lora_dim=lora_r,
            lora_scaling=lora_alpha,
            lora_droppout=lora_dropout
        )
        model = only_optimize_lora_parameters(model)
        # if gradient_checkpointing_enable:
        #     model.enable_input_require_grads()
        # model = get_peft_model(
        #     model=model,
        #     peft_config=LoraConfig(
        #         peft_type=TaskType.CAUSAL_LM,
        #         inference_mode=False,
        #         r=lora_r,
        #         lora_alpha=lora_alpha,
        #         lora_dropout=lora_dropout
        #     )
        # )
        # model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    # this tokenizer does not have pad token, solve it by erferring to
    # https://github.com/huggingface/transformers/issues/3859
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
