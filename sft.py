from openlm.config import get_args
from openlm.engine.sft import main


if __name__ == "__main__":
    main(get_args()) # entry for supervised finetuning training