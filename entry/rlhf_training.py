from openlm.config import get_args
from openlm.engine.rlhf_training import main


if __name__ == "__main__":
    main(get_args()) # entry for training the model with reinforment learning via human feedbacks