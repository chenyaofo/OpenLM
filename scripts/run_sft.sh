export NUM_TRAINERS=8
export OUTPUT_DIR=output/test0
export CONFFILE=config/sft.hocon
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_TRAINERS \
    -m entry.sft --o $OUTPUT_DIR --conf $CONFFILE