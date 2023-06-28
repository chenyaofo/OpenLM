GPUS='pzl002:4,5,6,7'

OUTDIR=output-1/test0
CONFIG=config/default.hocon

mkdir -p $OUTDIR

deepspeed -i $GPUS main.py -o $OUTDIR -c $CONFIG | tee $OUTDIR/log.txt