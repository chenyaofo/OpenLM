# OpenLM
A Codebase for Training Large Language Model via Finetuning and RLHF.

deepspeed --include localhost:0,1,2,3 sft.py --conf config/sft.hocon -o output/test1

deepspeed sft.py --conf config/sft.hocon -o output/test2