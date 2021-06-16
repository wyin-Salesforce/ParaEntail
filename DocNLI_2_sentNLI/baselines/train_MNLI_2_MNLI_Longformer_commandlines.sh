export BATCHSIZE=12 #32 is also ok
export EPOCHSIZE=5
export LEARNINGRATE=1e-6


CUDA_VISIBLE_DEVICES=5 python -u train_MNLI_2_MNLI_Longformer.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 42 > log.longformer.mnli.2.mnli.seed.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=6 python -u train_MNLI_2_MNLI_Longformer.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 16 > log.longformer.mnli.2.mnli.seed.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 python -u train_MNLI_2_MNLI_Longformer.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 32 > log.longformer.mnli.2.mnli.seed.32.txt 2>&1 &
