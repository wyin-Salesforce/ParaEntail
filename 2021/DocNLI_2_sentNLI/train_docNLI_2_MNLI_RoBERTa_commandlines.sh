export BATCHSIZE=4 #32 is also ok
export EPOCHSIZE=5
export LEARNINGRATE=1e-6
export MAXLEN=1024


CUDA_VISIBLE_DEVICES=0 python -u train_docNLI_2_MNLI_RoBERTa.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'DUC' \
    --seed 42 > log.DUC.docNLI.2.mnli.seed.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u train_docNLI_2_MNLI_RoBERTa.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'Curation' \
    --seed 42 > log.Curation.docNLI.2.mnli.seed.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train_docNLI_2_MNLI_RoBERTa.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'CNNDailyMail' \
    --seed 42 > log.CNNDailyMail.docNLI.2.mnli.seed.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u train_docNLI_2_MNLI_RoBERTa.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'SQUAD' \
    --seed 42 > log.SQUAD.docNLI.2.mnli.seed.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u train_docNLI_2_MNLI_RoBERTa.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'ANLI' \
    --seed 42 > log.ANLI.docNLI.2.mnli.seed.32.txt 2>&1 &
