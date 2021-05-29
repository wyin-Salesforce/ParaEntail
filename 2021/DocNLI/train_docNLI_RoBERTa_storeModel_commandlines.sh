export BATCHSIZE=4 #32 is also ok
export EPOCHSIZE=10
export LEARNINGRATE=1e-6
export MAXLEN=512

CUDA_VISIBLE_DEVICES=1 python -u train_docNLI_RoBERTa_storeModel.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'ANLI CNNDailyMail' \
    --seed 42 > log.ANLI.and.CNNDailyMail.docNLI.store.model.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train_docNLI_RoBERTa_storeModel.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'ANLI CNNDailyMail DUC' \
    --seed 42 > log.ANLI.and.CNNDailyMail.and.DUC.docNLI.store.model.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u train_docNLI_RoBERTa_storeModel.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'ANLI CNNDailyMail DUC Curation' \
    --seed 42 > log.ANLI.and.CNNDailyMail.and.DUC.and.Curation.docNLI.store.model.txt 2>&1 &

CUDA_VISIBLE_DEVICES=6 python -u train_docNLI_RoBERTa_storeModel.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --data_label 'ANLI CNNDailyMail DUC Curation SQUAD' \
    --seed 42 > log.ANLI.and.CNNDailyMail.and.DUC.and.Curation.and.SQUAD.docNLI.store.model.txt 2>&1 &
