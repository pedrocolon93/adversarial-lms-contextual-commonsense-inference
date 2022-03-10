SEED=42
BATCH_SIZE=50
PYTHON_BIN=/home/pedro/anaconda3/envs/joint_inference/bin/python
# Single Models
#/home/pedro/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_train --fp16 --seed $SEED --batch_size  $BATCH_SIZE  --log_every_n_steps 50 --cuda  --hint_subject --hint_object --hint_random --hint_relation --hint_specificity --train_file localdatasets/conceptnet/conceptnet_aligned_train600k.tsv --run_name conceptnet_only
#/home/pedro/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_train --fp16 --seed $SEED --batch_size  $BATCH_SIZE  --log_every_n_steps 50 --cuda  --hint_subject --hint_object --hint_random --hint_relation --hint_specificity --train_file localdatasets/atomic2020/contextualized_train.tsv --run_name atomic_only
#$PYTHON_BIN train_individual_model.py --do_train --fp16 --seed $SEED --batch_size  $BATCH_SIZE  --log_every_n_steps 50 --cuda  --hint_subject --hint_object --hint_random --hint_relation --hint_specificity --train_file localdatasets/glucose/glucose_training_processed.tsv --run_name glucose_only
# Combos of 2
$PYTHON_BIN train_individual_model.py --do_train --fp16 --seed $SEED --batch_size  $BATCH_SIZE  --log_every_n_steps 50 --cuda  --hint_subject --hint_object --hint_random --hint_relation --hint_specificity --train_file localdatasets/conceptnet/conceptnet_aligned_train600k.tsv,localdatasets/glucose/glucose_training_processed.tsv --run_name conceptnet_glucose
$PYTHON_BIN train_individual_model.py --do_train --fp16 --seed $SEED --batch_size  $BATCH_SIZE  --log_every_n_steps 50 --cuda  --hint_subject --hint_object --hint_random --hint_relation --hint_specificity --train_file localdatasets/atomic2020/contextualized_train.tsv,localdatasets/conceptnet/conceptnet_aligned_train600k.tsv --run_name conceptnet_atomic
$PYTHON_BIN train_individual_model.py --do_train --fp16 --seed $SEED --batch_size  $BATCH_SIZE  --log_every_n_steps 50 --cuda  --hint_subject --hint_object --hint_random --hint_relation --hint_specificity --train_file localdatasets/atomic2020/contextualized_train.tsv,localdatasets/glucose/glucose_training_processed.tsv --run_name atomic_glucose
# All 3
$PYTHON_BIN train_individual_model.py --do_train --fp16 --seed $SEED --batch_size  $BATCH_SIZE  --log_every_n_steps 50 --cuda  --hint_subject --hint_object --hint_random --hint_relation --hint_specificity --train_file localdatasets/atomic2020/contextualized_train.tsv,localdatasets/glucose/glucose_training_processed.tsv,localdatasets/conceptnet/conceptnet_aligned_train600k.tsv --run_name atomic_glucose_conceptnet
