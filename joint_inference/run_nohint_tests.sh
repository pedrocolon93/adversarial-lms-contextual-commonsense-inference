# Without hints

#'atomic-epoch=2-step=301889.ckpt'
#'conceptnet-glucose-atomic-epoch=1-step=351555.ckpt'
#'concetpnet-glucose-epoch=2-step=225446.ckpt'
#'glucose-epoch=2-step=113816.ckpt'
#'conceptnet-atomic-epoch=2-step=413519.ckpt'
#'concetpnet-epoch=2-step=111857.ckpt'

echo "Starting no hint tests"
BATCH_SIZE=8
# Conceptnet tests
# cn only
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_only/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv  --run_name conceptnet_only_test_conceptnet_nohint --batch_size $BATCH_SIZE --fp16 --cuda
# cn-gl cn-at
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_glucose/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv --run_name conceptnet_glucose_test_conceptnet_nohint --batch_size $BATCH_SIZE --fp16 --cuda
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_atomic/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv --run_name conceptnet_atomic_test_conceptnet_nohint --batch_size $BATCH_SIZE --fp16 --cuda
# cn-gl-at

# Glucose Tests
# gl only
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/glucose_only/final.ckpt" --test_file localdatasets/glucose/glucose_testing_processed.tsv --run_name glucose_only_test_glucose_nohint --batch_size $BATCH_SIZE --fp16 --cuda
# CN + GL
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_glucose/final.ckpt" --test_file localdatasets/glucose/glucose_testing_processed.tsv --run_name glucose_conceptnet_test_glucose_nohint --batch_size $BATCH_SIZE --fp16 --cuda
# cn + gl + at


# Atomic Tests
# at only
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_only/final.ckpt" --test_file localdatasets/atomic2020/contextualized_test.tsv --run_name atomic_only_test_atomic_nohint --batch_size $BATCH_SIZE --fp16 --cuda
# cn+at
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_atomic/final.ckpt" --test_file localdatasets/atomic2020/contextualized_test.tsv --run_name atomic_conceptnet_test_atomic_nohint --batch_size $BATCH_SIZE --fp16 --cuda
# cn+at+gl

echo 'Starting hint only subject+spec+rel'
# Conceptnet tests
# cn only
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_only/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv  --run_name conceptnet_only_test_conceptnet_hintsubjspecrel --hint_subject --hint_relation --hint_random --hint_specificity --batch_size $BATCH_SIZE --fp16 --cuda
# cn-gl cn-at
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_glucose/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv --run_name conceptnet_glucose_test_conceptnet_hintsubjspecrel --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject --hint_relation --hint_random --hint_specificity
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_atomic/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv --run_name conceptnet_atomic_test_conceptnet_hintsubjspecrel --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject --hint_relation --hint_random --hint_specificity
# cn-gl-at

# Glucose Tests
# gl only
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/glucose_only/final.ckpt" --test_file localdatasets/glucose/glucose_testing_processed.tsv --run_name glucose_only_test_glucose_hintsubjspecrel --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject --hint_relation --hint_random --hint_specificity
# CN + GL
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_glucose/final.ckpt" --test_file localdatasets/glucose/glucose_testing_processed.tsv --run_name glucose_conceptnet_test_glucose_hintsubjspecrel --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject --hint_relation --hint_random --hint_specificity
# cn + gl + at


# Atomic Tests
# at only
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_only/final.ckpt" --test_file localdatasets/atomic2020/contextualized_test.tsv --run_name atomic_only_test_atomic_hintsubjspecrel --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject --hint_relation --hint_random --hint_specificity
# cn+at
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_atomic/final.ckpt" --test_file localdatasets/atomic2020/contextualized_test.tsv --run_name atomic_conceptnet_test_atomic_hintsubjspecrel --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject --hint_relation --hint_random --hint_specificity
# cn+at+gl

echo 'Starting hint only subject'
# Conceptnet tests
# cn only
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_only/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv  --run_name conceptnet_only_test_conceptnet_hintsubj --hint_subject  --hint_random  --batch_size $BATCH_SIZE --fp16 --cuda
# cn-gl cn-at
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_glucose/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv --run_name conceptnet_glucose_test_conceptnet_hintsubj --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject  --hint_random
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_atomic/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv --run_name conceptnet_atomic_test_conceptnet_hintsubj --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject  --hint_random
# cn-gl-at

# Glucose Tests
# gl only
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/glucose_only/final.ckpt" --test_file localdatasets/glucose/glucose_testing_processed.tsv --run_name glucose_only_test_glucose_hintsubj --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject  --hint_random
# CN + GL
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_glucose/final.ckpt" --test_file localdatasets/glucose/glucose_testing_processed.tsv --run_name glucose_conceptnet_test_glucose_hintsubj --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject  --hint_random
# cn + gl + at


# Atomic Tests
# at only
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_only/final.ckpt" --test_file localdatasets/atomic2020/contextualized_test.tsv --run_name atomic_only_test_atomic_hintsubj --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject  --hint_random
# cn+at
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/conceptnet_atomic/final.ckpt" --test_file localdatasets/atomic2020/contextualized_test.tsv --run_name atomic_conceptnet_test_atomic_hintsubj --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject  --hint_random
# cn+at+gl
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_glucose_conceptnet/final.ckpt" --test_file localdatasets/atomic2020/contextualized_test.tsv --run_name atomic_conceptnet_glucose_test_atomic_hintsubj --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject  --hint_random
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_glucose_conceptnet/final.ckpt" --test_file localdatasets/glucose/glucose_testing_processed.tsv --run_name glucose_conceptnet_atomic_test_glucose_hintsubj --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject  --hint_random
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_glucose_conceptnet/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv --run_name conceptnet_atomic_glucose_test_conceptnet_hintsubj --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject  --hint_random
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_glucose_conceptnet/final.ckpt" --test_file localdatasets/atomic2020/contextualized_test.tsv --run_name atomic_conceptnet_glucose_test_atomic_hintsubjspecrel --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject --hint_relation --hint_random --hint_specificity
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_glucose_conceptnet/final.ckpt" --test_file localdatasets/glucose/glucose_testing_processed.tsv --run_name glucose_conceptnet_atomic_test_glucose_hintsubjspecrel --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject --hint_relation --hint_random --hint_specificity
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_glucose_conceptnet/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv --run_name conceptnet_atomic_glucose_test_conceptnet_hintsubjspecrel --batch_size $BATCH_SIZE --fp16 --cuda --hint_subject --hint_relation --hint_random --hint_specificity
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_glucose_conceptnet/final.ckpt" --test_file localdatasets/conceptnet/conceptnet_aligned_test.tsv --run_name conceptnet_atomic_glucose_test_conceptnet_nohint --batch_size $BATCH_SIZE --fp16 --cuda
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_glucose_conceptnet/final.ckpt" --test_file localdatasets/glucose/glucose_testing_processed.tsv --run_name glucose_conceptnet_atomic_test_glucose_nohint --batch_size $BATCH_SIZE --fp16 --cuda
/u/pe25171/anaconda3/envs/joint_inference/bin/python train_individual_model.py --do_test --load_checkpoint "final_models/atomic_glucose_conceptnet/final.ckpt" --test_file localdatasets/atomic2020/contextualized_test.tsv --run_name atomic_conceptnet_glucose_test_atomic_nohint --batch_size $BATCH_SIZE --fp16 --cuda


