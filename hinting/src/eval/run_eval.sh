MODEL_NAME=$1
for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
  echo "ITERATION:"$VARIABLE
  for VARIATION in "" 2 3 4
  do
    echo "VARIATION:"$VARIATION
    echo "HINT:"
    /u/pe25171/anaconda3/envs/hinting3/bin/python eval_bleu_2.py --decoded_file "../../data/gen_data/beam_outputs_base_${MODEL_NAME}_no_mem_hint${VARIATION}_e${VARIABLE}.jsonl"
    echo "NOHINT:"
    /u/pe25171/anaconda3/envs/hinting3/bin/python eval_bleu_2.py --decoded_file "../../data/gen_data/beam_outputs_base_${MODEL_NAME}_no_mem_no_hint${VARIATION}_e${VARIABLE}.jsonl"
    echo "DONE"
  done
done