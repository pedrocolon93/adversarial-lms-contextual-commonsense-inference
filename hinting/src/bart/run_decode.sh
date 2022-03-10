for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=$2 /u/pe25171/anaconda3/envs/hinting/bin/python decode.py --model_type ./$1/model/ --original_file ../../data/gold_fixed.jsonl --data_dir ../../data --save_dir ../../data/gen_data --save_filename outputs_$1_e$VARIABLE.jsonl --load_epoch $VARIABLE
done
