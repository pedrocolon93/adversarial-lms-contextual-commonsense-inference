echo "Starting decode t5 scripts..."
#cd bart
screen -dm bash -c "./run_decode.sh base_t5_no_mem_no_hint 0"
screen -dm bash -c "./run_decode.sh base_t5_no_mem_no_hint4 1"
screen -dm bash -c "./run_decode.sh base_t5_no_mem_no_hint2 2"
screen -dm bash -c "./run_decode.sh base_t5_no_mem_no_hint3 3"
screen -dm bash -c "./run_decode.sh base_t5_no_mem_hint 0"
screen -dm bash -c "./run_decode.sh base_t5_no_mem_hint4 1"
screen -dm bash -c "./run_decode.sh base_t5_no_mem_hint2 2"
screen -dm bash -c "./run_decode.sh base_t5_no_mem_hint3 3"
