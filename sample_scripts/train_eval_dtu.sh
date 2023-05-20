
method=$1
num_shots=$2
scan=$3
project=dtu${num_shots}-$method

# to overwrite the max_steps in the gin config file, use the following line
#   --gin_bindings "Config.freq_reg_end = $max_steps" 

export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py \
    --gin_configs configs/$method/dtu${num_shots}_${method}.gin \
    --gin_bindings "Config.dtu_scan = 'scan$scan'" \
    --gin_bindings "Config.expname = '$scan-train'" \
    --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
    --gin_bindings "Config.project = '$project'" \
    --gin_bindings "Config.render_chunk_size = 16384" 

python eval.py \
    --gin_configs configs/$method/dtu${num_shots}_${method}.gin \
    --gin_bindings "Config.dtu_scan = 'scan$scan'" \
    --gin_bindings "Config.expname = '$scan-eval'" \
    --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
    --gin_bindings "Config.log_img_to_wandb = True" \
    --gin_bindings "Config.project = '$project'"