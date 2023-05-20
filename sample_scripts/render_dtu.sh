method=$1
num_shots=$2
scan=$3
project=dtu${num_shots}-$method


export CUDA_VISIBLE_DEVICES=0,1,2,3
# Please change the checkpoint_dir to the path of your pretrained model
python render.py \
    --gin_configs configs/$method/dtu${num_shots}_${method}.gin \
    --gin_bindings "Config.dtu_scan = 'scan$scan'" \
    --gin_bindings "Config.expname = '$scan-render'" \
    --gin_bindings "Config.checkpoint_dir = 'out/$project/$scan'" \
    --gin_bindings "Config.render_chunk_size = 16384" \
    --gin_bindings "Config.project = '$project'"