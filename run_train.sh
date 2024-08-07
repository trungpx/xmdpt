pip install -e .
export OPENAI_LOGDIR=logs/XMDPT_S_2
NUM_GPUS=1

MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 2 --model XMDPT_S_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 32"
DATA_PATH=./datasets/deepfashion/

python -m torch.distributed.launch \
    --master_port 29514 \
    --nproc_per_node=$NUM_GPUS scripts/image_train.py \
    --data_dir $DATA_PATH \
    --work_dir $OPENAI_LOGDIR \
    --lr 1e-4 \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS \
