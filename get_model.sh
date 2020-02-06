MODEL_DIR=/home/kevin_leo_mcmanus/vipir/resnet_train/model
CHECKPOINT_NUMBER=345
CONFIG_PATH=/home/kevin_leo_mcmanus/vipir/resnet_train/data/faster_rcnn_resnet101_vipir.config
OUTPUT_DIR=/home/kevin_leo_mcmanus/vipir/vipIr_resnet
TENSORFLOW=/home/kevin_leo_mcmanus/vipir/TensorFlow/models/research
TENSORFLOW_OD=${TENSORFLOW}/object_detection

export PYTHONPATH=${TENSORFLOW}:${TENSORFLOW_OD}:${TENSORFLOW}/slim

cd ${MODEL_DIR}

python ${TENSORFLOW_OD}/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${CONFIG_PATH} \
    --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory $OUTPUT_DIR