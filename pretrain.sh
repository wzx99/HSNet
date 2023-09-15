#!/bin/bash/
  
GPU_NUM=4
YAML_FILE='experiments/seg_detector/synth_resnet50_deform_hierarchical.yaml'
LOG_DIR='log_path'

# python train.py ${YAML_FILE} --log_dir ${LOG_DIR} --no-validate
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} train.py ${YAML_FILE} --log_dir ${LOG_DIR} --no-validate -d --num_gpus ${GPU_NUM}
