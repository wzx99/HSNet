#!/bin/bash/

GPU_NUM=4
YAML_FILE='experiments/seg_detector/totaltext_resnet50_deform_hierarchical.yaml'
RESUME='pretrain_model_path'
LOG_DIR='log_path'


# python train.py ${YAML_FILE} --log_dir ${LOG_DIR} --resume ${RESUME}
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} train.py ${YAML_FILE} --log_dir ${LOG_DIR} -d --num_gpus ${GPU_NUM} --resume ${RESUME}

python eval.py ${YAML_FILE} --log_dir ${LOG_DIR} --polygon --box_thresh 0.7 --thresh 0.25 --result_dir ${LOG_DIR}/results/ --resume ${LOG_DIR}/model

