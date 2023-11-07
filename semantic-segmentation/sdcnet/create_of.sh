#!/bin/bash

python Cityscapes_val_optical_flow_scale512.py --pretrained ../pretrained_models/sdc_cityscapes_vrec.pth.tar \
 --flownet2_checkpoint ../pretrained_models/FlowNet2_checkpoint.pth.tar --source_dir \
 /share_chairilg/data/cityscapes/ --target_dir Cityscapes_train_optical_flow_scale1024_prev --vis --resize 1 --gap 1


# python Estimated_optical_flow_SynthiaSeq_train.py --pretrained ../pretrained_models/sdc_cityscapes_vrec.pth.tar \
#  --flownet2_checkpoint ../pretrained_models/FlowNet2_checkpoint.pth.tar --source_dir \
#  /share_chairilg/data/SYNTHIA-SEQS-04-DAWN/RGB/Stereo_Left/Omni_F/ --target_dir Estimated_optical_flow_SynthiaSeq_train \
#  --vis --resize 1 --gap 2