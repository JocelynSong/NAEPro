#!/bin/bash

data_path=./data/myoglobin.json

local_root=./models
pretrained_model="esm2_t6_8M_UR50D"
output_path=${local_root}/myoglobin
mkdir -p ${output_path}


python3 fairseq_cli/train.py ${data_path} \
--save-dir ${output_path} \
--task geometric_protein_design \
--dataset-impl-source "raw" \
--dataset-impl-target "coor" \
--criterion geometric_protein_loss --encoder-factor 1.0 --decoder-factor 1.0 \
--arch geometric_protein_model_esm \
--encoder-embed-dim 320 \
--decoder-layers 6 \
--pretrained-esm-model ${pretrained_model} \
--egnn-mode "rm-node" \
--dropout 0.3 \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 5e-4 --lr-scheduler inverse_sqrt \
--stop-min-lr '1e-10' --warmup-updates 4000 \
--warmup-init-lr '1e-4' \
--clip-norm 0.0001 \
--ddp-backend legacy_ddp \
--log-format 'simple' --log-interval 10 \
--max-sentences 8 \
--update-freq 1 \
--max-update 1000000 \
--max-epoch 100 \
--valid-subset valid \
--max-sentences-valid 8 \
--validate-interval 1 \
--save-interval 1 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--keep-interval-updates	10 \
--skip-invalid-size-inputs-valid-test
