#!/bin/bash

data_path=./data/myoglobin.json

local_root=./models
output_path=${local_root}
generation_path=${local_root}/output/myoglobin

mkdir -p ${generation_path}
mkdir -p ${generation_path}/pred_pdbs
mkdir -p ${generation_path}/tgt_pdbs

python3 fairseq_cli/validate.py ${data_path} \
--task geometric_protein_design \
--dataset-impl-source "raw" \
--dataset-impl-target "coor" \
--path ${output_path}/myoglobin.pt \
--batch-size 1 \
--results-path ${generation_path} \
--skip-invalid-size-inputs-valid-test \
--valid-subset test \
--eval-aa-recovery
