#!/usr/bin/env bash

#export randseg_experiment_name=english2german_uniformbpe
export randseg_pick_randomly=yes
export randseg_uniform=yes
export randseg_root_folder=./experiments
export randseg_raw_data_folder=./data/dpe-data/eng-deu/tokenized
export randseg_source_language=eng
export randseg_target_language=deu
export randseg_checkpoints_folder=./eng_deu_bin/sweep_randseg_ckpt_eng_deu_uniform_randbpe_${randseg_num_merges}mops_${randseg_random_seed}_temperature${randseg_temperature}_$(date +%s)_thisshouldwork
export randseg_binarized_data_folder=./eng_deu_bin/sweep_randseg_bindata_eng_deu_uniform_randbpe_${randseg_num_merges}mops_${randseg_random_seed}_temperature${randseg_temperature}_$(date +%s)_thisshouldwork
export randseg_model_name=transformer_uniform_randbpe_${randseg_num_merges}mops_${randseg_random_seed}_temperature${randseg_temperature}

mkdir -p $randseg_checkpoints_folder $randseg_binarized_data_folder

export randseg_max_tokens="30000"
export randseg_max_update="10000"
