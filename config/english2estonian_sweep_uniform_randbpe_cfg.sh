#!/usr/eng_est_bin/env bash

#export randseg_experiment_name=english2estonian_uniform_randbpe
export randseg_pick_randomly=yes
export randseg_uniform=yes
export randseg_root_folder=./experiments
export randseg_raw_data_folder=./data/dpe-data/eng-est/tokenized
export randseg_source_language=eng
export randseg_target_language=est
export randseg_checkpoints_folder=./eng_est_bin/sweep_randseg_ckpt_eng_est_randbpe_${randseg_num_merges}mops_${randseg_random_seed}_temperature${randseg_temperature}_$(date +%s)
export randseg_binarized_data_folder=./eng_est_bin/sweep_randseg_bindata_eng_est_randbpe_${randseg_num_merges}mops_${randseg_random_seed}_temperature${randseg_temperature}_$(date +%s)
export randseg_model_name=transformer_randbpe_${randseg_num_merges}mops_${randseg_random_seed}_temperature${randseg_temperature}

mkdir -p $randseg_checkpoints_folder $randseg_binarized_data_folder

export randseg_max_tokens="36000"
export randseg_max_update="10000"
