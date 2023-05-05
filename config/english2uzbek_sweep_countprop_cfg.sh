#!/usr/bin/env bash

#export randseg_experiment_name=english2uzbek_countprop
export randseg_pick_randomly=yes
export randseg_uniform=no
export randseg_count_proportional=yes
export randseg_root_folder=./experiments
export randseg_raw_data_folder=./data/eng-uzb/til
export randseg_source_language=eng
export randseg_target_language=uzb
export randseg_checkpoints_folder=./eng_uzb_bin/sweep_randseg_ckpt_eng_uzb_countprop_${randseg_num_merges}mops_${randseg_random_seed}_temperature${randseg_temperature}_$(date +%s)
export randseg_binarized_data_folder=./eng_uzb_bin/sweep_randseg_bindata_eng_uzb_countprop_${randseg_num_merges}mops_${randseg_random_seed}_temperature${randseg_temperature}_$(date +%s)
export randseg_model_name=transformer_countprop_${randseg_num_merges}mops_${randseg_random_seed}_temperature${randseg_temperature}

mkdir -p $randseg_checkpoints_folder $randseg_binarized_data_folder

export randseg_max_tokens="12000"
export randseg_max_update="10000"
