#!/usr/bin/env bash

#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --gres=gpu:V100:10
#SBATCH --output=%x-%j.out

env

test -z "${randseg_cfg_file}" && exit 1
test -z "${randseg_hparams_folder}" && exit 1

run_single_exp () {
    local gpu_idx=$1
    shift
    local hparams=$1
    shift

    export randseg_random_seed=$(echo $hparams | cut -f1 -d' ')
    export randseg_num_merges=$(echo $hparams | cut -f2 -d' ')
    export randseg_temperature=$(echo $hparams | cut -f3 -d' ')

    echo "seed: ${randseg_random_seed}"
    echo "mops: ${randseg_num_merges}"
    echo "temp: ${randseg_temperature}"

    CUDA_VISIBLE_DEVICES=${gpu_idx} ./full_experiment.sh "${randseg_cfg_file}" false false

}

export -f run_single_exp


gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," " ")
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
taskid=${SLURM_ARRAY_TASK_ID}

# Set env var `randseg_hparams_folder` to a folder
# where each SLURM worker can pick tasks from TSV
hparams_file=${randseg_hparams_folder}/worker${taskid}.tsv

echo "Number of GPUs: $num_gpus"

parallel --jobs $num_gpus --link "run_single_exp {1} {2}" ::: ${gpus} :::: $hparams_file
