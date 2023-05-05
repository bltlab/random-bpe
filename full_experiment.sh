#!/usr/bin/env bash

# Complete experiment sequence
set -eo pipefail

echo "Execution environment:"
env

source scripts/bpe_functions.sh

# Constants
config_file=$1
should_confirm=${2:-"true"}
append_meta=${3:-"false"}

cuda_visible=${CUDA_VISIBLE_DEVICES:-""}

check_these_vars=(
    "randseg_experiment_name"
    "randseg_model_name"
    "randseg_random_seed"
    "randseg_pick_randomly"
    "randseg_num_merges"
    "randseg_root_folder"
    "randseg_raw_data_folder"
    "randseg_binarized_data_folder"
    "randseg_checkpoints_folder"
    "randseg_source_language"
    "randseg_target_language"
    "randseg_should_create_experiment"
	"randseg_should_preprocess"
	"randseg_should_train"
	"randseg_should_evaluate"
)

activate_conda_env () {
    source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh
    conda activate randseg
}

check_deps() {
    echo "❗  Checking dependencies..."
    while read -r dep; do
        test -z "$(which $dep)" &&
            echo "Missing dependency: ${dep}" &&
            exit 1 || echo "Found ${dep} ➡  $(which $dep)"
    done <requirements_external.txt
    echo "✅  Dependencies seem OK"
}

fill_optionals() {
    source config/default_hparams.sh
}

check_env() {
    echo "❗ Checking environment..."

    # First fill optionals with defaults
    fill_optionals

    # Then source the config
    source "${config}"

    # Then check mandatory variables
    missing=false
    for var in "${check_these_vars[@]}"; do
        eval "test -z \$$var" &&
            echo "Missing variable: $var" &&
            missing="true"
    done
    test "$missing" = "true" && exit 1

    echo "✅  Environment seems OK"
}

create_experiment() {
    echo "❗ Creating experiment..."

    prepx create \
        --with-tensorboard --with-supplemental-data \
        --root-folder="${randseg_root_folder}" \
        --experiment-name="${randseg_experiment_name}" \
        --train-name="${randseg_model_name}" \
        --raw-data-folder="${randseg_raw_data_folder}" \
        --checkpoints-folder="${randseg_checkpoints_folder}" \
        --binarized-data-folder="${randseg_binarized_data_folder}" || echo "Error creating experiment folder! Maybe it exists already?"

    echo "✅  Done!"
}

preprocess() {
    echo "❗ Preprocessing..."

    train_folder="${randseg_root_folder}/${randseg_experiment_name}/train/${randseg_model_name}"
    data_folder="${train_folder}/raw_data"
    binarized_data_folder="${train_folder}/binarized_data"
    supplemental_data_folder="${train_folder}/supplemental_data"

    env | rg '^randseg' | tee ${supplemental_data_folder}/relevant_environment_variables.txt

    src=${randseg_source_language}
    tgt=${randseg_target_language}

    # Train BPE/RandBPE using the train seg
    for language in "${src}" "${tgt}"; do
        codes=${supplemental_data_folder}/${language}.bpe.codes

        echo "[${language}] Learning BPE on train..."
        learn_bpe \
            "${data_folder}/train.${language}" \
            "${randseg_num_merges}" \
            "${codes}" \
            "${randseg_pick_randomly}" \
            "${randseg_uniform}" \
            "${randseg_temperature}" \
            "${randseg_random_seed}" \
            "${randseg_count_proportional}"

        for split in "train" "dev" "test"; do
            echo "[${language}, ${split}] Segmenting with BPE..."
            text_file="${data_folder}/${split}.${language}"
            out_file=${supplemental_data_folder}/${split}.bpe.${language}
            apply_bpe \
                "${text_file}" \
                "${codes}" \
                "${out_file}"
        done
        vocab_file=${supplemental_data_folder}/bpe_vocab.${language}
        train_bpe_segmented="${supplemental_data_folder}/train.bpe.${language}"
        get_vocab "${train_bpe_segmented}" "${vocab_file}"
    done

    if [ "${randseg_train_on_dev}" = "yes" ]
    then
        trainpref="${supplemental_data_folder}/dev.bpe"
    else
        trainpref="${supplemental_data_folder}/train.bpe"
    fi

    fairseq-preprocess \
        --source-lang "${src}" --target-lang "${tgt}" \
        --trainpref "${trainpref}" \
        --validpref "${supplemental_data_folder}/dev.bpe" \
        --testpref "${supplemental_data_folder}/test.bpe" \
        --destdir "${randseg_binarized_data_folder}" \
        --workers "${randseg_num_parallel_workers}"

    echo "✅ Done!"

}

train() {
    echo "❗ Starting training..."

    train_folder="${randseg_root_folder}/${randseg_experiment_name}/train/${randseg_model_name}"
    data_folder="${train_folder}/raw_data"
    binarized_data_folder="${train_folder}/binarized_data"
    checkpoints_folder="${train_folder}/checkpoints"
    supplemental_data_folder="${train_folder}/supplemental_data"
    tensorboard_folder="${train_folder}/tensorboard"
    train_log_file="${train_folder}/train.log"
    cpu_gpu_fp16_flag=$(test -z "${cuda_visible}" && echo "--cpu" || echo "--fp16")

    src=${randseg_source_language}
    tgt=${randseg_target_language}

    warmup_updates_flag="--warmup-updates=${randseg_warmup_updates}"

    if [[ "${randseg_lr_scheduler}" == "inverse_sqrt" ]]; then
        warmup_init_lr_flag="--warmup-init-lr=${randseg_warmup_init_lr}"
    else
        warmup_init_lr_flag=""
    fi

    fairseq-train \
        "${binarized_data_folder}" \
        ${cpu_gpu_fp16_flag} ${warmup_updates_flag} ${warmup_init_lr_flag} \
        --save-dir="${checkpoints_folder}" \
        --tensorboard-logdir="${tensorboard_folder}" \
        --source-lang="${src}" \
        --target-lang="${tgt}" \
        --log-format="json" \
        --seed="${randseg_random_seed}" \
        --patience=${randseg_patience} \
        --arch=transformer \
        --attention-dropout="${randseg_p_dropout}" \
        --activation-dropout="${randseg_p_dropout}" \
        --activation-fn="${randseg_activation_fn}" \
        --encoder-embed-dim="${randseg_encoder_embedding_dim}" \
        --encoder-ffn-embed-dim="${randseg_encoder_hidden_size}" \
        --encoder-layers="${randseg_encoder_layers}" \
        --encoder-attention-heads="${randseg_encoder_attention_heads}" \
        --encoder-normalize-before \
        --decoder-embed-dim="${randseg_decoder_embedding_dim}" \
        --decoder-ffn-embed-dim="${randseg_decoder_hidden_size}" \
        --decoder-layers="${randseg_decoder_layers}" \
        --decoder-attention-heads="${randseg_decoder_attention_heads}" \
        --decoder-normalize-before \
        --share-decoder-input-output-embed \
        --criterion="${randseg_criterion}" \
        --label-smoothing="${randseg_label_smoothing}" \
        --optimizer="${randseg_optimizer}" \
        --lr="${randseg_lr}" \
        --lr-scheduler="${randseg_lr_scheduler}" \
        --clip-norm="${randseg_clip_norm}" \
        --max-tokens="${randseg_max_tokens}" \
        --max-update="${randseg_max_update}" \
        --save-interval="${randseg_save_interval}" \
        --validate-interval-updates="${randseg_validate_interval_updates}" \
        --adam-betas '(0.9, 0.98)' --update-freq="${randseg_update_freq}" \
        --no-epoch-checkpoints \
        --max-source-positions=2500 --max-target-positions=2500 \
        --eval-bleu \
        --eval-bleu-remove-bpe \
        --eval-bleu-detok "moses" \
        --skip-invalid-size-inputs-valid-test |
        tee "${train_log_file}"

    echo "✅ Done training..."
    echo "✅ Done!"
}

evaluate() {

    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local split="${1/dev/valid}"

    train_folder="${randseg_root_folder}/${randseg_experiment_name}/train/${randseg_model_name}"
    eval_folder="${randseg_root_folder}/${randseg_experiment_name}/train/${randseg_model_name}/eval"
    data_folder="${eval_folder}/raw_data"
    binarized_data_folder="${train_folder}/binarized_data"
    checkpoints_folder="${train_folder}/checkpoints"
    supplemental_data_folder="${train_folder}/supplemental_data"
    train_log_file="${train_folder}/train.log"
    cpu_gpu_fp16_flag=$(test -z "${cuda_visible}" && echo "--cpu" || echo "--fp16")

    src=${randseg_source_language}
    tgt=${randseg_target_language}

    echo "❗ [${split}] Evaluating..."

    if [[ -z $randseg_beam_size ]]; then
        readonly randseg_beam_size=5
    fi

    CHECKPOINT_FILE="${eval_folder}/checkpoint"
    UNTOUCHED_DETOK_REF="${eval_folder}/raw_data/${split}.detok.${tgt}"

    OUT="${eval_folder}/${split}.out"
    SOURCE_TSV="${eval_folder}/${split}_with_source.tsv"
    GOLD="${eval_folder}/${split}.gold"
    HYPS="${eval_folder}/${split}.hyps"
    SOURCE="${eval_folder}/${split}.source"
    SCORE="${eval_folder}/${split}.eval.score"
    SCORE_TSV="${eval_folder}/${split}_eval_results.tsv"

    # Make raw predictions
    fairseq-generate \
        "${binarized_data_folder}" \
        --source-lang="${src}" \
        --target-lang="${tgt}" \
        --path="${CHECKPOINT_FILE}" \
        --seed="${randseg_random_seed}" \
        --gen-subset="${split}" \
        --beam="${randseg_beam_size}" \
        --max-source-positions=2500 --max-target-positions=2500 \
        --no-progress-bar | tee "${OUT}"

    # Also separate gold/system output/source into separate text files
    # (Sort by index to ensure output is in the same order as plain text data)
    cat "${OUT}" | grep '^T-' | sed "s/^T-//g" | sort -k1 -n | cut -f2 >"${GOLD}"
    cat "${OUT}" | grep '^H-' | sed "s/^H-//g" | sort -k1 -n | cut -f3 >"${HYPS}"
    cat "${OUT}" | grep '^S-' | sed "s/^S-//g" | sort -k1 -n | cut -f2 >"${SOURCE}"

    # Detokenize fairseq output
    SOURCE_ORIG=$SOURCE
    SOURCE=${SOURCE}.detok
    reverse_bpe_segmentation $SOURCE_ORIG $SOURCE

    GOLD_ORIG=$GOLD
    GOLD=${GOLD}.detok
    reverse_bpe_segmentation $GOLD_ORIG $GOLD

    HYPS_ORIG=$HYPS
    HYPS=${HYPS}.detok
    reverse_bpe_segmentation $HYPS_ORIG $HYPS

    paste "${GOLD}" "${HYPS}" "${SOURCE}" >"${SOURCE_TSV}"

    # Sacrebleu scores
    for metric in bleu chrf
    do
        sacrebleu \
            "${GOLD}" \
            -i "${HYPS}" \
            -b -m ${metric} -w 4 \
            > ${SCORE}.${metric}
    done

    echo "✅ Done!"

}

construct_command () {
    local flag=$1
    local command_name=$2
    test "${flag}" = "yes" && echo "${command_name}" || echo "skip"
}

main() {
    local config=$1
    local should_confirm_commands=${2:-"true"}

    activate_conda_env

    confirm_commands_flag=$(
        test "${should_confirm_commands}" = "false" &&
            echo "cat" ||
            echo "fzf --sync --multi"
    )

    # These should always happen
    check_deps
    check_env

    create_experiment_flag=$(construct_command $randseg_should_create_experiment create_experiment)
    preprocess_flag=$(construct_command $randseg_should_preprocess preprocess)
    train_flag=$(construct_command $randseg_should_train train)
    evaluate_flag=$(construct_command $randseg_should_evaluate evaluate)

    echo "$create_experiment_flag" "$preprocess_flag" "$train_flag" "$evaluate_flag" |
        tr " " "\n" |
        ${confirm_commands_flag} |
        while read command; do
            if [ "$command" = "skip" ]; then
                continue
            elif [ "$command" = "evaluate" ]; then
                for split in "dev" "test"; do evaluate $split; done
            else
                $command
            fi
        done
}

main "${config_file}" "${should_confirm}"
