#!/usr/bin/env bash

learn_bpe() {
    local text_file=$1
    local num_operations=$2
    local codes_file=$3
    local pick_randomly=$4
    local uniform=$5
    local temperature=$6
    local random_bpe_seed=$7
    local count_proportional=$8

    pick_randomly_flag=""
   
    if [ "${pick_randomly}" = "yes" ]; then
        pick_randomly_flag="${pick_randomly_flag} --pick-randomly"

        if [ "${uniform}" = "yes" ] 
        then
            pick_randomly_flag="${pick_randomly_flag} --uniform"
        elif [ "${count_proportional}" = "yes" ]
        then
            pick_randomly_flag="${pick_randomly_flag} --count-proportional"
        fi

        if [ -n "${temperature}" ] 
        then
            pick_randomly_flag="${pick_randomly_flag} --temperature ${temperature}"
        fi
        if [ -n "${random_bpe_seed}" ] 
        then
            pick_randomly_flag="${pick_randomly_flag} --random-seed-for-merges ${random_bpe_seed}"
        fi
    fi

    subword-nmt learn-bpe \
        $pick_randomly_flag \
        -s "${num_operations}" \
        <"${text_file}" \
        >"${codes_file}"
}

apply_bpe() {
    local text_file=$1
    local codes_file=$2
    local out_file=$3

    subword-nmt apply-bpe \
        -c "${codes_file}" \
        <"${text_file}" \
        >"${out_file}"
}

get_vocab() {
    local text_file=$1
    local vocab_file=$2
    subword-nmt get-vocab \
        -i "${text_file}" \
        -o "${vocab_file}"
}

reverse_bpe_segmentation() {
    local text_file=$1
    local out_file=$2
    sed -r 's/(@@ )|(@@ ?$)//g' <"${text_file}" | sacremoses detokenize >"${out_file}"
}

main() {
    local text_file=$1
    local out_file=$2
    local vocab_file=$3
    local codes_file=$4
    local num_operations=$5
    local pick_randomly=$6
    local uniform=$7
    local temperature=$8
    local random_bpe_seed=$9

    check_args
    learn_bpe \
        "${text_file}" \
        "${num_operations}" \
        "${codes_file}" \
        "${pick_randomly}" \
        "${uniform}" \
        "${temperature}" \
        "${random_bpe_seed}"
    apply_bpe \
        "${text_file}" \
        "${codes_file}" \
        "${out_file}"

    get_vocab "${out_file}" "${vocab_file}"
}


get_rank_freq_stats () {
    local vocab_file=$1
    cat -n ${vocab_file} \
        | pee "echo rank token count" "cat" \
        | sed "s/^\s*//g" \
        | sed "s/\s/\t/g" \
        | xsv select -d "\t" 1,3,2 \
        | xsv fmt -t "\t"
}
