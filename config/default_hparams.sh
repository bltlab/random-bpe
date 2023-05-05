#!/usr/bin/env bash
export randseg_activation_fn="relu"
export randseg_max_tokens="12000"
export randseg_beam_size=5
export randseg_clip_norm="1"
export randseg_criterion="label_smoothed_cross_entropy"
export randseg_decoder_attention_heads="8"
export randseg_decoder_embedding_dim="512"
export randseg_decoder_hidden_size="2048"
export randseg_decoder_layers="6"
export randseg_encoder_attention_heads="8"
export randseg_encoder_embedding_dim="512"
export randseg_encoder_hidden_size="2048"
export randseg_encoder_layers="6"
export randseg_eval_mode="dev"
export randseg_eval_name="transformer"
export randseg_label_smoothing="0.1"
export randseg_langs_file=""
export randseg_lr="0.001"
export randseg_lr_scheduler="inverse_sqrt"
export randseg_max_update="20000"
export randseg_num_parallel_workers=16
export randseg_optimizer="adam"
export randseg_patience="-1" # don't early stop
export randseg_p_dropout="0.1"
export randseg_save_interval="5"
export randseg_validate_interval="1"
export randseg_validate_interval_updates="5000"
export randseg_warmup_init_lr="0.001"
export randseg_warmup_updates="1000"
export randseg_update_freq=16
export randseg_uniform="no"
export randseg_train_on_dev="no"
