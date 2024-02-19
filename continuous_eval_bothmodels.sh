# Repeat the below script continuously to evaluate the model on the test set
# /lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/codec_comparisons/droplast_selenephonebranch_ctc_sp_LRHM_nemocodec_8to15
EXP_DIRS=(
    # "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/mulitlingual_phonemebranch"
    # "/lustre/fs8/portfolios/llmservice/users/shehzeenh/mountdir/experiments/MLS_EXPS"
    # "/lustre/fs8/portfolios/llmservice/users/shehzeenh/mountdir/experiments/MLS_EXPS"
    # "/lustre/fs8/portfolios/llmservice/users/shehzeenh/mountdir/experiments/MLS_EXPS"
    "/lustre/fs8/portfolios/llmservice/users/pneekhara/gitrepos/experiments/t5_feb24/"
    # "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/codec_comparisons"
    # "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/codec_comparisons"
    # "/lustre/fsw/swdl/swdl-langspeech/shehzeenh/mountdir/speechlm/Nemo_Codec_Exps"
    # "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/codec_comparisons"
    # "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/codec_comparisons/"
    # "/lustre/fsw/swdl/swdl-langspeech/shehzeenh/mountdir/speechlm/Nemo_Codec_Exps"
    # "/lustre/fsw/swdl/swdl-langspeech/shehzeenh/mountdir/speechlm/Nemo_Codec_Exps"
    # "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/ctc_experiments"
    # "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/ctc_experiments"
    # "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/ctc_experiments"
    # "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/ctc_experiments"
    # "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/CodeCompare"
    # "/lustre/fsw/swdl/swdl-langspeech/pneekhara/gitrepos/experiments/CodeCompare"
)

SERVER_ADDRESSES=(
    "draco-oci-login-01.draco-oci-iad.nvidia.com"
    # "selene-login"
    # "selene-login"
)

EXP_NAMES=(
    # "oldcode_nemocodec_parallelPattern_0.01"
    # "riva_ctc_spp_LRHM_nemocodec_8to15"
    "newcode_LRHM_decodercontext1e-4"
    # "droplast2_EnglishCheckpoint_ctc_sp_LRHM_nemocodec_8to15"
    # "droplastDebug_EnglishCheckpoint_ctc_sp_LRHM_nemocodec_8to15"
    # "sp_plus_phonemectc_nemo_codec_mls_correctduration"
    # "sp_plus_phonemectc_nemo_codec_mls_correctduration"
    # "sp_plus_phonemectc_nemo_codec_mls_correctduration"
    # "droplast2_EnglishCheckpoint_ctc_sp_LRHM_nemocodec_8to15"
    # "CTC_0.05_All_Data_PS_lr5e-5_Parallel_8k_to_15k"
    # "droplast_selenephonebranch_ctc_sp_LRHM_nemocodec_8to15"
    # "Noctc_All_Data_Phoneme_lr5e-5_Parallel"
    # "Noctc_All_Data_Phoneme_lr5e-5_Parallel"
    # "CTC_0.05_All_Data_Phoneme_lr5e-5"
    # "Noctc_All_Data_Phoneme_lr5e-5_Delay_Parallel"
    # "oldcode_nemocodec_parallel_noctc"
    # "oldcode_dac_speakerid"
    # "oldcode_encodec_speakerid"
    # "oldcode_encodec_speakerid_noctc"
    # "oldcode_ctc_0.05_parallel"
    # "correct_ctc_dac_speakerid_newcode_step0_scale0.1"
)
# TEST_DS="/datap/misc/speechllm_codecdatasets/manifests/speaker_id_manifests/LRH_OnlyHifi_dac_test_speakerid.json"
TEST_DSS=(
    # "/datap/misc/speechllm_codecdatasets/manifests/LibriValOrig_nemo_codec_bw_6.0_phoneme_tts.json"
    "/datap/misc/speechllm_codecdatasets/manifests/LRHM_val_nemo_codec_bw_6.0_phoneme_plus_sentencepiece_tts.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/LRHM_val_nemo_codec_bw_6.0_phoneme_plus_sentencepiece_tts.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/LRHM_val_nemo_codec_bw_6.0_phoneme_plus_sentencepiece_tts.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/challenging_nemo_codec.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/challenging_nemo_codec.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/challenging_encodec_phoneme.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/challenging_nemo_codec_phoneme.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/challenging_nemo_codec_phoneme.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/challenging_nemo_codec_phoneme.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/challenging_nemo_codec_phoneme.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/speaker_id_manifests/LRH_dac_test_speakerid.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/speaker_id_manifests/LRH_encodec_test_speakerid.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/speaker_id_manifests/LRH_encodec_test_speakerid.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/LRH_encodec_test.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/speaker_id_manifests/LRH_dac_test_speakerid.json"
)

CODEC_MODEL_TYPES=(
    "nemo_codec"
    # "nemo_codec"
    # "nemo_codec"
    # "nemo_codec"
    # "nemo_codec"
    # "nemo_codec"
    # "nemo_codec"
    # "dac"
    # "encodec"
    # "encodec"
    # "encodec"
    # "dac"
)

SEQ_PATTERNS=(
    "parallel"
    # "parallel"
    # "parallel"
    # "parallel"
    # "delay_parallel"
    # "delay_parallel"
    # "delay_parallel"
    # "delay_parallel"
    # "delay_parallel"
    # "delay_parallel"
)

FIXED_CHECKPOINTS=(
    # "oldbranch_LRHM_2048_textpretrained_step188958"
    "newcode_LRHM_decodercontext1e-4_step50326"
    # "none"
    # "CTC_0.05_All_Data_PS_lr5e-5_Parallel_8k_to_15k_step165426"
    # "CTC_0.05_All_Data_PS_lr5e-5_Parallel_8k_to_15k_step238256"
    # "droplast2_EnglishCheckpoint_ctc_sp_LRHM_nemocodec_8to15_step197838"
    # "sp_plus_phonemectc_nemo_codec_mls_correctduration_step62230"
    # "CTC_0.05_All_Data_PS_lr5e-5_Parallel_8k_to_15k_step165426"
    # "droplast_selenephonebranch_ctc_sp_LRHM_nemocodec_8to15_step236964"
)

ENGLISH_ONLY_MODEL=(
    # "false"
    "true"
    "true"
    "true"
    "false"
)

ADD_SPECIAL_TOKENS_TO_FIRST=(
    "false"
    "false"
    "false"
    "true"
    "false"
    "true"
)

MAX_SEQ_LEN=(
    "2048"
    # "1536"
    # "1536"
    # "1536"
)

# Repeat whole thing 10 times

for ((j=0; j<1; j++)); do

for ((i=0; i<${#EXP_NAMES[@]}; i++)); do

EXP_DIR=${EXP_DIRS[i]}
EXP_NAME=${EXP_NAMES[i]}
TEST_DS=${TEST_DSS[i]}
CODEC_MODEL_TYPE=${CODEC_MODEL_TYPES[i]}
SEQ_PATTERN=${SEQ_PATTERNS[i]}
FIXED_CHECKPOINT=${FIXED_CHECKPOINTS[i]}
SERVER_ADDRESS=${SERVER_ADDRESSES[i]}

# if english only model is true, then set the language model path to the english only model
if [ "${ENGLISH_ONLY_MODEL[i]}" = "true" ]; then
    # LANGUAGE_MODEL_PATH="/datap/misc/Checkpoints/megatron_t5_220m/tp1_pp1/megatron_t5_expanded_vocab_posemb1536_220m.nemo"
    if "${MAX_SEQ_LEN[i]}" = "1536"; then
        LANGUAGE_MODEL_PATH="/datap/misc/Checkpoints/megatron_t5_expanded_vocab_posemb1536.nemo"
    else
        LANGUAGE_MODEL_PATH="/datap/misc/Checkpoints/megatron_t5_expanded_vocab_posemb_2048.nemo"
    fi
    OVERRIDE_TOKEN_MODEL="null"
    SPEECH_OFFSET=30128
    LM_VCOAB_SIZE=30000
    NUM_SENTINEL_TOKENS=10128
else
    LANGUAGE_MODEL_PATH="/datap/misc/Checkpoints/multilingualT5/megatron_mt5_expanded_vocab_posemb.nemo"
    OVERRIDE_TOKEN_MODEL="/datap/misc/Checkpoints/mt5_tokenizer_w_phones_v2.model"
    SPEECH_OFFSET=250265
    LM_VCOAB_SIZE=250265
    NUM_SENTINEL_TOKENS=9832
fi

# if codec model type is dac, then set the codec fps to 100
if [ "$CODEC_MODEL_TYPE" = "nemo_codec" ]; then
    CODEC_FPS=86
    CODEC_MODEL_CODEBOOKS=8
else
    CODEC_FPS=75
    CODEC_MODEL_CODEBOOKS=8
fi


LOCAL_CKPT_DIR="/datap/misc/temp_checkpoints_new"

# If fixed_checkpoint is none, then copy the checkpoint from selene-login

if [ "$FIXED_CHECKPOINT" = "none" ]; then
    echo "Copying checkpoint from selene-login"
    CHECKPOINT_PATH=$EXP_DIR/$EXP_NAME/p_tuning_squad_t5/checkpoints/*last.ckpt
    
    scp pneekhara@$SERVER_ADDRESS:$CHECKPOINT_PATH $LOCAL_CKPT_DIR

    # Read name of the checkpoint file in LOCAL_CKPT_DIR
    CHECKPOINT_FILE=$(ls $LOCAL_CKPT_DIR | grep "last.ckpt")
    # Take 1st line of the above output and first word of that line
    CHECKPOINT_FILE=$(echo $CHECKPOINT_FILE | head -n 1 | awk '{print $1;}')

    # Get the iter number after "step=" in the checkpoint file name
    CHECKPOINT_ITER=$(echo $CHECKPOINT_FILE | sed -e 's/.*step=\([0-9]*\).*/\1/')

    echo "Checkpoint file: $CHECKPOINT_FILE"
    echo "Checkpoint iter: $CHECKPOINT_ITER"

    # New checkpoint filename is EXP_NAME + CHECKPOINT_ITER .ckpt
    NEW_CHECKPOINT_FILE=$EXP_NAME"_step"$CHECKPOINT_ITER".ckpt"
    NEW_EXP_NAME=$EXP_NAME"_step"$CHECKPOINT_ITER

    echo "New checkpoint file: $NEW_CHECKPOINT_FILE"

    # rename the checkpoint file
    echo "mv $LOCAL_CKPT_DIR/$CHECKPOINT_FILE $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE"

    mv $LOCAL_CKPT_DIR/$CHECKPOINT_FILE $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE
else
    echo "Using Given Checkpoint: $FIXED_CHECKPOINT"
    NEW_CHECKPOINT_FILE=$FIXED_CHECKPOINT".ckpt"
    NEW_EXP_NAME=$FIXED_CHECKPOINT
fi

export HYDRA_FULL_ERROR=1 ;

read -r -d '' cmd <<EOF
python examples/nlp/language_modeling/megatron_t5_speechlm_sft_inference.py \
--config-name=megatron_t5_speechlm_inference.yaml \
name=$NEW_EXP_NAME \
model.data.test_ds='["$TEST_DS"]' \
model.data.g2p.english.phoneme_dict="/home/pneekhara/2023/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt" \
model.data.g2p.english.heteronyms="/home/pneekhara/2023/NeMo/scripts/tts_dataset_files/heteronyms-052722" \
model.data.g2p.spanish.phoneme_dict="/home/pneekhara/2023/NeMo/scripts/tts_dataset_files/es_ES/es_ES_nv230301.dict" \
model.data.g2p.mandarin.phoneme_dict="/home/pneekhara/2023/NeMo/scripts/tts_dataset_files/zh/36finals/ipa_dict_nv23.05.txt" \
model.data.g2p.german.phoneme_dict="/home/pneekhara/2023/NeMo/scripts/tts_dataset_files/de/de_nv240125.dict" \
model.data.g2p.german.heteronyms="/home/pneekhara/2023/NeMo/scripts/tts_dataset_files/de/de_nv230119.heteronym" \
+model.data.add_special_tokens_to_only_first_codebook=${ADD_SPECIAL_TOKENS_TO_FIRST[i]} \
model.data.train_task=all \
+model.freeze_model=False \
model.data.max_seq_length=${MAX_SEQ_LEN[i]} \
model.max_inference_timesteps=2000 \
+model.data.context_duration_min=2.9 \
+model.data.context_duration_max=2.9 \
+model.data.context_pattern=parallel \
model.top_k=80 \
model.temperature=0.9 \
exp_manager.exp_dir=/datap/misc/ContinuousEval/Feb24AfterPresentation \
model.data.sup_data_path=/datap/misc/librittscodec/codec \
model.global_batch_size=2 \
model.micro_batch_size=2 \
+model.num_sentinel_tokens=$NUM_SENTINEL_TOKENS \
model.data.speech_offset=$SPEECH_OFFSET \
+model.lm_vocab_size=$LM_VCOAB_SIZE \
+model.data.num_speech_codebooks=$CODEC_MODEL_CODEBOOKS \
+model.data.codebook_fps=$CODEC_FPS \
+model.codecmodel_type=$CODEC_MODEL_TYPE \
+model.codecmodel_path=/datap/misc/Checkpoints/SpeechCodec.nemo \
trainer.devices=1 \
trainer.precision=bf16 \
model.language_model_path=$LANGUAGE_MODEL_PATH \
+model.override_token_model=$OVERRIDE_TOKEN_MODEL \
model.seq_pattern=$SEQ_PATTERN \
checkpoint_path="$LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE" \
+model.english_only_model=${ENGLISH_ONLY_MODEL[i]} \
+model.context_conditioning="decoder" \
model.speech_head_type=linear
EOF

echo "Running command: $cmd"

eval $cmd

# Remove the checkpoint file
echo "rm $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE"

# rm $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE

# sleep 1m # sleep for 2 minutes

done

done
