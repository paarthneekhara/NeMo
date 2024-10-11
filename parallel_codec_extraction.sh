#!/bin/bash

# Eg. usage
# ./parallel_codec_extraction.sh \
#   /datap/misc/speechllm_codecdatasets/manifests/libri100_remaining.json \
#   16 \
#   /datap/misc/checkpoints/AudioCodec_21Hz-1k-codes.nemo \
#   /datap/misc/speechllm_codecdatasets/ \
#   LibriTTS10021Hz1k \
#   --codec_bw 6.0 \
#   --codec_model nemo_codec211k \
#   --use_context_as_same_speaker_audio \
#   --save_only_tts_records

# Check if the required arguments are provided
if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <manifest_path> <batch_size> <codec_model_path> <out_dir> <dataset_name_prefix> [additional python script arguments...]"
  exit 1
fi

# Parse the input arguments
MANIFEST_PATH=$1
BATCH_SIZE=$2
CODEC_MODEL_PATH=$3
OUT_DIR=$4
DATASET_NAME_PREFIX=$5

# Capture any additional arguments to be passed to the Python script
ADDITIONAL_ARGS="${@:6}"

# Number of parts (should match the number of GPUs you want to use)
N=8

# Split the manifest file into N parts
split -n l/$N --numeric-suffixes=1 --additional-suffix=.json $MANIFEST_PATH ${MANIFEST_PATH%.json}_${DATASET_NAME_PREFIX}_part

# Loop to print the data preparation command for each part
for i in $(seq 1 $N)
do
  GPU_ID=$((i-1))
  MANIFEST_PART="${MANIFEST_PATH%.json}_${DATASET_NAME_PREFIX}_part$(printf "%02d" $i).json"
  
  # Print the command instead of running it
  CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/speechllm_multitask_dataprep.py \
    --manifest_paths $MANIFEST_PART \
    --batch_size $BATCH_SIZE \
    --split_num $i \
    --codec_model_path $CODEC_MODEL_PATH \
    --out_dir $OUT_DIR \
    --dataset_name $DATASET_NAME_PREFIX \
    $ADDITIONAL_ARGS &
  
  # Sleep for 10 seconds befor launching next job.
  sleep 10
done

wait

echo "Command printing completed for $N GPUs!"