#!/usr/bin/env sh

######################################################################
# @author      : adithyare (adithyare@selene-login-01)
# @file        : containers
# @created     : Saturday Aug 13, 2022 15:29:54 PDT
#
# @description : file with holds all the paths required for docker containers
######################################################################


# CONTAINER="/lustre/fsw/llmservice_nemo_speechlm/users/shehzeenh/mountdir/nemo_25-02_rc5.sqsh"
CONTAINER="/lustre/fsw/convai_convaird_nemo-speech/data/containers/nemo_25.04-pytorch2.7-libupdate-251208.sqsh"

CODE="/lustre/fsw/llmservice_nemo_speechlm/users/pneekhara/gitrepos:/gitrepos"

CUSTOM_MOUNTS="/lustre/fsw/convai_convaird_nemo-speech/data/TTS/tts_lhotse_datasets:/data,/lustre/fsw/llmservice_nemo_speechlm/users/shehzeenh/mountdir/:/mountdir/,/lustre/fsw/llmservice_nemo_speechlm/users/rlangman/model_artifacts:/model_artifacts,/lustre/fsw/llmservice_nemo_speechlm/data/TTS/:/data/TTS/"

MOUNTS="--container-mounts=$CODE,$CUSTOM_MOUNTS"