import json
import os
import shutil
import torch

train_manifest_path = "/datap/misc/manifests/manifests/libritts/train_clean_300_speechlm_ttstasks.json"
val_manifest_path = "/datap/misc/manifests/manifests/libritts/val_clean_300_speechlm_ttstasks.json"
val_manifest_path_filtered = "/datap/misc/manifests/manifests/libritts/val_clean_300_speechlm_ttstasks_filtered.json"
val_phoneme_manifest = "/datap/misc/ptts_val_correct.json"

seen_speakers = {}
with open(train_manifest_path) as f:
    all_lines = f.read().split("\n")
    train_records = []
    for line in all_lines:
        if len(line) > 0:
            record = json.loads(line)
            seen_speakers[record["speaker_id"]] = True
            train_records.append(record)

with open(val_manifest_path) as f:
    all_lines = f.read().split("\n")
    val_seen_records = []
    val_unseen_records = []
    for line in all_lines:
        if len(line) > 0:
            record = json.loads(line)
            if record["speaker_id"] in seen_speakers:
                val_seen_records.append(record)
            else:
                val_unseen_records.append(record)


val_phoneme_records = []
val_tts_records = []
for val_record in val_seen_records:
    context_audio_path = val_record['context']
    context_id = context_audio_path.split("train-clean-360/")[-1]
    context_id = context_id.replace("/", "_")
    context_id = context_id.replace(".wav", ".pt")

    answer_audio_path = val_record['answer']
    answer_id = answer_audio_path.split("train-clean-360/")[-1]
    answer_id = answer_id.replace("/", "_")
    answer_id = answer_id.replace(".wav", ".pt")

    answer_path = "/datap/misc/multitask_audiocodec/target_codes_{}".format(answer_id)
    
    if os.path.exists(answer_path):
        print("answer path exists: {}".format(answer_path))
    else:
        continue

    context_path = "/datap/misc/multitask_audiocodec/target_codes_{}".format(context_id)
    if os.path.exists(context_path):
        # print("Path exists: {}".format(context_path))
        codec_data = torch.load(context_path)
    else:
        print("Path does not exist: {}".format(context_path))
        sup_codec_path = "/datap/misc/librittscodec/codec/codec/{}".format(context_id)
        assert os.path.exists(sup_codec_path)
        # copy sup_codec_path to context_path
        print("Saving {} to {}".format(sup_codec_path, context_path))
        sup_data = torch.load(sup_codec_path)
        # convert to int16
        sup_data = sup_data.to(torch.int16)
        torch.save(sup_data, context_path)
    
    if val_record['context_duration'] > 1.5 and val_record['answer_duration'] > 1.5 and val_record['answer_duration'] < 15.0:
        val_phoneme_records.append({
            'audio_filepath' : val_record['audio_filepath'],
            'text' : val_record['text'],
            'question'  : val_record['question'].replace("Text to speech this ", "Phoneme TTS "),
            "answer" : answer_path,
            "context" : context_path,
            "question_type" : "TEXT",
            "answer_type" : "AUDIOCODEC",
            "context_type" : "REFSPEAKERCODEC",
            "context_duration" : val_record['context_duration'],
            "answer_duration" : val_record['answer_duration'],
            "taskname"  : "squad",
        })
        val_tts_records.append(val_record)

with open(val_phoneme_manifest, "w") as f:
    for record in val_phoneme_records:
        f.write(json.dumps(record) + "\n")
    
    print(val_phoneme_manifest)

with open(val_manifest_path_filtered, "w") as f:
    for record in val_tts_records:
        f.write(json.dumps(record) + "\n")
    
    print(val_manifest_path_filtered)