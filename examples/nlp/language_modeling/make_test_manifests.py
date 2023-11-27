import json

train_manifest_path = "/datap/misc/manifests/manifests/libritts/train_clean_300_speechlm_ttstasks.json"
val_manifest_path = "/datap/misc/manifests/manifests/libritts/val_clean_300_speechlm_ttstasks.json"

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


import ipdb; ipdb.set_trace()
