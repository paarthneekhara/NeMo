import json
import re

def process_dialogue(dialogue):
    dialogue = dialogue.replace("User:", "[SPK-BWL-B-M]")
    dialogue = dialogue.replace("Assistant:", "[SPK-BWL-B-F]")
    dialogue = dialogue.replace("\n\n", " ")
    dialogue = dialogue.replace("\n", " ")

    return dialogue

def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            records.append(json.loads(line))
    return records

def write_manifest(manifest_path, records):
    with open(manifest_path, 'w') as f:
        file_str = ""
        for record in records:
            file_str += json.dumps(record) + "\n"
        file_str = file_str.strip()
        f.write(file_str)
        print("Wrote {} records to: {}".format(len(records), manifest_path))

manifest_path = "/home/pneekhara/2023/SimpleT5NeMo/multiturn_speech2speechdata.json"
with open(manifest_path) as f:
    data = json.load(f)

dialogues = []
for record in data:
    dialogue = record['question'] + " " + record['answer']
    dialogues.append(dialogue)

unique_dialogues = []
max_num_turns = 0
overall_max_turn_length = 0
for didx, dialogue in enumerate(dialogues[:-1]):
    if dialogue in dialogues[didx+1]:
        continue
    dialogue_formatted = process_dialogue(dialogue)
    split_dialogues = [d.strip() for d in re.split(r'(?=\[SPK-[^\]]+\])', dialogue_formatted) if d.strip()]
    num_turns = len(split_dialogues)
    max_num_turns = max(max_num_turns, num_turns)
    max_turn_length = max([len(d) for d in split_dialogues])
    overall_max_turn_length = max(overall_max_turn_length, max_turn_length)
    if max_turn_length < 300:
        unique_dialogues.append((didx, dialogue_formatted, split_dialogues, max_turn_length, num_turns))


print(f"Max number of turns: {max_num_turns}")
print(f"Max turn length: {overall_max_turn_length}")
print(f"Number of unique dialogues: {len(unique_dialogues)}")
print(f"Number of dialogues: {len(dialogues)}")

female_audio = "/datap/misc/Blackwell-Demo/t5tts_audios/S7_A4_SC8_singleturntarget_35_channel_1.wav"
male_audio = "/datap/misc/Blackwell-Demo/t5tts_audios/S2_A9_SC1_singleturntarget_99_channel_2.wav"

for turn_idx in range(max_num_turns):
    turn_records = []
    for d in unique_dialogues:
        if turn_idx < len(d[2]):
            speaker_audio = female_audio
            speaker = "[SPK-BWL-B-F]"
            if "[SPK-BWL-B-M]" in d[2][turn_idx]:
                speaker_audio = male_audio
                speaker = "[SPK-BWL-B-M]"

            record = {
                "dialogue_id": d[0],
                "turn_id": turn_idx,
                "full_dialogue": d[1],
                "text": d[2][turn_idx],
                "context_text": "MIXED SPEECH TTS",
                "audio_filepath": speaker_audio,
                "speaker": speaker,
                "duration": 10.0
            }
            if turn_idx > 0:
                del record['context_text']
                record['context_audio_filepath'] = "/generated_audio_dir/dialogueturn_{}_{}.wav".format(d[0], turn_idx-1)
                record['context_audio_duration'] = 5.0

            turn_records.append(record)
    
    write_manifest(f"/home/pneekhara/2023/SimpleT5NeMo/multiturn_speech2speechdata_manifests/turn_{turn_idx}.json", turn_records)

