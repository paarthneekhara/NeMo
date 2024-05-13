import json
import random
from pathlib import Path
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir
import os
import numpy as np
import tqdm
import torch

# emb_out_dir = "/datap/misc/speaker_embeddings_extracted"
# manifest_with_similarities = "/datap/misc/manifest_train_with3sec_ref_tts_speechllm_with_embs.json"
# codec_manifest = "/datap/misc/speechllm_codecdatasets/manifests/hifitts_nemo_codec_bw_6.0_train_sentencepiece_tts.json"

# emb_out_dir = "/datap/misc/speaker_embeddings_extracted_libritts"
# manifest_with_similarities = "/datap/misc/manifests/manifests/libritts/train_clean_300_speechlm_ttstasks_with_embs.json"
codec_manifest = "/datap/misc/speechllm_codecdatasets/manifests/RivattsAllLanguagesUpdated_train_nemo_codec_bw_6.0_phoneme_tts.json"

emb_out_dir = "/datap/misc/speaker_embeddings_extracted_riva"

out_manifest = codec_manifest.replace(".json", "_highsimilarity2.json")

def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            record = json.loads(line)
            record['audio_filepath'] = record['audio_filepath'].replace("/Data/LibriTTS/train-clean-360/", "/mnt/drive1/datasets/LibriTTS/train-clean-360/")
            records.append(record)
    return records

def write_manifest(records, manifest_path):
    with open(manifest_path, "w") as f:
        for ridx, record in enumerate(records):
            if ridx < len(records) - 1:
                f.write(json.dumps(record) + "\n")
            else:
                f.write(json.dumps(record))
    print("Written to ", manifest_path)

def find_fileid_from_path(path, base_dir):
    rel_audio_path = Path(path).relative_to(base_dir).with_suffix("")
    rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")
    return rel_audio_path_as_text_id

def find_best_candidate(target_record, candidate_records, base_dir):
    target_fileid = find_fileid_from_path(target_record['audio_filepath'], base_dir)
    target_embedding_fp = os.path.join(emb_out_dir, "{}.npy".format(target_fileid))
    if not os.path.exists(target_embedding_fp):
        return None, None
    target_embedding = np.load(target_embedding_fp)
    target_embedding_torch = torch.from_numpy(target_embedding).cuda()
    
    similarity_and_records = []
    candidate_embeddings_torch = []
    filtered_candidate_records = []
    for candidate_record in candidate_records:
        candidate_fileid = find_fileid_from_path(candidate_record['audio_filepath'], base_dir)
        candidate_embedding_fp = os.path.join(emb_out_dir, "{}.npy".format(candidate_fileid))
        if os.path.exists(candidate_embedding_fp):
            candidate_embedding = np.load(candidate_embedding_fp)
            candidate_embedding_torch = torch.from_numpy(candidate_embedding).cuda()
            candidate_embeddings_torch.append(candidate_embedding_torch)
            filtered_candidate_records.append(candidate_record)
        # cossim = np.dot(target_embedding, candidate_embedding) / (np.linalg.norm(target_embedding) * np.linalg.norm(candidate_embedding))
        # similarity_and_records.append((-cossim, candidate_record))
    if len(candidate_embeddings_torch) == 0:
        return None, None
    
    candidate_embeddings_torch = torch.stack(candidate_embeddings_torch)
    target_embedding_torch = target_embedding_torch[None]
    with torch.no_grad():
        cossims = torch.nn.functional.cosine_similarity(target_embedding_torch, candidate_embeddings_torch)
    for cidx, candidate_record in enumerate(filtered_candidate_records):
        similarity_and_records.append((-cossims[cidx].item(), candidate_record))
    
    # Sort by first value
    similarity_and_records.sort(key=lambda x: x[0])
    if abs(similarity_and_records[0][0]) > 0.6:
        return abs(similarity_and_records[0][0]), similarity_and_records[0][1]
    else:
        return None, None
    
def get_speaker_info_from_record(record):
    if "/train-clean-360" in record["audio_filepath"]:
        language = "en"
        speaker_name = record["audio_filepath"].split("train-clean-360/")[1].split("/")[0]
        dataset_name = "Libri360"
    elif "RIVA-TTS" in record["audio_filepath"]:
        language = record["audio_filepath"].split("RIVA-TTS/")[1].split("/")[0]
        if language == "en":
            speaker_name = record["audio_filepath"].split("RIVA-TTS/en/")[1].split("/")[0]
        else:
            if language in ["es", "fr"]:
                speaker_name = record["audio_filepath"].split("RIVA-TTS/{}/".format(language))[1].split("/")[1]
            else:
                speaker_name = record["audio_filepath"].split("RIVA-TTS/{}/".format(language))[1].split("/")[0]
        dataset_name = "Riva"
    elif "HiFiTTS" in record["audio_filepath"]:
        language = "en"
        speaker_name = record["audio_filepath"].split("hi_fi_tts_v0/audio/")[1].split("/")[0]
        dataset_name = "HiFiTTS"
    elif "/data/filtered_24khz/audio_24khz/" in record["audio_filepath"]:
        language = "en"
        speaker_name = record["audio_filepath"].split("/data/filtered_24khz/audio_24khz/")[1].split("/")[0]
        dataset_name = "MLS"
        
    speaker_full_name = "| Language:{} Dataset:{} Speaker:{} |".format(language, dataset_name, speaker_name)
    return speaker_full_name
        

# with_similarity_records = read_manifest(manifest_with_similarities)
codec_records = read_manifest(codec_manifest)
base_dir = get_base_dir([record["audio_filepath"] for record in codec_records])

speakerwise_codec_records = {}
for record in codec_records:
    record['speaker'] = get_speaker_info_from_record(record)
    if record['speaker'] not in speakerwise_codec_records:
        speakerwise_codec_records[record['speaker']] = []
    if record['answer_duration'] > 5.0:
        speakerwise_codec_records[record['speaker']].append(record)


high_similarity_records = []
for ridx, record in enumerate(tqdm.tqdm(codec_records)):
    speaker = record['speaker']
    if len(speakerwise_codec_records[speaker]) > 100:
        candidate_contexts = random.sample(speakerwise_codec_records[speaker], 100)
    else:
        candidate_contexts = speakerwise_codec_records[speaker]
    candidate_contexts = [c for c in candidate_contexts if c['audio_filepath'] != record['audio_filepath'] ]
    if len(candidate_contexts) == 0:
        print("No candidates found for speaker {}, record {}".format(speaker, ridx) )
    
    best_similarity, best_candidate_record = find_best_candidate(record, candidate_contexts, base_dir)
    if best_similarity is not None:
        record['context'] = best_candidate_record['answer']
        record['context_duration'] = best_candidate_record['answer_duration']
        record['similarity'] = best_similarity
        high_similarity_records.append(record)
    else:
        print("No high similarity found for record: ", ridx, len(codec_records))
        print("Len high similarity records: ", len(high_similarity_records))

print("Len high similarity records: ", len(high_similarity_records))
print("Orig len", len(codec_records))
write_manifest(high_similarity_records, out_manifest)