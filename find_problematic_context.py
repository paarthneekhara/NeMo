import json
import os
import torch
from nemo.collections.tts.models import AudioCodecModel
import torchaudio
from torchaudio.transforms import Resample
import nemo.collections.asr as nemo_asr
import random
import matplotlib.pyplot as plt
import numpy as np

def plot_similarity_historgram(records):
    similarities = [r['similarity'] for r in records]
    plt.hist(similarities, bins=20)
    plt.xlabel('Speaker Similarity')
    plt.ylabel('Count')
    plt.title('Speaker Similarity Histogram')
    # Save the figure
    plt.savefig('speaker_similarity_histogram.png')
    # Clear the plot
    plt.clf()
    plt.close()
    

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

resampler = Resample(orig_freq=22050, new_freq=16000).cuda()

def decode_codes_and_find_similarity(codec_model, codec_fps):
    with torch.no_grad():
        max_codec_len = 0
        codec_list = []
        codec_lens = []
        for codec_fp in codec_fps:
            codec = torch.load(codec_fp)
            codec_len = torch.Tensor([codec.shape[1]]).long().cuda()
            codec_lens.append(codec_len)
            if codec.shape[1] > max_codec_len:
                max_codec_len = codec.shape[1]
            codec_list.append(codec)
        
        codec_lens = torch.stack(codec_lens).long().cuda()
        codecs = torch.zeros(len(codec_list), codec_list[0].shape[0], max_codec_len).cuda()
        for i, codec in enumerate(codec_list):
            codecs[i, :, :codec.shape[1]] = codec
        codecs = codecs.long()
        # import ipdb; ipdb.set_trace()
        codec_decoded_audios, _ = codec_model.decode(tokens=codecs, tokens_len=codec_lens[:,0])

        audios_16 = []
        audio_16_lens = []
        max_16_len = 0
        for idx in range(len(codec_decoded_audios)):
            codec_decoded_audio = codec_decoded_audios[idx]
            codec_decoded_audio = codec_decoded_audio[:codec_lens[idx][0].item() * int(codec_model_downsampling_factor)]
            
            # Resample from 22050 to 16000
            codec_decoded_audio_16 = resampler(codec_decoded_audio)
            audios_16.append(codec_decoded_audio_16)
            audio_16_lens.append(codec_decoded_audio_16.shape[0])
            if codec_decoded_audio_16.shape[0] > max_16_len:
                max_16_len = codec_decoded_audio_16.shape[0]

            # codec_decoded_audio_path = os.path.join("/datap/misc/decodingtesting/testing_{}.wav".format(idx))
            # torchaudio.save(codec_decoded_audio_path, codec_decoded_audio[None].cpu(), codec_model_sample_rate)

            # codec_decoded_audio_path = os.path.join("/datap/misc/decodingtesting/testing_{}_16k.wav".format(idx))
            # torchaudio.save(codec_decoded_audio_path, codec_decoded_audio_16[None].cpu(), 16000)

        audio_16_batch = torch.zeros(len(audios_16), max_16_len).cuda()
        for idx in range(len(audios_16)):
            audio_16_batch[idx, :audio_16_lens[idx]] = audios_16[idx]
        
        audio_signal_len_16 = torch.Tensor(audio_16_lens).long().cuda()
        # import ipdb; ipdb.set_trace()
        nemo_sv_model.eval()
        # nemo_sv_model.freeze()
        _, embs = nemo_sv_model.forward(input_signal=audio_16_batch, input_signal_length=audio_signal_len_16)
        # Find cosine similarity between embs
        similarities = []
        for i in range(embs.shape[0]-1):
            emb1 = embs[i]
            emb2 = embs[-1]
            similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
            similarities.append(similarity)

        return sum(similarities) / len(similarities)


manifest_path = "/datap/misc/speechllm_codecdatasets/manifests/LRHM_train_nemo_codec_bw_6.0_phoneme_tts.json"

records = read_manifest(manifest_path)

codec_model = AudioCodecModel.restore_from("/datap/misc/Checkpoints/SpeechCodec.nemo")
codec_model.to('cuda')
codec_model.eval()
codec_model_sample_rate = 22050
codec_model_downsampling_factor = 256.0

nemo_sv_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')

similarity_wise_sorted = []
with_similarity_records = []
low_similarity_records = []
high_similarity_records = []

for ridx, record in enumerate(records):
    print("{} out of {}".format(ridx, len(records)))
    context_fps = record["context"].split(";")
    answer_fp = record["answer"]
    try:
        similarity = decode_codes_and_find_similarity(codec_model, context_fps + [answer_fp])
    except:
        print("Error in record: ", record)
        continue

    record["similarity"] = round(similarity, 4)
    if similarity < 0.35:
        low_similarity_records.append(record)
    elif similarity > 0.7:
        high_similarity_records.append(record)
    
    with_similarity_records.append(record)
    print(ridx, record["similarity"])
    similarity_wise_sorted.append((similarity, record))
    if (ridx+1) % 100 == 0:
        updated_manifest_path = manifest_path.replace(".json", "_withsimilarity_inprogress.json")
        write_manifest(updated_manifest_path, with_similarity_records)
        plot_similarity_historgram(with_similarity_records)

        low_similarity_manifest_path = manifest_path.replace(".json", "_low_similarity.json")
        write_manifest(low_similarity_manifest_path, low_similarity_records)

        high_similarity_manifest_path = manifest_path.replace(".json", "_high_similarity.json")
        write_manifest(high_similarity_manifest_path, high_similarity_records)


# speakerwise_records = {}
# for record in records:
#     speaker = get_speaker_info_from_record(record)
#     if speaker not in speakerwise_records:
#         speakerwise_records[speaker] = []
#     speakerwise_records[speaker].append(record)

# for speaker in speakerwise_records:
#     random.shuffle(speakerwise_records[speaker])
#     speakerwise_records[speaker] = speakerwise_records[speaker][:20]

# print("Total Speakers: {}".format(len(speakerwise_records)))


# for speaker in speakerwise_records:
#     for ridx, record in enumerate(speakerwise_records[speaker]):
#         print("{} out of {}".format(ridx, len(speakerwise_records[speaker])))
#         context_fps = record["context"].split(";")
#         answer_fp = record["answer"]
#         try:
#             similarity = decode_codes_and_find_similarity(codec_model, context_fps + [answer_fp])
#         except:
#             print("Error in record: ", record)
#             continue
#         record["similarity"] = round(similarity, 4)
#         if similarity < 0.35:
#             low_similarity_records.append(record)
#         elif similarity > 0.7:
#             high_similarity_records.append(record)
#         with_similarity_records.append(record)
#         print(ridx, record["similarity"])
#         similarity_wise_sorted.append((similarity, record))
#         if (ridx+1) % 100 == 0:
#             updated_manifest_path = manifest_path.replace(".json", "_withsimilarity_inprogress.json")
#             write_manifest(updated_manifest_path, with_similarity_records)
#             plot_similarity_historgram(with_similarity_records)

#             low_similarity_manifest_path = manifest_path.replace(".json", "_low_similarity.json")
#             write_manifest(low_similarity_manifest_path, low_similarity_records)

#             high_similarity_manifest_path = manifest_path.replace(".json", "_high_similarity.json")
#             write_manifest(high_similarity_manifest_path, high_similarity_records)

# updated_manifest_path = manifest_path.replace(".json", "_similarity.json")

# similarity_wise_sorted = sorted(similarity_wise_sorted, key=lambda x: x[0])

# sorted_records = [x[1] for x in similarity_wise_sorted]


# write_manifest(updated_manifest_path, sorted_records)
