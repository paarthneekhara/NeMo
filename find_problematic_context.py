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
import time
import tqdm

# manifest_path = "/datap/misc/speechllm_codecdatasets/manifests/LRHM_train_nemo_codec_bw_6.0_phoneme_tts.json"
# manifest_path = "/datap/misc/speechllm_codecdatasets/manifests/hifitts_nemo_codec_bw_6.0_train_sentencepiece_tts_highsimilarity.json"
manifest_path = "/datap/misc/speechllm_codecdatasets/manifests/LibriTTSCorrectContext_train_nemo_codec_bw_6.0_sentencepiece_tts_highsimilarity2.json"
# Len high similarity records:  299057
# Orig len 312347
codec_model_path = "/datap/misc/Checkpoints/SpeechCodec.nemo"

resampler = Resample(orig_freq=22050, new_freq=16000).cuda()
batch_size = 16

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


def decode_codes_and_find_similarity(codec_model, codec_fps):
    """
    Decodes the codec_fps (codec filepaths, typically context and answer)
    and finds the similarity between the embeddings
    """
    assert len(codec_fps) % 2 == 0, "Expecting even number of codecs"

    st = time.time()
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
        
        print("Loading codecs took: ", round(time.time()-st, 4) )
        t1 = time.time()
        codec_lens = torch.stack(codec_lens).long().cuda()
        codecs = torch.zeros(len(codec_list), codec_list[0].shape[0], max_codec_len).cuda()
        for i, codec in enumerate(codec_list):
            codecs[i, :, :codec.shape[1]] = codec
        codecs = codecs.long()
        t5 = time.time()
        print("Stacking took: ", round(t5-t1, 4) )
        codec_decoded_audios, _ = codec_model.decode(tokens=codecs, tokens_len=codec_lens[:,0])
        print("Decoding took: ", round(time.time()-t5, 4) )
        t2 = time.time()
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
        print("Resampling took: ", round(time.time()-t2, 4) )
        t3 = time.time()
        audio_16_batch = torch.zeros(len(audios_16), max_16_len).cuda()
        for idx in range(len(audios_16)):
            audio_16_batch[idx, :audio_16_lens[idx]] = audios_16[idx]
        
        audio_signal_len_16 = torch.Tensor(audio_16_lens).long().cuda()
        # import ipdb; ipdb.set_trace()
        nemo_sv_model.eval()
        # nemo_sv_model.freeze()
        _, embs = nemo_sv_model.forward(input_signal=audio_16_batch, input_signal_length=audio_signal_len_16)
        print("embs", embs.shape)
        # Find cosine similarity between embs
        print("Embedding took: ", round(time.time()-t3, 4) )
        similarities = []
        for i in range(0, len(embs), 2):
            emb1 = embs[i]
            emb2 = embs[i+1]
            similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
            similarities.append(similarity)

        assert len(similarities) == len(embs) // 2
        return similarities




records = read_manifest(manifest_path)

codec_model = AudioCodecModel.restore_from(codec_model_path)
codec_model.to('cuda')
codec_model.eval()
codec_model_sample_rate = 22050
codec_model_downsampling_factor = 256.0

nemo_sv_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')

with_similarity_records = []
low_similarity_records = []
high_similarity_records = []
batch_fps = []
batch_records = []

# Change this to tqdm loop
# for ridx, record in enumerate(records):
for ridx, record in enumerate(tqdm.tqdm(records)):
    print("{} out of {}".format(ridx, len(records)))
    context_fps = record["context"].split(";")
    assert len(context_fps) == 1, "Multiple context files not supported"
    answer_fp = record["answer"]
    batch_fps.extend(context_fps + [answer_fp])
    batch_records.append(record)
    if ridx % 16 == 0 or ridx == len(records)-1:
        try:
            similarities = decode_codes_and_find_similarity(codec_model, batch_fps)
        except:
            print("Error decoding")
            batch_fps = []
            batch_records = []
            continue
        for idx, batch_record in enumerate(batch_records):
            batch_record["similarity"] = round(similarities[idx], 4)
            with_similarity_records.append(batch_record)
            print("Len with_similarity_records", len(with_similarity_records), len(batch_records))
            if similarities[idx] < 0.35:
                low_similarity_records.append(batch_record)
            elif similarities[idx] > 0.7:
                high_similarity_records.append(batch_record)

        batch_fps = []
        batch_records = []
    
    if (ridx+1) % 1000==0 or ridx == len(records)-1:
        updated_manifest_path = manifest_path.replace(".json", "_withsimilarity_inprogress2.json")
        write_manifest(updated_manifest_path, with_similarity_records)
        plot_similarity_historgram(with_similarity_records)

        low_similarity_manifest_path = manifest_path.replace(".json", "_low_similarity2.json")
        write_manifest(low_similarity_manifest_path, low_similarity_records)

        high_similarity_manifest_path = manifest_path.replace(".json", "_high_similarity2.json")
        write_manifest(high_similarity_manifest_path, high_similarity_records)