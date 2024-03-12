import os
import json
import argparse
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate_detail
import string
import pprint
from transformers import AutoFeatureExtractor, WavLMForXVector
import librosa
import torch

def _find_audio_files(audio_dir):
    audio_file_lists = {
        'gt' : [],
        'pred' : []
    }
    for f in os.listdir(audio_dir):
        if f.endswith(".wav") and "16khz" not in f:
            audio_number = int(f.split("_")[-1].split(".wav")[0])
            if "dec_input" in f:
                audio_file_lists['gt'].append((audio_number, os.path.join(audio_dir, f)))
            elif 'predicted' in f:
                audio_file_lists['pred'].append((audio_number, os.path.join(audio_dir, f)))
    
    return audio_file_lists


def find_sample_audios(exp_name, exp_base_dir, no_subdir=False):
    exp_dir = os.path.join(exp_base_dir, exp_name)
    sub_dirs = [x[0] for x in os.walk(exp_dir)]
    
    if not no_subdir:
        for sub_dir in sub_dirs:
            if "Sample_Audios" in sub_dir:
                audio_file_lists = _find_audio_files(sub_dir)
    else:
        audio_file_lists = _find_audio_files(exp_dir)

    audio_file_lists['gt'].sort()
    audio_file_lists['pred'].sort()
    audio_file_lists['gt'] = [t[1] for t in audio_file_lists['gt']]
    audio_file_lists['pred'] = [t[1] for t in audio_file_lists['pred']]
    return audio_file_lists


def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            records.append(json.loads(line))
    return records

def process_text(input_text):
    # Convert text to lowercase
    lower_case_text = input_text.lower()
    
    # Remove commas from text
    no_comma_text = lower_case_text.replace(",", "")
    
    # Replace "-" with spaces
    no_dash_text = no_comma_text.replace("-", " ")
    
    # Replace double spaces with single space
    single_space_text = " ".join(no_dash_text.split())

    single_space_text = single_space_text.translate(str.maketrans('', '', string.punctuation))
    
    return single_space_text

def contains_invalid_text(text):
    invalid_substrings = [
        "one b two zero four nine two eight zero zero zero",
    ]
    for invalid_substring in invalid_substrings:
        if invalid_substring in text:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description='Evaluate on challenging texts')
    parser.add_argument('--exp_name', type=str, default="2024-01-16_17-30-29")
    parser.add_argument('--exp_base_dir', type=str, default="/Data/Experiments/EVAL_NEMO_CODEC/temp08/jason_challenging_texts_ctc_corrected")
    parser.add_argument('--manifest_path', type=str, default="/Data/CodecDatasets/updatedcodecs/manifests/challenging_nemo_codec_phoneme.json")
    parser.add_argument('--eval_type', type=str, default="pred") # pred or gt
    parser.add_argument('--no_subdir', type=str, default="false")
    parser.add_argument('--eval_speaker', type=str, default="false") # false or true
    args = parser.parse_args()

    audio_file_lists = find_sample_audios(args.exp_name, args.exp_base_dir, no_subdir=args.no_subdir=="true")
    pred_audio_files = audio_file_lists[args.eval_type]
    

    manifest_records = read_manifest(args.manifest_path)
    max_answer_duration = 0
    max_record = None
    for record in manifest_records:
        if record['answer_duration'] > max_answer_duration:
            max_answer_duration = record['answer_duration']
            max_record = record
    print("Max Answer Duration:", max_answer_duration)

    # import ipdb;  ipdb.set_trace()

    # assert len(pred_audio_files) == len(manifest_records), "Len Pred Audio Files: {} Len Manifest Records: {}".format(len(pred_audio_files), len(manifest_records))

    device = "cuda"
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                    model_name="stt_en_conformer_transducer_xlarge"
                )
    asr_model = asr_model.to(device)
    asr_model.eval()

    if args.eval_speaker == "true":
        wavlm_feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        wavlm_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
        wavlm_model = wavlm_model.to(device)
        wavlm_model.eval()
        gt_audio_files = audio_file_lists['gt']
        assert len(gt_audio_files) == len(pred_audio_files), "Len GT Audio Files: {} Len Pred Audio Files: {}".format(len(gt_audio_files), len(pred_audio_files))

    pred_texts = []
    gt_texts = []
    wer_ranked_list = []
    all_cers = []
    wavlm_similarities = []
    for ridx, record in enumerate(manifest_records[:len(pred_audio_files)]):
        gt_text = process_text(record['text'])
        if contains_invalid_text(gt_text):
            continue
        pred_text = asr_model.transcribe([pred_audio_files[ridx]])[0][0]
        pred_text = process_text(pred_text)
        pred_texts.append(pred_text)
        gt_texts.append(gt_text)

        print ("Ridx", ridx)
        print ("GT:", gt_text)
        print ("Pred:", pred_text)

        detailed_cer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=True)
        detailed_wer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=False)
        print("CER:", detailed_cer[0])
        wer_ranked_list.append(
            (detailed_cer[0], detailed_wer[0], gt_text, pred_text, pred_audio_files[ridx])
        )

        if args.eval_speaker == "true":
            pred_wav_np_16, _ = librosa.load(pred_audio_files[ridx], sr=16000)
            gt_audio_file = record['audio_filepath']
            print("gt_audio_file:", gt_audio_file)
            # gt_audio_file = gt_audio_files[ridx]
            gt_wav_np_16, _ = librosa.load(gt_audio_file, sr=16000)
            wavlm_inputs = wavlm_feature_extractor([pred_wav_np_16, gt_wav_np_16], sampling_rate=16000, return_tensors="pt", padding=True)
            wavlm_inputs = {k: v.to(device) for k, v in wavlm_inputs.items()}
            wavlm_embeddings = wavlm_model(**wavlm_inputs).embeddings
            wavlm_embeddings = torch.nn.functional.normalize(wavlm_embeddings, dim=-1).cpu()
            cosine_sim = torch.nn.CosineSimilarity(dim=-1)
            wavlm_similarity = cosine_sim(wavlm_embeddings[0], wavlm_embeddings[1]).item()
            wavlm_similarities.append(wavlm_similarity)
            print("WAVLM Similarity:", wavlm_similarity)

    
    # Reverse sort by CER
    # Print challenging texts with highest CER
    print("*"*50)
    print("Top 10 Challenge Texts")
    print("*"*50)
    wer_ranked_list.sort(key=lambda x: x[0], reverse=True)
    for item in wer_ranked_list[:10]:
        print ("CER:", item[0])
        print ("WER:", item[1])
        print ("GT:", item[2])
        print ("Pred:", item[3])
        print ("Audio:", item[4])
        print ("-"*50)
    
    cumulative_cer_metrics = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=True)
    cumulative_wer_metrics = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=False)
    cer, words, ins_rate, del_rate, sub_rate = cumulative_cer_metrics
    wer, words_wer, ins_rate_wer, del_rate_wer, sub_rate_wer = cumulative_wer_metrics
    print("*"*50)
    print("Cumulative CER Metrics")
    print("*"*50)
    print ("CER:", cer)
    print ("WER:", wer)
    print ("Words:", words)
    print ("Ins:", ins_rate)
    print ("Del:", del_rate)
    print ("Sub:", sub_rate)
    

    out_dir = os.path.join(args.exp_base_dir, args.exp_name)
    all_metrics = {
        'average' : {
            'cer' : cer,
            'wer' : wer,
            'words' : words,
            'ins' : ins_rate,
            'del' : del_rate,
            'sub' : sub_rate,
        },
        'detailed' : wer_ranked_list
    }

    if args.eval_speaker == "true":
        all_metrics['average']['wavlm_similarity'] = sum(wavlm_similarities) / len(wavlm_similarities)


    pprint.pprint(all_metrics['average'])
    
    with open(os.path.join(out_dir, "hallucination_metrics.json"), 'w') as f:
        json.dump(all_metrics, f, indent=4)
        print("Wrote metrics to:", os.path.join(out_dir, "hallucination_metrics.json"))

if __name__ == "__main__":
    main()