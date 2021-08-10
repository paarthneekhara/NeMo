import os
import json
from nemo.collections.asr.models import EncDecSpeakerLabelModel
import torch

from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import argparse
import random

class SimpleAudioDataset(Dataset):
    def __init__(self, data_dir, records, speaker_emb_cfg):
        self.data_dir = data_dir
        self.records = records
        self.speaker_emb_cfg = speaker_emb_cfg

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        path2audio_file = os.path.join( self.data_dir, self.records[idx]['audio_filepath'] )
        audio, sr = librosa.load(path2audio_file, sr=None)
        target_sr = self.speaker_emb_cfg.train_ds.get('sample_rate', 16000)
        if sr != target_sr:
            audio = librosa.core.resample(audio, sr, target_sr)
        audio_length = audio.shape[0]
        target_len = int(10 * target_sr)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), 'constant', constant_values=(0.0, 0.0))
        audio = audio[:target_len]
        audio_signal, audio_signal_len = (
            torch.tensor(audio),
            torch.tensor(audio_length),
        )

        return audio_signal, audio_signal_len, path2audio_file


def main():

    parser = argparse.ArgumentParser(description='Extract speaker embeddings')
    parser.add_argument('--speaker_num', type=int)
    parser.add_argument('--max_records', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default="/home/pneekhara/SpeakerEmbeddings/")
    args = parser.parse_args()

    data_dir = "/home/pneekhara/Datasets/78419/Hi_Fi_TTS_v_0_backup"
    emb_dir = os.path.join(args.out_dir, str(args.speaker_num))
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)

    speaker = args.speaker_num

    clean_other_mapping = {
        92 : 'clean',
        6097 : 'clean',
        9017 : 'clean',
        6670 : 'other',
        6671 : 'other',
        8051 : 'clean',
        9136 : 'other',
        11614 : 'other',
        11697 : 'other',
    }

    manifest_path = os.path.join(data_dir, "{}_manifest_{}_{}.json".format(speaker, clean_other_mapping[speaker], "train"))

    speaker_verification_model = EncDecSpeakerLabelModel.from_pretrained("speakerverification_speakernet")
    speaker_verification_model.eval().cuda()

    train_records = []
    with open(manifest_path, "r") as f:
        for i, line in enumerate(f):
            train_records.append( json.loads(line) )
    
    random.seed(42)
    random.shuffle(train_records)
    if args.max_records is not None:
        train_records = train_records[:args.max_records]

    dataset = SimpleAudioDataset(data_dir, train_records, speaker_verification_model._cfg)
    dataloader = DataLoader(dataset, batch_size=64,shuffle=False, num_workers=4)

    speaker_verification_model.freeze()

    for bidx, batch in enumerate(dataloader):
        audio_signal, audio_signal_len, audio_paths = batch
        with torch.no_grad():
            _, embs = speaker_verification_model.forward(input_signal=audio_signal.cuda(), input_signal_length=audio_signal_len.cuda())
            for pidx, path in enumerate(audio_paths):
                subdir, filename = path.split("/")[-2:]
                emb_file_name = "{}-{}".format(subdir, filename.replace(".wav", ".npy"))
                emb_file_path = os.path.join(emb_dir, emb_file_name)
                emb_np = embs[pidx].cpu().numpy()
                np.save(emb_file_path, emb_np)
        
        if bidx % 10 == 0:
            print ("Completed {} out of {}".format(bidx, len(dataloader)))

if __name__ == '__main__':
    main()