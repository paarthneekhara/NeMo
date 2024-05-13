from nemo.core.classes import Dataset
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir
import json
import os
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
import torch
from pathlib import Path
import nemo.collections.asr as nemo_asr
import tqdm
import numpy as np

def write_manifest(records, manifest_path):
    with open(manifest_path, "w") as f:
        for ridx, record in enumerate(records):
            if ridx < len(records) - 1:
                f.write(json.dumps(record) + "\n")
            else:
                f.write(json.dumps(record))
    print("Written to ", manifest_path)

class AudioDataset(Dataset):
    def __init__(
        self,
        manifest_paths,
        min_duration=0,
        max_duration=60.0,
        pad_multiple=1,
        sample_rate=16000,
        sup_data_dir=None,
    ):
        self.data = []
        for manifest_path in manifest_paths:
            with open(manifest_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    record["audio_filepath"] = record["audio_filepath"].replace("/data/filtered_24khz/", "/mnt/drive3/mls/filtered_24khz/")
                    if 'duration' in record and (record['duration'] < min_duration or record['duration'] > max_duration):
                        continue
                    self.data.append(record)

        self.base_data_dir = get_base_dir([item["audio_filepath"] for item in self.data])
        if sup_data_dir is not None:
            self.sup_data_dir = sup_data_dir
        else:
            self.sup_data_dir = os.path.join(self.base_data_dir, "sup_data")
        if not os.path.exists(self.sup_data_dir):
            os.makedirs(self.sup_data_dir)

        self.pad_multiple = pad_multiple
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def _get_wav_from_filepath(self, audio_filepath):
        features = AudioSegment.segment_from_file(
            audio_filepath, target_sr=self.sample_rate, n_segments=-1, trim=False,
        )
        audio_samples = features.samples
        audio, audio_length = torch.tensor(audio_samples), torch.tensor(audio_samples.shape[0]).long()

        # pad audio to a multiple of self.pad_multiple
        if audio.shape[0] % self.pad_multiple != 0:
            audio = torch.cat(
                [audio, torch.zeros(self.pad_multiple - audio.shape[0] % self.pad_multiple, dtype=torch.float)]
            )
            audio_length = torch.tensor(audio.shape[0]).long()

        return audio, audio_length

    def pad_collate_fn(self, batch):
        final_batch = {}
        for row in batch:
            for key in row:
                if key not in final_batch:
                    final_batch[key] = []
                final_batch[key].append(row[key])

        max_audio_len = max([_audio_len.item() for _audio_len in final_batch["audio_len"]])

        audios_padded = []
        for audio in final_batch["audio"]:
            audio_padded = torch.nn.functional.pad(audio, (0, max_audio_len - audio.size(0)), value=0)
            audios_padded.append(audio_padded)

        final_batch["audio"] = audios_padded
        for key in final_batch:
            if key not in ["rel_audio_path_as_text_id", "wav_path", "valid"]:
                final_batch[key] = torch.stack(final_batch[key])

        return final_batch

    def __getitem__(self, index):
        sample = self.data[index]
        rel_audio_path = Path(sample["audio_filepath"]).relative_to(self.base_data_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")
        # speaker = torch.tensor(sample["speaker"]).long()
        valid = False
        try:
            audio, audio_length = self._get_wav_from_filepath(sample["audio_filepath"])
            valid = True
        except:
            print("Invalid audio", sample["audio_filepath"])
            audio = torch.zeros(10000)
            audio_length = torch.tensor(audio.shape[0]).long()

        return {
            "audio": audio,
            "audio_len": audio_length,
            "rel_audio_path_as_text_id": rel_audio_path_as_text_id,
            "wav_path": sample["audio_filepath"],
            "valid" : valid
            # "speaker": speaker,
        }

manifest_paths = [
    # "/datap/misc/manifest_train_with3sec_ref_tts_speechllm.json",
    # "/datap/misc/manifests/manifests/libritts/train_clean_300_speechlm_ttstasks.json"
    "/datap/misc/RIVA-TTS/updated_manifests/all_train.json"
]

emb_out_dir = "/datap/misc/speaker_embeddings_extracted_riva"
if not os.path.exists(emb_out_dir):
    os.makedirs(emb_out_dir)

dataset = AudioDataset(
    manifest_paths, pad_multiple=1, sample_rate=16000, sup_data_dir=None
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=dataset.pad_collate_fn,
    num_workers=4,
)


nemo_sv_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large').cuda()
nemo_sv_model.eval()

# for batch in dataloader:
record_idx = 0
valid_records = []
for batch in tqdm.tqdm(dataloader):
    audio_signal, audio_signal_length, fileids, validities = batch['audio'], batch['audio_len'], batch['rel_audio_path_as_text_id'], batch['valid']
    with torch.no_grad():
        _, embs = nemo_sv_model.forward(input_signal=audio_signal.cuda(), input_signal_length=audio_signal_length.cuda())
    for emb, fileid, is_valid in zip(embs, fileids, validities):
        if is_valid:
            emb_path = os.path.join(emb_out_dir, fileid + ".npy")
            np.save(emb_path, emb.cpu().numpy())
            dataset.data[record_idx]["emb_path"] = emb_path
            valid_records.append(dataset.data[record_idx])
            # print(emb_path)
            if record_idx % 10000 == 0:
                out_manifest_path = manifest_paths[0].replace(".json", "_with_embs.json")
                write_manifest(valid_records, out_manifest_path)
        
        record_idx += 1

out_manifest_path = manifest_paths[0].replace(".json", "_with_embs.json")
write_manifest(valid_records, out_manifest_path)