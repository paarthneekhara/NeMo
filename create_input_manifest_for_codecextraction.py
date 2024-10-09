import glob
import soundfile as sf
import json
import argparse

def write_manifest(manifest_path, records):
    with open(manifest_path, 'w') as f:
        file_str = ""
        for record in records:
            file_str += json.dumps(record) + "\n"
        file_str = file_str.strip()
        f.write(file_str)
        print("Wrote {} records to: {}".format(len(records), manifest_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--manifest_path', type=str, required=True)
    parser.add_argument('--file_extension', type=str, required=False, default='wav')
    args = parser.parse_args()

    data_dir = args.data_dir
    manifest_path = args.manifest_path

    wav_files = glob.glob(data_dir + "/**/*.{}".format(args.file_extension) , recursive=True)
    wav_files = [ f for f in wav_files ]
    print("Found {} wav files in {}".format(len(wav_files), data_dir))
    records = []
    for aidx, audio_file in enumerate(wav_files):
        if aidx % 100 == 0:
            print("Processed {} files out of {}".format(aidx, len(wav_files)))
        audio_duration = sf.info(audio_file).duration
        record = {
            'audio_filepath' : audio_file,
            'text' : "dummy_text",
            'context' : audio_file,
            'duration' : round(float(audio_duration), 2),
            'context_duration' : round(float(audio_duration), 2),
            'speaker' : 'dummy_speaker'
        }
        records.append(record)
    
    if len(records) > 0:
        write_manifest(manifest_path, records)

if __name__ == '__main__':
    main()