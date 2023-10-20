from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import glob
from pathlib import Path
import random

noise_dir = Path("/datap/misc/noisedata")

def create_noise_manifest(base_dir, subset, offset=0, duration=None):
    """Split the noise data set into train and test subsets.
    """
    complete_noise_manifests = glob.glob(str(base_dir / 'manifests' / '*.json'))
    subset_noise_manifest = base_dir / f'{subset}_manifest.json'
    
    subset_metadata = []

    for noise_manifest in complete_noise_manifests:
        complete_metadata = read_manifest(noise_manifest)
    
        for item in complete_metadata:
            new_item = item.copy()
            offset = random.randint(0, 100)
            duration = 20
            new_item['offset'] = offset
            new_item['duration'] = duration
            subset_metadata.append(new_item)

    write_manifest(subset_noise_manifest.as_posix(), subset_metadata)

    return subset_noise_manifest

noise_manifest = {
    'train': create_noise_manifest(noise_dir, 'train', offset=0, duration=200),
    'test': create_noise_manifest(noise_dir, 'test', offset=200, duration=100),
}