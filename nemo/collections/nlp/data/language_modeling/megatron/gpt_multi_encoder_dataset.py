from dataclasses import dataclass
from nemo.core import Dataset
import json
import torch
import math
from nemo.collections.tts.parts.utils.tts_dataset_utils import BetaBinomialInterpolator, beta_binomial_prior_distribution
from hydra.utils import instantiate
from omegaconf import OmegaConf

@dataclass
class G2PConfig:
    _target_: str = "nemo.collections.tts.g2p.models.en_us_arpabet.EnglishG2p"
    phoneme_dict: str = "scripts/tts_dataset_files/cmudict-0.7b_nv22.10"
    heteronyms: str = "scripts/tts_dataset_files/heteronyms-052722"
    phoneme_probability: float = 0.5


@dataclass
class TextTokenizer:
    _target_: str = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.EnglishPhonemesTokenizer"
    punct: bool = True
    stresses: bool = True
    chars: bool = True
    apostrophe: bool = True
    pad_with_space: bool = True
    add_blank_at: bool = True
    g2p: G2PConfig = G2PConfig()


@dataclass
class TextTokenizerConfig:
    text_tokenizer: TextTokenizer = TextTokenizer()


def _get_default_text_tokenizer_conf(phoneme_probability=0.5):
    g2p = G2PConfig(phoneme_probability=phoneme_probability)
    _text_tokenizer = TextTokenizer(g2p=g2p)
    text_tokenizer: TextTokenizerConfig = TextTokenizerConfig(text_tokenizer=_text_tokenizer)
    return OmegaConf.create(OmegaConf.to_yaml(text_tokenizer))

phoneme_tokenizer = instantiate(_get_default_text_tokenizer_conf(phoneme_probability=0.5)).text_tokenizer

class GPTMultiEncoderSpeechDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_paths,
        speech_token_offset,
        num_speech_codebooks=8,
        speech_codebook_size=1027,
        codebook_fps=86,
        max_seq_length=2048,
        attention_prior_scaling_factor=0.05,
        phoneme_probability=0.5,
        lm_vocab_size=32100,
    ):
        self.dataset_paths = dataset_paths
        self.speech_token_offset = speech_token_offset
        self.speech_codebook_size = speech_codebook_size
        self.num_speech_codebooks = num_speech_codebooks
        self.speech_bos_id = speech_codebook_size - 3
        self.speech_eos_id = speech_codebook_size - 2
        self.speech_pad_id = speech_codebook_size - 1
        self.lm_vocab_size = lm_vocab_size

        self.max_seq_length = max_seq_length
        self.codebook_fps = codebook_fps
        self.tokenizer = tokenizer
        self.attention_prior_scaling_factor = attention_prior_scaling_factor
        self.records = self.load_data()
        
    
    def load_data(self):
        records = []
        for dataset_path in self.dataset_paths:
            with open(dataset_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    record = json.loads(line.strip())
                    estimated_answer_length = int(math.ceil( (record['answer_duration']+0.1) * self.codebook_fps) + 2)
                    estimated_context_length = int(math.ceil( (record['context_duration']+0.1) * self.codebook_fps) + 2)
                    if estimated_answer_length > self.max_seq_length or estimated_context_length > self.max_seq_length:
                        continue
                    records.append(record)
        return records
    
    def __len__(self):
        return len(self.records)
    
    def collate_fn(self, batch):
        answer_codes_BCT = torch.nn.utils.rnn.pad_sequence(
            [x['answer_codes_TC'] for x in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_id,
        ).transpose(1, 2)

        context_code_BCT = torch.nn.utils.rnn.pad_sequence(
            [x['context_codes_TC'] for x in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_id,
        ).transpose(1, 2)
        context_mask = (context_code_BCT[:,0,:] != self.tokenizer.pad_id).float()

        text_tokens = torch.nn.utils.rnn.pad_sequence(
            [x['text_tokens'] for x in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_id,
        )
        text_mask = (text_tokens != self.tokenizer.pad_id).float()
        
        max_text_len = text_tokens.size(1)
        max_audio_len = answer_codes_BCT.size(2) - 1
        cross_attention_text_prior_list = []
        for x in batch:
            cross_attention_text_prior = x['cross_attention_text_prior']
            cross_attention_text_prior = torch.nn.functional.pad(
                cross_attention_text_prior,
                (0, max_text_len - cross_attention_text_prior.size(1), 0, max_audio_len - cross_attention_text_prior.size(0)),
                value=1,
            )
            cross_attention_text_prior_list.append(cross_attention_text_prior)

        cross_attention_text_prior = torch.stack(cross_attention_text_prior_list, dim=0)

        position_ids = [list(range(max_audio_len)) for _ in batch]
        position_ids = torch.tensor(position_ids).long()

        answer_codes_sliced = answer_codes_BCT[:,0,1:]
        loss_mask = torch.ones_like(answer_codes_sliced).float()
        loss_mask[answer_codes_sliced == self.tokenizer.pad_id] = 0
        
        # Causal Attention Mask
        batch_size = answer_codes_BCT.size(0)
        attention_mask = torch.tril(torch.ones((batch_size, max_audio_len, max_audio_len))).view(
            batch_size, 1, max_audio_len, max_audio_len
        )

        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5

        gpt_batch = {
            'tokens': answer_codes_BCT[:,:,:-1],
            'labels' : answer_codes_BCT[:,:,1:],
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
        }

        gpt_batch['context_BCT'] = context_code_BCT
        gpt_batch['context_mask'] = context_mask
        gpt_batch['text_tokens'] = text_tokens
        gpt_batch['text_mask'] = text_mask
        gpt_batch['cross_attention_text_prior'] = cross_attention_text_prior

        return gpt_batch
    
    def offset_speech_codes(self, codes_TC):
        codes_adjusted = codes_TC * 1
        for _c in range(self.num_speech_codebooks):
            codes_adjusted[:,_c] += self.speech_token_offset + self.speech_codebook_size * _c
        return codes_adjusted.long()

    def get_speech_special_tokens(self, token_id):
        # token_id is 1024, 1025 or 1026
        special_token_ids = []
        for _c in range(self.num_speech_codebooks):
            special_token_ids.append(token_id + self.speech_token_offset + self.speech_codebook_size * _c)
        special_token_ids = torch.tensor(special_token_ids).long().unsqueeze(0)
        return special_token_ids

    def __getitem__(self, idx):
        record = self.records[idx]
        answer_codes_TC = torch.load(record['answer']).long().transpose(0, 1) # T x 8
        context_codes_TC = torch.load(record['context']).long().transpose(0, 1) # T x 8
        context_codes_TC = context_codes_TC + 1 # 0 is reserved for padding
        answer_codes_TC = self.offset_speech_codes(answer_codes_TC)
        # Add bos and eos tokens to speech to the Time dimension
        answer_codes_TC = torch.cat(
            [
                self.get_speech_special_tokens(self.speech_bos_id), 
                answer_codes_TC, 
                self.get_speech_special_tokens(self.speech_eos_id), 
            ], 
            dim=0
        )
        audio_len = answer_codes_TC.size(0)

        context_codes_TC = torch.cat(
            [
                context_codes_TC, 
                torch.tensor([self.speech_codebook_size-2 for _ in range(self.num_speech_codebooks)]).long().unsqueeze(0), # EOS Token
            ], 
            dim=0
        )
        
        question_text = record['question'].replace("Phoneme TTS ", "").replace("Text to speech this ", "")
        # text_tokens = self.tokenizer.text_to_ids(question_text)
        text_tokens = phoneme_tokenizer.encode(question_text)
        text_tokens = [t + self.lm_vocab_size for t in text_tokens]
        # Add bos and eos tokens
        text_tokens = text_tokens + [self.tokenizer.eos_id]
        text_len = len(text_tokens)

        text_tokens = torch.tensor(text_tokens).long()
        
        cross_attention_text_prior = torch.from_numpy(beta_binomial_prior_distribution(
            text_len,
            audio_len,
            scaling_factor=self.attention_prior_scaling_factor,
        ))

        return {
            'answer_codes_TC': answer_codes_TC,
            'context_codes_TC': context_codes_TC,
            'text_tokens': text_tokens,
            'cross_attention_text_prior': cross_attention_text_prior
        }