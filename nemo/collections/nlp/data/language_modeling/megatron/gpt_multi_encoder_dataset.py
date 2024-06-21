from nemo.core.classes import Dataset
import json
import torch
import math

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
    ):
        self.dataset_paths = dataset_paths
        self.speech_token_offset = speech_token_offset
        self.speech_codebook_size = speech_codebook_size
        self.num_speech_codebooks = num_speech_codebooks
        self.speech_bos_id = speech_codebook_size - 3
        self.speech_eos_id = speech_codebook_size - 2
        self.speech_pad_id = speech_codebook_size - 1

        self.max_seq_length = max_seq_length
        self.codebook_fps = codebook_fps
        self.tokenizer = tokenizer
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
        question_tokens = torch.nn.utils.rnn.pad_sequence(
            [x['question_tokens'] for x in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_id,
        )
        
        max_length = answer_codes_BCT.size(2) - 1
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.tensor(position_ids).long()

        answer_codes_sliced = answer_codes_BCT[:,0,1:]
        loss_mask = torch.ones_like(answer_codes_sliced).float()
        loss_mask[answer_codes_sliced == self.tokenizer.pad_id] = 0
        
        gpt_batch = {
            'tokens': answer_codes_BCT[:,:,:-1],
            'labels' : answer_codes_BCT[:,:,1:],
            'loss_mask': loss_mask,
            'position_ids': position_ids,
        }

        gpt_batch['context_BCT'] = context_code_BCT
        gpt_batch['question_tokens'] = question_tokens

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
        answer_codes_TC = self.offset_speech_codes(answer_codes_TC)
        context_codes_TC = self.offset_speech_codes(context_codes_TC)
        # Add bos and eos tokens to speech to the Time dimension
        answer_codes_TC = torch.cat(
            [
                self.get_speech_special_tokens(self.speech_bos_id), 
                answer_codes_TC, 
                self.get_speech_special_tokens(self.speech_eos_id), 
            ], 
            dim=0
        )

        context_codes_TC = torch.cat(
            [
                self.get_speech_special_tokens(self.speech_bos_id), 
                context_codes_TC, 
                self.get_speech_special_tokens(self.speech_eos_id), 
            ], 
            dim=0
        )
        
        question_text = record['question']
        question_tokens = self.tokenizer.text_to_ids(question_text)
        # Add bos and eos tokens
        question_tokens = [self.tokenizer.bos_id] + question_tokens + [self.tokenizer.eos_id]
        question_tokens = torch.tensor(question_tokens).long()
        
        return {
            'answer_codes_TC': answer_codes_TC,
            'context_codes_TC': context_codes_TC,
            'question_tokens': question_tokens,
        }