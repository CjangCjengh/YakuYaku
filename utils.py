import json
import torch
import tokenizer
import cleaner
from model import init_model
from beam_decoder import beam_search

class Translator:
    def __init__(self, model_dir, device='cpu'):
        self._is_terminated = False

        with open(f'{model_dir}/config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        with open(f'{model_dir}/vocabs_source.json', 'r', encoding='utf-8') as f:
            self.vocabs_source = json.load(f)
        with open(f'{model_dir}/vocabs_target.json', 'r', encoding='utf-8') as f:
            self.vocabs_target = json.load(f)
        self.model = init_model(len(self.vocabs_source)+259, len(self.vocabs_target)+259,
                                self.config['n_layers'], self.config['d_model'],
                                self.config['d_ff'], self.config['n_heads']).to(device)
        self.model.load_state_dict(torch.load(f'{model_dir}/model.pth', map_location=device))
        self.model.eval()
        self.tokenizer = getattr(tokenizer, self.config['tokenizer'], None)
        self.cleaner = getattr(cleaner, self.config['cleaner'], None)
        if self.tokenizer is not None:
            self.encode, _ = self.tokenizer(self.vocabs_source)
            _, self.decode = self.tokenizer(self.vocabs_target)

    def is_terminated(self):
        return self._is_terminated
    
    def terminate(self):
        self._is_terminated = True

    def translate(self, text, beam_size=3, device='cpu'):
        bos_idx = self.config['bos_idx']
        eos_idx = self.config['eos_idx']
        pad_idx = self.config['pad_idx']
        if self.cleaner is not None:
            text = self.cleaner(text)
        src_tokens = torch.LongTensor([[bos_idx] + self.encode(text) + [eos_idx]])
        src_mask = (src_tokens != pad_idx).unsqueeze(-2)
        results, _ = beam_search(self.model.to(device), src_tokens, src_mask, self.config['max_len'],
                                 pad_idx, bos_idx, eos_idx, beam_size, device, self.is_terminated)
        if results is None:
            return None
        texts = []
        for result in results[0]:
            index_of_eos = result.index(2) if 2 in result else len(result)
            result = result[:index_of_eos + 1]
            texts.append(self.decode(result))
        return texts
