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
        results, _ = beam_search(self.model.to(device), src_tokens.to(device), src_mask.to(device), self.config['max_len'][1],
                                 pad_idx, bos_idx, eos_idx, beam_size, device, self.is_terminated)
        if results is None:
            return None
        texts = []
        for result in results[0]:
            index_of_eos = result.index(2) if 2 in result else len(result)
            result = result[:index_of_eos + 1]
            texts.append(self.decode(result))
        return texts

    def translate_file(self, file, output, beam_size=3, device='cpu'):
        def translate_and_write(text):
            text = self.translate(text, beam_size, device)
            if text is not None:
                with open(output, 'a', encoding='utf-8') as f:
                    f.write(text[0] + '\n')
        try:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.readline()
                while True:
                    if self.is_terminated():
                        break
                    line = f.readline()
                    if not line:
                        if text:
                            translate_and_write(text)
                        break
                    if len(text + line) <= self.config['max_len'][0]:
                        text += line
                    else:
                        translate_and_write(text)
                        text = line
        except UnicodeDecodeError:
            print(f"Error decoding file: {file}. It may contain characters that are not UTF-8 encoded.")


import opencc


class ChineseConverter:
    def __init__(self):
        self.s2t_converter = opencc.OpenCC('s2t')  # 简体到繁体
        self.t2s_converter = opencc.OpenCC('t2s')  # 繁体到简体

    def convert(self, content, conversion_type='s2t'):
        """
        Convert the Chinese characters in content from Simplified to Traditional or vice versa.

        Args:
        - content (str): The text to be converted.
        - conversion_type (str): 's2t' for Simplified to Traditional, 't2s' for Traditional to Simplified.

        Returns:
        - str: Converted text.
        """
        if conversion_type == 's2t':
            return self.s2t_converter.convert(content)
        elif conversion_type == 't2s':
            return self.t2s_converter.convert(content)
        else:
            raise ValueError("Invalid conversion_type. Must be either 's2t' or 't2s'.")

    def convert_file(self, input_file_path, output_file_path, conversion_type='s2t'):
        """
        Convert the Chinese characters in a file from Simplified to Traditional or vice versa.

        Args:
        - input_file_path (str): Path to the input file.
        - output_file_path (str): Path to the output file.
        - conversion_type (str): 's2t' for Simplified to Traditional, 't2s' for Traditional to Simplified.
        """
        with open(input_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        converted_content = self.convert(content, conversion_type)

        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(converted_content)



