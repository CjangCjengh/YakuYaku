import json, re
import torch
import tokenizer
import cleaner
import zipfile
import glob, os, shutil
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
        
        ic_names = self.config.get('input_cleaners', None)
        if ic_names is None:
            ic_names = [self.config['cleaner']]
        oc_names = self.config.get('output_cleaners', [])
        self.input_cleaners = [getattr(cleaner, c, None) for c in ic_names]
        self.output_cleaners = [getattr(cleaner, c, None) for c in oc_names]

        if self.tokenizer is not None:
            self.encode, _ = self.tokenizer(self.vocabs_source)
            _, self.decode = self.tokenizer(self.vocabs_target)

    def is_terminated(self):
        return self._is_terminated
    
    def terminate(self):
        self._is_terminated = True

    def translate(self, text, beam_size=3, device='cpu', input_cleaner=None, output_cleaner=None):
        bos_idx = self.config['bos_idx']
        eos_idx = self.config['eos_idx']
        pad_idx = self.config['pad_idx']
        if self.input_cleaners is not None:
            for c in self.input_cleaners:
                text = c(text)
        if input_cleaner:
            text = getattr(cleaner, input_cleaner)(text)
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
            text = self.decode(result)
            for c in self.output_cleaners:
                text = c(text)
            if output_cleaner:
                text = getattr(cleaner, output_cleaner)(text)
            texts.append(text)
        return texts

    def translate_txt(self, file, output, beam_size=3, device='cpu', input_cleaner=None, output_cleaner=None):
        def translate_and_write(text):
            text = self.translate(text, beam_size, device, input_cleaner, output_cleaner)
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
            print(f"Error decoding file: {file}. Please ensure that the file is encoded in UTF-8.")

    def translate_epub(self, file, output, beam_size=3, device='cpu', input_cleaner=None, output_cleaner=None):
        def translate_and_replace(text, file_text, matches, pre_end):
            text = self.translate(text, beam_size, device, input_cleaner, output_cleaner)
            new_file_text = ''
            if text is not None:
                text = text[0].split('\n')
                if len(text) < len(matches):
                    text += [''] * (len(matches) - len(text))
                else:
                    text = text[:len(matches)-1] + ['<br/>'.join(text[len(matches)-1:])]
                for t, match in zip(text, matches):
                    t = match.group(0).replace(match.group(2), t)
                    new_file_text += file_text[pre_end:match.start()] + t
                    pre_end = match.end()
            return new_file_text
        
        def clean_text(text):
            text=re.sub(r'<rt[^>]*?>.*?</rt>','',text)
            text=re.sub(r'<[^>]*>|\n','',text)
            return text

        if os.path.exists('./temp'):
            shutil.rmtree('./temp')
        with zipfile.ZipFile(file, 'r') as f:
            f.extractall('./temp')
        files = glob.glob("./temp/**/*html", recursive=True)
        for file in files:
            if not os.path.isfile(file):
                continue
            try:
                print(f'Translating {file}...')
                with open(file, 'r', encoding='utf-8') as f:
                    file_text = f.read()
                    matches = re.finditer(r'<(h[1-6]|p).*?>(.+?)</\1>',file_text,flags=re.DOTALL)
                    if not matches:
                        continue
                    new_file_text = ''
                    group = []
                    text = ''
                    pre_end = 0
                    for match in matches:
                        if self.is_terminated():
                            break
                        if len(text + match.group(2)) <= self.config['max_len'][0]:
                            new_text = clean_text(match.group(2))
                            if new_text:
                                group.append(match)
                                text += '\n' + new_text
                        else:
                            new_file_text += translate_and_replace(text, file_text, group, pre_end)
                            pre_end = group[-1].end()
                            new_text = clean_text(match.group(2))
                            if new_text:
                                group = [match]
                                text = clean_text(match.group(2))
                            else:
                                group = []
                                text = ''
                    if text:
                        new_file_text += translate_and_replace(text, file_text, group, pre_end)
                        new_file_text += file_text[group[-1].end():]
                if new_file_text:
                    with open(file, 'w', encoding='utf-8') as f:
                        f.write(new_file_text)
            except UnicodeDecodeError:
                print(f"Error decoding file: {file}. Please ensure that the file is encoded in UTF-8.")
        if not self.is_terminated():
            with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as f:
                for file_path in glob.glob(f'./temp/**', recursive=True):
                    if not os.path.isdir(file_path):
                        relative_path = os.path.relpath(file_path, './temp')
                        f.write(file_path, relative_path)
        shutil.rmtree('./temp')
