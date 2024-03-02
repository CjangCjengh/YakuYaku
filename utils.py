import json, re
import torch
import tokenizer
import cleaner
import zipfile
import glob, os, shutil
from model import init_model
from beam_decoder import beam_search
import torch.nn.utils.rnn as rnn_utils
import requests


def translate_txt(file, output, max_len, batch_size, translate_batch, is_terminated):
    def translate_and_write(text):
            results = translate_batch(text)
            if results is not None:
                with open(output, 'a', encoding='utf-8') as f:
                    for text in results:
                        f.write(text[0] + '\n')
    try:
        with open(file, 'r', encoding='utf-8') as f:
            text_batch = []
            text = f.readline()
            while not is_terminated():
                if len(text_batch) == batch_size:
                    translate_and_write(text_batch)
                    text_batch = []
                line = f.readline()
                if not line:
                    if text:
                        text_batch.append(text)
                    if text_batch:
                        translate_and_write(text_batch)
                    break
                if len(text + line) <= max_len:
                    text += line
                else:
                    text_batch.append(text)
                    text = line
    except UnicodeDecodeError:
        print(f'Error decoding file: {file}. Please ensure that the file is encoded in UTF-8.')


def translate_epub(file, output, max_len, batch_size, translate_batch, is_terminated):
    def translate_and_replace(text_batch, file_text):
            texts = [text for text, _, _ in text_batch]
            texts = translate_batch(texts)
            if texts is None:
                return ''
            new_file_text = ''
            for text, (_, matches, pre_end) in zip(texts, text_batch):
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
    files = glob.glob('./temp/**/*html', recursive=True)
    for file in files:
        if not os.path.isfile(file):
            continue
        try:
            print(f'Translating {file}...')
            with open(file, 'r', encoding='utf-8') as f:
                file_text = f.read()
                matches = re.finditer(r'<(h[1-6]|p|a|title).*?>(.+?)</\1>',file_text,flags=re.DOTALL)
                if not matches:
                    continue
                new_file_text = ''
                text_batch = []
                group = []
                text = ''
                pre_end = 0
                for match in matches:
                    if is_terminated():
                        break
                    if len(text_batch) == batch_size:
                        new_file_text += translate_and_replace(text_batch, file_text)
                        text_batch = []
                    if len(text + match.group(2)) <= max_len:
                        new_text = clean_text(match.group(2))
                        if new_text:
                            group.append(match)
                            text += '\n' + new_text
                    else:
                        text_batch.append((text, group, pre_end))
                        pre_end = group[-1].end()
                        new_text = clean_text(match.group(2))
                        if new_text:
                            group = [match]
                            text = clean_text(match.group(2))
                        else:
                            group = []
                            text = ''
                if text:
                    text_batch.append((text, group, pre_end))
                if text_batch:
                    new_file_text += translate_and_replace(text_batch, file_text)
                    new_file_text += file_text[group[-1].end():]
            if new_file_text:
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(new_file_text)
        except UnicodeDecodeError:
            print(f'Error decoding file: {file}. Please ensure that the file is encoded in UTF-8.')
    if not is_terminated():
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as f:
            for file_path in glob.glob(f'./temp/**', recursive=True):
                if not os.path.isdir(file_path):
                    relative_path = os.path.relpath(file_path, './temp')
                    f.write(file_path, relative_path)
    shutil.rmtree('./temp')


class Translator:
    def __init__(self, model_dir, device='cpu'):
        self._is_terminated = False

        with open(f'{model_dir}/config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.model = init_model(self.config['vocab_size'][0], self.config['vocab_size'][1],
                                self.config['n_layers'], self.config['d_model'],
                                self.config['d_ff'], self.config['n_heads']).to(device)
        self.model.load_state_dict(torch.load(f'{model_dir}/model.pth', map_location=device))
        self.model.eval()
        self.src_tokenizer = getattr(tokenizer, self.config['tokenizer'][0], None)
        self.tgt_tokenizer = getattr(tokenizer, self.config['tokenizer'][1], None)
        
        ic_names = self.config.get('input_cleaners', None)
        if ic_names is None:
            ic_names = [self.config['cleaner']]
        oc_names = self.config.get('output_cleaners', [])
        self.input_cleaners = [getattr(cleaner, c, None) for c in ic_names]
        self.output_cleaners = [getattr(cleaner, c, None) for c in oc_names]

        self.encode, _ = self.src_tokenizer(f'{model_dir}/{self.config["vocab_path"][0]}')
        _, self.decode = self.tgt_tokenizer(f'{model_dir}/{self.config["vocab_path"][1]}')

    def is_terminated(self):
        return self._is_terminated
    
    def terminate(self):
        self._is_terminated = True

    def translate(self, text, beam_size=3, device='cpu', input_cleaner=None, output_cleaner=None):
        text = self.translate_batch([text], beam_size, device, input_cleaner, output_cleaner)
        if text:
            return text[0]
        return None

    def translate_batch(self, text, beam_size=3, device='cpu', input_cleaner=None, output_cleaner=None):
        bos_idx = self.config['bos_idx']
        eos_idx = self.config['eos_idx']
        pad_idx = self.config['pad_idx']

        if self.input_cleaners is not None:
            for c in self.input_cleaners:
                text = [c(text_single) for text_single in text]

        if input_cleaner:
            text = [getattr(cleaner, input_cleaner)(text_single) for text_single in text]
        
        src_tokens = rnn_utils.pad_sequence((torch.LongTensor([bos_idx] + self.encode(t) + [eos_idx]) for t in text),
                                            batch_first=True, padding_value=pad_idx).to(device)
        src_mask = (src_tokens != pad_idx).unsqueeze(-2).to(device)
       
        results, _ = beam_search(self.model.to(device), src_tokens, src_mask, self.config['max_len'][1],
                                 pad_idx, bos_idx, eos_idx, beam_size, device, self.is_terminated)
        if results is None:
            return None
        texts_last = []
        for result_idx in results:
            texts = []
            for result in result_idx:
                index_of_eos = result.index(2) if 2 in result else len(result)
                result = result[:index_of_eos + 1]
                text = self.decode(result)
                for c in self.output_cleaners:
                    text = c(text)
                if output_cleaner:
                    text = getattr(cleaner, output_cleaner)(text)
                texts.append(text)
            texts_last.append(texts)
        return texts_last

    def translate_txt(self, file, output, beam_size=3, device='cpu', input_cleaner=None, output_cleaner=None, batch_size=1):
        def _translate_batch(text):
            return self.translate_batch(text, beam_size, device, input_cleaner, output_cleaner)
        translate_txt(file, output, self.config['max_len'][0], batch_size, _translate_batch, self.is_terminated)

    def translate_epub(self, file, output, beam_size=3, device='cpu', input_cleaner=None, output_cleaner=None, batch_size=1):
        def _translate_batch(text):
            return self.translate_batch(text, beam_size, device, input_cleaner, output_cleaner)
        translate_epub(file, output, self.config['max_len'][0], batch_size, _translate_batch, self.is_terminated)


class SakuraTranslator:
    def __init__(self, url):
        self._is_terminated = False
        self.url = url
        self.translate('こんにちは')

    def translate(self, text, input_cleaner=None, output_cleaner=None, **kwargs):
        if input_cleaner:
            text = getattr(cleaner, input_cleaner)(text)
        data = {
            'prompt': f'<reserved_106>将下面的日文文本翻译成中文：{text}<reserved_107>',
            'max_new_tokens': 1024,
            'do_sample': True,
            'temperature': 0.1,
            'top_p': 0.3,
            'repetition_penalty': 1.0,
            'num_beams': 1,
            'frequency_penalty': 0.05,
            'top_k': 40,
            'seed': -1
        }
        resp = requests.post(f'{self.url}/api/v1/generate', json=data).json()
        text = resp['results'][0]['text']
        if output_cleaner:
            text = getattr(cleaner, output_cleaner)(text)
        return [text]
    
    def is_terminated(self):
        return self._is_terminated
    
    def terminate(self):
        self._is_terminated = True

    def translate_txt(self, file, output, input_cleaner=None, output_cleaner=None, **kwargs):
        def translate_batch(text):
            return [self.translate(text[0], input_cleaner, output_cleaner)]
        translate_txt(file, output, 768, 1, translate_batch, self.is_terminated)
    
    def translate_epub(self, file, output, input_cleaner=None, output_cleaner=None, **kwargs):
        def translate_batch(text):
            return [self.translate(text[0], input_cleaner, output_cleaner)]
        translate_epub(file, output, 768, 1, translate_batch, self.is_terminated)
