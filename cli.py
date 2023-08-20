# import torch
import argparse
import os

from utils import Translator
class Text:
    def __init__(self, name, content, dir):
        self.name = name
        self.content = content
        self.dir = dir

class Model:
    def __init__(self, name, dir):
        self.name = name
        self.dir = dir
class cli_process:
    DEFAULT_PARAM_MODEL_NAME = "value1"
    DEFAULT_PARAM_MODEL_DIR = "./models"
    DEFAULT_PARAM_INPUT_TEXT = "input_text"
    DEFAULT_PARAM_INPUT_TEXT_NAME = "input.txt"
    DEFAULT_PARAM_INPUT_TEXT_DIR = "./inputs"
    DEFAULT_PARAM_OUTPUT_TEXT = "output_text"
    DEFAULT_PARAM_OUTPUT_TEXT_NAME = "output.txt"
    DEFAULT_PARAM_OUTPUT_TEXT_DIR = "./outputs"
    DEFAULT_PARAM_BEAM_SIZE = 4;
    def __init__(self, model_name = None, model_dir=None, input_text_name=None, input_text_content=None, input_text_dir = None, output_text_name=None,
                 output_text_content=None, output_text_dir=None,device='cpu',beam_size=None):
        self.model = Model(model_name or self.DEFAULT_PARAM_MODEL_NAME,
                           model_dir or self.DEFAULT_PARAM_MODEL_DIR)

        self.input_text = Text(input_text_name or self.DEFAULT_PARAM_INPUT_TEXT_NAME,
                               input_text_content or self.DEFAULT_PARAM_INPUT_TEXT,
                               input_text_dir or self.DEFAULT_PARAM_INPUT_TEXT_DIR)

        self.output_text = Text(output_text_name or self.DEFAULT_PARAM_OUTPUT_TEXT_NAME,
                                output_text_content or self.DEFAULT_PARAM_OUTPUT_TEXT,
                                output_text_dir or self.DEFAULT_PARAM_OUTPUT_TEXT_DIR)
        self.device = device
        self.beam_size = beam_size or self.DEFAULT_PARAM_BEAM_SIZE

    # For input_text
    def set_model_name(self, model_name):
        self.model.name = model_name

    def get_model_name(self):
        return self.model.name

    def set_model_dir(self, model_dir):
        self.model.dir = model_dir

    def get_model_dir(self):
        return self.model.dir

    def set_input_text_name(self, input_text_name):
        self.input_text.name = input_text_name

    def get_input_text_name(self):
        return self.input_text.name

    def set_input_text_content(self, input_text_content):
        self.input_text.content = input_text_content

    def get_input_text_content(self):
        return self.input_text.content

    def set_input_text_dir(self, input_text_dir):
        self.input_text.dir = input_text_dir

    def get_input_text_dir(self):
        return self.input_text.dir

    # For output_text
    def set_output_text_name(self, output_text_name):
        self.output_text.name = output_text_name

    def get_output_text_name(self):
        return self.output_text.name

    def set_output_text_content(self, output_text_content):
        self.output_text.content = output_text_content

    def get_output_text_content(self):
        return self.output_text.content

    def set_output_text_dir(self, output_text_dir):
        self.output_text.dir = output_text_dir

    def get_output_text_dir(self):
        return self.output_text.dir

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def get_beam_size(self):
        return self.beam_size

    def set_device(self, device):
        self.output_text.dir = device

    def get_deivce(self):
        return self.device

    def load_model(self):
        model_dir = self.get_model_dir()
        try:
            self.translator = Translator(model_dir, self.device)
            if self.translator.tokenizer is None:
                print("")
                return
            if self.translator.cleaner is None:
                print("")
                return

            self.max_text_length = self.translator.config['max_len'][0]

        except:
           print("")

    def run(self):
        if self.model is None or self.translator is None:
            return

        def _translate():
            translated_texts = self.translator.translate(self.input_text.content, self.beam_size, self.device)
            if translated_texts is None:
                return

            # 输出到控制台
            for idx, text in enumerate(translated_texts):
                print(f"Option {idx + 1}: {text}")

            # 保存翻译到指定的文件
            output_path = os.path.join(self.output_text.dir, self.output_text.name)
            with open(output_path, 'w', encoding='utf-8') as f:
                for text in translated_texts:
                    f.write(text + '\n')
            print(f"Translated text saved to: {output_path}")

        _translate()


def main(args):
    cli = cli_process(model_name=args.model_name,
                      model_dir=args.model_dir,
                      input_text_name=args.input_text_name,
                      input_text_content=args.input_text_content,
                      input_text_dir=args.input_text_dir,
                      output_text_name=args.output_text_name,
                      output_text_dir=args.output_text_dir,
                      device=args.device,
                      beam_size=args.beam_size)

    cli.load_model()
    cli.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command-line interface for translation.')

    # Model arguments
    parser.add_argument('--model-name', default=cli_process.DEFAULT_PARAM_MODEL_NAME, help='Name of the model.')
    parser.add_argument('--model-dir', default=cli_process.DEFAULT_PARAM_MODEL_DIR, help='Directory of the model.')

    # Input text arguments
    parser.add_argument('--input-text-name', default=cli_process.DEFAULT_PARAM_INPUT_TEXT_NAME,
                        help='Name of the input text file.')
    parser.add_argument('--input-text-content', default=cli_process.DEFAULT_PARAM_INPUT_TEXT,
                        help='Content of the input text.')
    parser.add_argument('--input-text-dir', default=cli_process.DEFAULT_PARAM_INPUT_TEXT_DIR,
                        help='Directory of the input text file.')

    # Output text arguments
    parser.add_argument('--output-text-name', default=cli_process.DEFAULT_PARAM_OUTPUT_TEXT_NAME,
                        help='Name of the output text file.')
    parser.add_argument('--output-text-dir', default=cli_process.DEFAULT_PARAM_OUTPUT_TEXT_DIR,
                        help='Directory of the output text file.')

    # Translation arguments
    parser.add_argument('--device', default='cpu', help='Device to use for translation (e.g., cpu, cuda).')
    parser.add_argument('--beam-size', type=int, default=cli_process.DEFAULT_PARAM_BEAM_SIZE,
                        help='Size of the beam for beam search.')

    args = parser.parse_args()
    main(args)

