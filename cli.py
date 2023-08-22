"""
该代码实现了一个命令行界面 (CLI)，用于执行文本翻译任务。

参数(args)说明:
    --model-name: 模型的名称，默认为 'value1'。
    --model-dir: 存放模型的目录，默认为 './models'。
    --input-text-name: 输入文本的文件名，默认为 'input.txt'。
    --input-text-content: 输入文本的内容，默认为 'input_text'。
    --input-text-dir: 输入文本的目录，默认为 './inputs'。
    --output-text-name: 输出文本的文件名，默认为 'output.txt'。
    --output-text-dir: 输出文本的目录，默认为 './outputs'。
    --device: 用于翻译的设备 (例如, 'cpu', 'cuda')，默认为 'cpu'。
    --beam-size: beam搜索的大小，默认为 4。

函数解释:
1. Text: 代表文本的类，有三个属性：文件名(name)、内容(content)和目录(dir)。
2. Model: 代表模型的类，有两个属性：名称(name)和目录(dir)。
3. cli_process: 主处理类，其中:
    - 初始化 (__init__): 创建模型、输入文本和输出文本的实例，并初始化翻译器。
    - get_model_path: 获取模型的完整路径。
    - get_input_text_path: 获取输入文本的完整路径。
    - get_output_text_path: 获取输出文本的完整路径。
    - load_model: 加载翻译模型。
    - run: 执行翻译并保存结果。

4. main: 主函数，用于从命令行接收参数，初始化 cli_process 类并执行翻译。

如何运行:
你可以通过以下的方式执行代码:
```bash
python cli.py --model-name=LNTW_ja2zh --model-dir=./models --input-text-name="test.txt"
--input-text-dir="./" --output-text-name="output_test.txt" --output-text-dir="./output" --device=cuda
"""


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
    DEFAULT_PARAM_BEAM_SIZE = 4
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
        self.translator = None;

    def get_model_path(self):
        return os.path.join(self.model.dir, self.model.name)

    def get_input_text_path(self):
        file_path = os.path.join(self.input_text.dir, self.input_text.name)
        return file_path

    def get_output_text_path(self):
        file_path = os.path.join(self.output_text.dir, self.output_text.name)
        return file_path

    def load_model(self):
        model_dir = self.get_model_path()
        try:
            self.translator = Translator(model_dir, self.device)
            if self.translator.tokenizer is None:

                print("Error loading model: None tokenizer")

            return
            if self.translator.cleaner is None:
                print("Error loading model: None cleaner")
                return

            self.max_text_length = self.translator.config['max_len'][0]


        except Exception as e:
            print(f"Error loading model: {e}")

    def run(self):
        if self.model is None or self.translator is None:
            return

        def _translate():
            origin_text_dir = self.get_input_text_path()
            output_text_dir = self.get_output_text_path()
            translated_texts = self.translator.translate_file(file=origin_text_dir,output=output_text_dir, beam_size = self.beam_size, device=self.device)
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

