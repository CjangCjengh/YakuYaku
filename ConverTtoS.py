"""
传入参数:
    --input-file-path:需要翻译的文件的路径+名字
    --output-fie-path:翻译之后文件的路径+名字
    --conversion-type:翻译类型,t2s:繁体转简体
                            s2t:简体转繁体


"""


import argparse

from utils import ChineseConverter



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert between Simplified and Traditional Chinese in text files.")
    parser.add_argument("--input-file-path", type=str, help="Path to the input file.")
    parser.add_argument("--output-file-path", type=str, help="Path to the output file.")
    parser.add_argument("--conversion-type", type=str, choices=["s2t", "t2s"], default="t2s",
                        help="Conversion type. 's2t' for Simplified to Traditional (default). 't2s' for Traditional to Simplified.")

    args = parser.parse_args()

    converter = ChineseConverter()
    converter.convert_file(args.input_file_path, args.output_file_path, args.conversion_type)
