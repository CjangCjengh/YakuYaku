import argparse

from utils import ChineseConverter



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert between Simplified and Traditional Chinese in text files.")
    parser.add_argument("input_file_path", type=str, help="Path to the input file.")
    parser.add_argument("output_file_path", type=str, help="Path to the output file.")
    parser.add_argument("--conversion_type", type=str, choices=["s2t", "t2s"], default="s2t",
                        help="Conversion type. 's2t' for Simplified to Traditional (default). 't2s' for Traditional to Simplified.")

    args = parser.parse_args()

    converter = ChineseConverter()
    converter.convert_file(args.input_file_path, args.output_file_path, args.conversion_type)
