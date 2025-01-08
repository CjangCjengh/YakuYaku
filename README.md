# 此变体：
添加XPU（Intel-GPU）支持

修改API默认使用llama.cpp默认值

合并[添加翻译进度条支持](https://github.com/CjangCjengh/YakuYaku/pull/11)

基于[YakuYaku-llm ae8ca20](https://github.com/CjangCjengh/YakuYaku/commit/ae8ca203b41b4ad1693109e31883f67e1f8d606f)修改

# 部署 for Windows
## 安装[显卡驱动](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

## 安装[Intel显卡torch开发包0.5.3.37](_https://registrationcenter-download.intel.com/akdlm/IRC_NAS/9d1a91e2-e8b8-40a5-8c7f-5db768a6a60c/w_intel-for-pytorch-gpu-dev_p_0.5.3.37_offline.exe)

## 活化开发包环境
```
"C:\Program Files (x86)\Intel\oneAPI\pytorch-gpu-dev-0.5\oneapi-vars.bat"
"C:\Program Files (x86)\Intel\oneAPI\ocloc\2024.2\env\vars.bat"
```

## 安装Pytroch
预览版
```
pip3 install torch --index-url https://download.pytorch.org/whl/test/xpu
```
夜间版
```
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu
```

## 安装其他依赖
```
pip install -r requirements.txt
```
# 启动
```
call "C:\Program Files (x86)\Intel\oneAPI\pytorch-gpu-dev-0.5\oneapi-vars.bat"
call "C:\Program Files (x86)\Intel\oneAPI\ocloc\2024.2\env\vars.bat"

python YakuYaku.py
```
## Models
[台版轻小说翻译模型v1.2（日本語 → 繁體中文）](https://drive.google.com/file/d/1eUh7J6WOEujLrQSBO1gV6tpbLMvzIkRF/view?usp=sharing)

[台版轻小说翻译模型（large）v1.0（日本語 → 繁體中文）](https://huggingface.co/CjangCjengh/NMT-ja2zh_2)

[A better practice](https://huggingface.co/sakuraumi/Sakura-13B-Galgame) on the same dataset produced by [@SakuraUmi](https://github.com/pipixia244)

[台版轻小说翻译模型v1.0（繁體中文 → 日本語）](https://drive.google.com/file/d/1PJRP5ucEeicvc-p7cXwaTWOU3mU4ozXt/view?usp=sharing)

[轻小说翻译模型（测试版）（日本語 → 한국어）](https://drive.google.com/file/d/1-wvmBLPzqbUM9iECAoWkBUJtVIp27GFm/view?usp=sharing)

## Sakura-13B-LNovel
Run `server.py` in [SakuraLLM/Sakura-13B-Galgame](https://github.com/SakuraLLM/Sakura-13B-Galgame/tree/dev_server)
```sh
python server.py \
    --listen "127.0.0.1:5000" \
    --model_name_or_path "SakuraLLM/Sakura-13B-LNovel-v0.8" \
    --trust_remote_code \
    --model_version 0.8 \
    --use_gptq_model \
    --no-auth
```
