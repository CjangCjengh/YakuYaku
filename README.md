## Models
[台版轻小说翻译模型v1.2（日本語 → 繁體中文）](https://drive.google.com/file/d/1eUh7J6WOEujLrQSBO1gV6tpbLMvzIkRF/view?usp=sharing)

[台版轻小说翻译模型（large）v1.0（日本語 → 繁體中文）](https://huggingface.co/CjangCjengh/NMT-ja2zh_2)

[A better practice](https://huggingface.co/sakuraumi/Sakura-13B-Galgame) on the same dataset produced by [@SakuraUmi](https://github.com/pipixia244)

[台版轻小说翻译模型v1.0（繁體中文 → 日本語）](https://drive.google.com/file/d/1PJRP5ucEeicvc-p7cXwaTWOU3mU4ozXt/view?usp=sharing)

[轻小说翻译模型（测试版）（日本語 → 한국어）](https://drive.google.com/file/d/1-wvmBLPzqbUM9iECAoWkBUJtVIp27GFm/view?usp=sharing)

## Other Practices

[LN-Korean-14B-v0.2](https://huggingface.co/CjangCjengh/LN-Korean-14B-v0.2) : Sakura-14B-Qwen2beta-Base-v2 finetuned for（한국어 → 简体中文）

[LN-Thai-14B-v0.1](https://huggingface.co/CjangCjengh/LN-Thai-14B-v0.1) : Sakura-14B-Qwen2beta-Base-v2 finetuned for（ภาษาไทย → 简体中文）

[LN-ko2ja-14B-v0.1](https://huggingface.co/CjangCjengh/LN-ko2ja-14B-v0.1) : Sakura-14B-Qwen2beta-Base-v2 finetuned for（한국어 → 日本語）

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
