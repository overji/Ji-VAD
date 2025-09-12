# this script is used for generating chinese spoken text data with topics from a dataset
import json
import os
import random
from openai import OpenAI
from datasets import Dataset, load_dataset
from tqdm import tqdm

data = []
TOTAL = 2000

def generate(prompt:str):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        extra_body={"enable_thinking": False},
    )
    res_json = json.loads(completion.model_dump_json())
    text = res_json["choices"][0]["message"]["content"]
    dialog_texts = text.split("\n")
    for dt in dialog_texts:
        print(dt.strip())
        data.append({
            "text": dt.strip()
        })

ds = load_dataset("ThatsGroes/dialog-topics")
for index, item in enumerate(tqdm(ds["train"], total=TOTAL, desc="生成中")):
    if index > TOTAL:
        break
    topic = item["topic_en"]
    prompt = f"""
    你是一个生成语音数据的助手，请生成一段主题为**{topic}**的中文口语文本，要求如下:
    1. 口语内容可以包含逗号、顿号、句号，使用尽量口语化的词汇
    2. 口语内容不要过长，尽可能在一句话内完成
    3. 可以是疑问句，也可以是回答，也可以是单纯的陈述句
    4. 不要包含无法转化为语音的内容，如表格、markdown语法内容、emoji表情等
    5. 是一个完整的句子
    6. 不要在句子的结尾和中间添加语气词
    7. 虽然主题是英文，但是生成的句子必须是汉语的
    现在生成五个口语文本，一个一行，不要生成除了口语文本外的任何文字，不要给每一个句子前面标号
    """
    generate(prompt)

# 打乱data
random.shuffle(data) 
dataset = Dataset.from_list(data)

SAVE_PATH = "**YOUR_SAVE_PATH**"
dataset.save_to_disk(SAVE_PATH)