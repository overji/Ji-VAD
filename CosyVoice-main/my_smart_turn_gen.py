import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import json
from tqdm import tqdm
import random
from datasets import load_from_disk, Dataset

import os
os.environ["MODELSCOPE_CACHE"] = "/obs/xuke/modelscope_cache"
cosyvoice = CosyVoice2('/obs/xuke/cosyvoicepretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# 加载说话人
speaker_base_path = "/obs/xuke/audio_source"
with open(f"{speaker_base_path}/real_audio.json","r",encoding="utf-8") as f:
    real_audio = json.load(f)

ds = load_from_disk("/obs/xuke/smart-turn/v2_llm_gen_topic_cn_filler")
save_path = "/obs/xuke/smart-turn/v2_llm_gen_topic_cn_filler_audio/audio_data"

# 加载现在的数据集
# cur_ds = load_from_disk("/obs/xuke/smart-turn/llm_gen_topic_cn_punc_audio/temp_2500")

final_result = []

for index, item in tqdm(enumerate(ds), total=len(ds)):
    speaker = random.choice(real_audio)
    prompt_speech_16k = load_wav(f"{speaker_base_path}/{speaker['filename']}", 16000)
    text = item['text']
    data_type = 0
    if item["endpoint_bool"]:
        data_type += 1
    if item["midfiller"]:
        data_type += 2
    if item["endfiller"]:
        data_type += 4
    save_dir = f"{save_path}/dt_{data_type}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dir_file_name = f"dt_{data_type}/audio_{index}.wav"
    for i, j in enumerate(cosyvoice.inference_zero_shot(text, speaker["text"], prompt_speech_16k, stream=False)):
        torchaudio.save(f'{save_path}/{dir_file_name}', j['tts_speech'], cosyvoice.sample_rate)
    final_result.append({
        "text": text,
        "audio": dir_file_name,
        "endpoint_bool": item["endpoint_bool"],
        "midfiller": item["midfiller"],
        "endfiller": item["endfiller"]
    })
    # 每500条保存一次临时进度
    if (index + 1) % 500 == 0:
        temp_dataset = Dataset.from_list(final_result)
        temp_dataset.save_to_disk(f"/obs/xuke/smart-turn/v2_llm_gen_topic_cn_filler_audio/temp_{index+1}")
        print(f"已保存 {index+1} 条数据到临时文件。")

# 最后保存全部数据
audio_dataset = Dataset.from_list(final_result)
audio_dataset.save_to_disk("/obs/xuke/smart-turn/v2_llm_gen_topic_cn_filler_audio")