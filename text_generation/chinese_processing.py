# this file is used to segment chinese sentences for tts model training
import jieba
import random
from datasets import load_from_disk, Dataset

punctuations = [",",".","?","!","，","。","！","？"]
data = []

def segment_sentence(sentence):
    words = list(jieba.cut(sentence))
    if len(words) < 4:
        return sentence
    start_loc = len(words)//2
    end_loc = len(words)*3//4
    cut_loc = random.randint(start_loc, end_loc)
    if sentence[cut_loc-1] in punctuations:
        return sentence
    else:
        return "".join(words[:cut_loc])
    
DATA_PATH = "**YOUR_DATA_PATH**"
dataset = load_from_disk(DATA_PATH)
for index,item in enumerate(dataset):
    sentence = item["text"]
    segmented = segment_sentence(sentence)
    print(f"Original: {sentence}")
    print(f"Segmented: {segmented}")
    data.append({"original": sentence, "segmented": segmented})
dataset = Dataset.from_list(data)

SAVE_PATH = "**YOUR_SAVE_PATH**"
dataset.save_to_disk(SAVE_PATH)