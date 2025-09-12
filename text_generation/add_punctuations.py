# this script is used to add punctuations and fillers to the end of sentences in a dataset
# 本文件用于直接添加语气词
import random
from datasets import load_from_disk,Dataset
import jieba

LLM_TOPIC_PATH = "**YOUR_PATH**"

dataset = load_from_disk(LLM_TOPIC_PATH)
punctuations = [",",".","?","!","，","。","！","？"]
think_punctuations = ["嗯...","那个...","这个...","呃...","啊...","额...","唔..."]
end_no_punc_data = []
end_mid_filler_data = []
unend_mid_filler_data = []
unend_end_filler_data = []
unend_both_filler_data = []

def segment_sentence(sentence):
    words = list(jieba.cut(sentence))
    if len(words) <= 4:
        return sentence
    if len(words) < 10:
        start_loc = len(words)//2
        end_loc = len(words)*3//4
    else:
        start_loc = len(words)*3//4
        end_loc = len(words)*9//10
    cut_loc = random.randint(start_loc, end_loc)
    if sentence[cut_loc-1] in punctuations:
        return sentence
    else:
        return "".join(words[:cut_loc])

def add_midfiller(sentence):
    words = list(jieba.cut(sentence))
    if len(words) < 4:
        return sentence
    filler = random.choice(think_punctuations)
    insert_loc = random.randint(1, len(words)-2)
    if words[insert_loc] in punctuations:
        return sentence
    words.insert(insert_loc, filler)
    return "".join(words)

def add_endfiller(sentence):
    filler = random.choice(think_punctuations)
    words = list(jieba.cut(sentence))
    words.append(filler)
    return "".join(words)

def process_end_no_punc(sentence):
    end_no_punc_data.append({
        "text":sentence,
        "endpoint_bool": True,
        "midfiller": False,
        "endfiller": False
    })

def process_end_mid_filler(sentence):
    new_sentence = add_midfiller(sentence)
    if new_sentence == sentence:
        process_end_no_punc(sentence)
    else:
        end_mid_filler_data.append({
            "text": new_sentence,
            "endpoint_bool": True,
            "midfiller": True,
            "endfiller": False
        })

def process_end(sentence):
    end_prob = random.random()
    if end_prob < 0.49:
        process_end_no_punc(sentence)
    else:
        process_end_mid_filler(sentence)

def process_unend_end_filler(seg_sentence):
    new_sentence = add_endfiller(seg_sentence)
    unend_end_filler_data.append({
        "text": new_sentence,
        "endpoint_bool": False,
        "midfiller": False,
        "endfiller": True
    })

def process_unend_mid_filler(seg_sentence):
    new_sentence = add_midfiller(seg_sentence)
    if new_sentence == seg_sentence:
        process_unend_end_filler(seg_sentence)
    else:
        add_end_prob = random.random()
        if add_end_prob < 0.5:
            unend_mid_filler_data.append({
                "text": new_sentence,
                "endpoint_bool": False,
                "midfiller": True,
                "endfiller": False
            })
        else:
            new_sentence = add_endfiller(new_sentence)
            unend_both_filler_data.append({
                "text": new_sentence,
                "endpoint_bool": False,
                "midfiller": True,
                "endfiller": True
            })

def process_unend(sentence):
    seg_sentence = segment_sentence(sentence)
    if(seg_sentence == sentence):
        process_end(sentence)
    else:
        mid_prob = random.random()
        if mid_prob < 0.33:
            # 这部分直接添加endfiller
            process_unend_end_filler(seg_sentence)
        else:
            process_unend_mid_filler(seg_sentence)


end_no_punc = 0.25
end_mid_filler = 0.5
unend_mid_filler = 0.625
unend_end_filler = 0.875
unend_both_filler = 1.0

for index,item in enumerate(dataset):
    text = item['text']
    if text == '' or text == ' ':
        continue
    random_prob = random.random()
    if random_prob < 0.45:
        process_end(text)
    else:
        process_unend(text)

print("end_no_punc:", len(end_no_punc_data))
print("end_mid_filler:", len(end_mid_filler_data))
print("unend_mid_filler:", len(unend_mid_filler_data))
print("unend_end_filler:", len(unend_end_filler_data))
print("unend_both_filler:", len(unend_both_filler_data))

print("\nend:",len(end_no_punc_data)+len(end_mid_filler_data))
print("unend:", len(unend_mid_filler_data)+len(unend_end_filler_data)+len(unend_both_filler_data))

# 每个各自随即打印十个，作为参考
print("end no filler:")
for item in end_no_punc_data[:10]:
    print(item)

print("end mid filler")
for item in end_mid_filler_data[:10]:
    print(item)

print("unend mid filler")
for item in unend_mid_filler_data[:10]:
    print(item)

print("unend end filler")
for item in unend_end_filler_data[:10]:
    print(item)

print("unend both filler")
for item in unend_both_filler_data[:10]:
    print(item)

full_data = end_no_punc_data + end_mid_filler_data + unend_mid_filler_data + unend_end_filler_data + unend_both_filler_data
ds = Dataset.from_list(full_data)

SAVE_PATH = "**YOUR_SAVE_PATH**"
ds.save_to_disk(SAVE_PATH)