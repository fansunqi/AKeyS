# Temporal Summarizer
# Adapted from https://github.com/Hyu-Zhang/HCQA/
import os
import re
import json
import time
import pdb
import ast
import openai
from tqdm import tqdm
from model import GPT
from datetime import datetime
from util import parse_json


systerm = '''
You're a visual summary expert. You can accurately make a [SUMMARY] based on [CAPTION], where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.
'''

incontext_prompt = '''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each caption represents a frame in the video in chronological order, although they may not be consecutive. And each description is separated by a newline character. At the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
[SUMMARY]: Based on the CAPTIONS of these video clips, you need to summarise them into an overall description of the video, in chronological order.
I will give you an example as follow:
<Example>
{example}
Now, you should make a [SUMMARY] based on the [CAPTION] below. You SHOULD follow the format of example.
[CAPTION]
{caption}
[SUMMARY]
'''

systerm_qa='''
You are a visual question answering expert. You can choose the correct answer from five options of [OPTION] based on the [CAPTION], [SUMMARY], [QUESTION], and [REASON]. Where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.
'''

incontext_prompt_qa='''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each caption represents a frame in the video in chronological order, although they may not be consecutive. And each description is separated by a newline character. At the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
[SUMMARY]: Based on the CAPTIONS of these video clips, an overall description of the video, in chronological order.
[QUESTION]: A question about video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [SUMMARY], reasoning step by step to get the answer. If [SUMMARY] doesn't have enough information, you need to get it from the [CAPTION].
I will give you some examples as follow:
{example}
Now, you should first make a [REASON] based on [QUESTION] and [SUMMARY], then give right number of [OPTION] as [ANSWER] . Additionally, you need to give me [CONFIDENCE] that indicates your confidence in answering the question accurately, on a scale from 1 to 5. You SHOULD answer the question, even given a low confidence.
[CAPTION]
{caption}
[SUMMARY]
{summary}
[QUESTION]
{question}
[OPTION]
{option}
'''

response_prompt='''
YOU MUST output in the JSON format.
{
"REASON":"[REASON]",
"ANSWER":"[ANSWER]",
"CONFIDENCE": "[CONFIDENCE]"
}
'''


# example_summary_path = "data/egoschema/example_summary.txt"
# with open(example_summary_path,'r') as ex:
#     example_summary = ex.read()

# example_qa_by_summary_path = "data/egoschema/example_qa_by_summary.txt"
# with open(example_qa_by_summary_path,'r') as ex:
#     example_qa_by_summary = ex.read()


def summarize_one_video(summarizer, video_id, sample_caps, \
                        example_summary, use_cache, logger):
    caps = ''
    for _, c in sample_caps.items():
        caps += c + "\n"
    instruction = str(video_id) + "\n" + systerm + "\n" + incontext_prompt.format(example = example_summary, caption = caps)
    response, info = summarizer.forward(head = None, prompt = instruction, \
                                        use_cache=use_cache, logger=logger)
    return response


def qa_one_video_by_summary(qa_model, ann, summary, video_id, sample_caps, \
                            example_qa_by_summary, use_cache, logger):
    
    caps = ''
    for _, c in sample_caps.items():
        caps += c + "\n"

    opt = ''
    que = 'question: ' + ann['question']
    opt += 'option 0: ' + ann['option 0'] + "\n"
    opt += 'option 1: ' + ann['option 1'] + "\n"
    opt += 'option 2: ' + ann['option 2'] + "\n"
    opt += 'option 3: ' + ann['option 3'] + "\n"
    opt += 'option 4: ' + ann['option 4'] + "\n"

    instruction = str(video_id) + "\n" + systerm_qa + "\n" + incontext_prompt_qa.format(
        example = example_qa_by_summary, caption = caps, summary = summary, question = que, option = opt
        ) + "\n" + response_prompt
    response, info = qa_model.forward(head=None, prompt=instruction, \
                                      use_cache=use_cache, logger=logger)

    response_dict = None
    try: 
        response_dict = ast.literal_eval(response)
    except:
        response_dict = parse_json(response, video_id)
    
    # if response_dict == None:
    #     pdb.set_trace()

    return response_dict


def postprocess_response_dict(response_dict):

    if (response_dict == None) or \
        ("ANSWER" not in response_dict) or \
        ("CONFIDENCE" not in response_dict):
        return -1, -1
    
    try:
        id_value = int(response_dict["ANSWER"])
        if id_value < 0 or id_value > 4:
            id_value = -1
        conf = int(response_dict["CONFIDENCE"])
    except:
        pattern = r'\d+'
        match = re.search(pattern, str(response_dict["ANSWER"]))
        conf = re.search(pattern, str(response_dict["CONFIDENCE"]))
        if match:
            id_value = int(match.group())
            conf = int(conf.group())
        else:
            id_value = -1
            conf = -1

    return id_value, conf
