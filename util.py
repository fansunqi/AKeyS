import re
import pickle
import json
from pathlib import Path
from pprint import pprint
import os
import numpy as np
import random
import logging
import pdb
from arg_parser import parse_args


args = parse_args()
cache_path = args.cache_path


# load cache
if os.path.exists(cache_path):
    cache_llm = pickle.load(open(cache_path, "rb"))
else:
    cache_llm = {}


def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_video_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            filenames.append(os.path.splitext(filename)[0])
    return filenames


def get_intersection(list_a, list_b):
    return list(set(list_a) & set(list_b))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if you are using GPU
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # import tensorflow as tf
    # tf.random.set_seed(seed)

def get_frames_descriptions(parsed_candidate_descriptions):
    if parsed_candidate_descriptions == None:
        return None
    
    if isinstance(parsed_candidate_descriptions, list) and len(parsed_candidate_descriptions) > 0:
        parsed_candidate_descriptions = parsed_candidate_descriptions[0]
    
    if not isinstance(parsed_candidate_descriptions, dict):
        return None

    if "frame_descriptions" in parsed_candidate_descriptions:
        frames_descriptions = parsed_candidate_descriptions["frame_descriptions"]
        return frames_descriptions
    # elif "descriptions" in parsed_candidate_descriptions:
    #     frames_descriptions = parsed_candidate_descriptions["descriptions"]
    # elif "description" in parsed_candidate_descriptions:
    #     frames_descriptions = parsed_candidate_descriptions["description"]
    else:
        print(f"\nERROR --util.get_frames_descriptions--: {parsed_candidate_descriptions}\n")
        # if parsed_candidate_descriptions == None:
        #     print("\nparsed_candidate_descriptions is None\n")
        # raise KeyError
        return None


# Add more robust error handling
def parse_json(text, video_id=None):

    if text == None:
        print(f"{video_id}: No valid JSON found in the text {text}")
        return None


    pattern = r'```json\n({.*?})\n```'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        text = match.group(1)

    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON

        # Pattern for JSON objects and arrays
        json_pattern = r"\{.*?\}|\[.*?\]"  

        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        print(f"{video_id}: No valid JSON found in the text {text}")
        return None


def get_segment_id(description):
    for key in description.keys():
        if key.lower() == "segment_id":
            return int(description[key])
    return None


def get_duration(description):
    for key in description.keys():
        if key.lower() == "duration":
            return description[key]
    return None


def get_value_from_dict(d):
    if isinstance(d, dict) and len(d) == 1:
        key, value = next(iter(d.items()))
        return value
    return None


# Add more robust error handling
def parse_text_find_number(text, logger):
    item = parse_json(text)
    try:
        match = int(get_value_from_dict(item))
        if match in range(-1, 5):
            return match
        else:
            return random.randint(0, 4)
    except Exception as e:
        logger.error(f"Answer Parsing Error: {e}")
        # pdb.set_trace()
        return -1

def print_nested_list(nested_list):
    print("\n")
    for one_list in nested_list:
        print(one_list)
    print("\n") 


def print_segment_list(video_segments):
    for seg in video_segments:
        print(f"[{seg.start} - {seg.end}] ", end='')
    print() 


def parse_text_find_confidence(text, logger):
    item = parse_json(text)
    try:
        match = int(item["confidence"])
        if match in range(1, 4):
            return match
        else:
            return 1
    except Exception as e:
        logger.error(f"Confidence Parsing Error: {e}")
        return 1


def read_caption(captions, sample_idx):
    video_caption = {}
    for idx in sample_idx:
        video_caption[f"frame {idx}"] = captions[idx - 1]
    return video_caption


def get_from_cache(key, logger=None, use_logger=True):
    try:
        return cache_llm[key.encode()].decode()
    except KeyError:
        pass
    except Exception as e:
        if use_logger:
            logger.warning(f"Error getting from cache: {e}")
    return None


def save_to_cache(key, value, logger=None, use_logger=True):
    try:
        cache_llm[key.encode()] = value.encode()
        pickle.dump(cache_llm, open(cache_path, "wb"))
    except Exception as e:
        if use_logger:
            logger.warning(f"Error saving to cache: {e}")


def set_logger(timestamp, logger_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger_file_path = os.path.join(logger_path, f"{timestamp}.log")
    file_handler = logging.FileHandler(logger_file_path)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)"
    )
    file_handler.setFormatter(formatter)
   
    # remove terminal output
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(file_handler)
    
    return logger
    
    