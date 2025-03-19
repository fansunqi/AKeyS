import os
import json
import pdb
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from util import *
from model import GPT
from arg_parser import parse_args
from summarizer import summarize_one_video, \
      qa_one_video_by_summary, postprocess_response_dict
from video_seg import VideoSeg, extract_videoseg_from_descriptions, split_and_reconnect_segments
from arg_parser import parse_args


set_random_seed(42)

global_args = parse_args()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = set_logger(timestamp, global_args.logger_base_path)


api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")


with open(global_args.example_summary_path,'r') as ex:
    example_summary = ex.read()

with open(global_args.example_qa_by_summary_path,'r') as ex:
    example_qa_by_summary = ex.read()


summarizer = GPT(api_key=api_key, model_name=global_args.model_name, temperature=global_args.temperature, base_url=base_url)
qa_model = GPT(api_key=api_key, model_name=global_args.model_name, temperature=global_args.temperature, base_url=base_url)
planner = GPT(api_key=api_key, model_name=global_args.model_name, temperature=global_args.temperature, base_url=base_url)
self_evaluator = GPT(api_key=api_key, model_name=global_args.model_name, temperature=global_args.temperature, base_url=base_url)



def bfs_select_segments(question, caption, num_frames, segment_des, use_cache=True):
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "3", "duration": "xxx - xxx", "description": "frame of xxx"},
        ]
    }
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial frames is not suffient.
    Objective:
    Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Divide the video into segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple frames from one segment if necessary to gather comprehensive insights. 
    Return the descriptions and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response, _ = planner.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response



def gbfs_select_one_segment(question, caption, num_frames, segment_des, use_cache=True):
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"}
        ]
    }
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial frames is not suffient.
    Objective:
    Our goal is to step-by-setp identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Consider the video segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which single segment is most likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For the segment identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of the segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Return the description and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response, _ = planner.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response



def dijkstra_select_one_segment(question, caption, num_frames, segment_des, use_cache=True):
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"}
        ]
    }

    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    
    Based on the intervals between the sample frames, we have candiate video segments: {segment_des}
    
    Please identify which candidate video segment contains richest visual elements and most dramatic scene changes, making it most suitable for splitting into smaller video segments. For example, if the characters and scenes have changed between the two sampled frames, then this video segment is suitable for splitting into smaller and atomic video segments.
    For the segment identified as most suitable for splitting into smaller video segments, provide a concise description focusing on the segment's rich visual elements or scene changes. If the specifics of the segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects.
    Return the description and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response, _ = planner.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response



def a_star_select_one_segment(question, caption, num_frames, segment_des, use_cache=True):
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"}
        ]
    }
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial frames is not suffient. 
    
    Objective:
    In order to obtain more information about the video and ultimately answer the question, we need to step-by-setp identify video segment between the initial frames that meet the following two conditions:
    1. contains crucial information necessary for answering the question. This video segment should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames. This segment should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    2. contains rich visual elements and dramatic scene changes, making it suitable for splitting into smaller video segments. For example, if the characters and scenes have changed between the two sampled frames, then this video segment is suitable for splitting into smaller and atomic video segments.

    To achieve this, we will:
    1. Consider the video segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which single candidate segment is most likely to meet the above two conditions.
    For the segment identified as most suitable, provide a concise description focusing on essential visual elements in the segment. Use a single sentence per frame. If the specifics of the segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects.
    Return the description and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response, _ = planner.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response



def self_eval(previous_prompt, answer, use_cache=True):
    confidence_format = {"confidence": "xxx"}
    prompt = f"""Please assess the confidence level in the decision-making process.
    The provided information is as as follows,
    {previous_prompt}
    The decision making process is as follows,
    {answer}
    Criteria for Evaluation:
    Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.
    Partial Information (Confidence Level: 2): If information partially supports an informed guess.
    Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.
    Assessment Focus:
    Evaluate based on the relevance, completeness, and clarity of the provided information in relation to the decision-making context.
    Please generate the confidence with JSON format {confidence_format}
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response, _ = self_evaluator.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response



def generate_answer_cot(question, caption, num_frames, use_cache=True):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response, _ = qa_model.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=False)
    return prompt, response



def generate_answer_direct(question, caption, num_frames, use_cache=True):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think carefully and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question, and you must select one answer index from the candidates.
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response, _ = qa_model.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response



def summarize_and_qa(video_id, sampled_caps, ann, args):
    summary = summarize_one_video(summarizer, video_id, sampled_caps, \
                                        example_summary, use_cache=args.use_cache, logger=logger)
    response_dict = qa_one_video_by_summary(qa_model, ann, summary, video_id, sampled_caps, \
                                            example_qa_by_summary, use_cache=args.use_cache, logger=logger)
    answer, confidnce = postprocess_response_dict(response_dict)
    return answer, confidnce



def qa_and_reflect(formatted_question, sampled_caps, num_frames, args):
    previous_prompt, answer_str = generate_answer_cot(                  
        formatted_question, sampled_caps, num_frames, args.use_cache                   
    )
    answer = parse_text_find_number(answer_str, logger)                

    confidence_str = self_eval(previous_prompt, answer_str, args.use_cache)
    confidence = parse_text_find_confidence(confidence_str, logger)

    return answer, confidence



def choose_ans(s_qa_ans, s_qa_conf, s_conf_lower, \
               r_qa_ans, r_qa_conf, r_conf_lower, \
                ans_mode, step):
    
    answer = -1
    get_ans_step = None 

    if ans_mode == "s":
        if s_qa_ans != -1 and s_qa_conf >= s_conf_lower:
            answer = s_qa_ans
            get_ans_step = f"{step}_s_qa"
    
    elif ans_mode == "r":
        if r_qa_ans != -1 and r_qa_conf >= r_conf_lower:
            answer = r_qa_ans
            get_ans_step = f"{step}_r_qa"
    
    elif ans_mode == "sr":
        if s_qa_ans != -1 and s_qa_conf >= s_conf_lower:
            answer = s_qa_ans
            get_ans_step = f"{step}_s_qa"
        
        elif r_qa_ans != -1 and r_qa_conf >= r_conf_lower:
            answer = r_qa_ans
            get_ans_step = f"{step}_r_qa"
    
    elif ans_mode == "rs":
        if r_qa_ans != -1 and r_qa_conf >= r_conf_lower:
            answer = r_qa_ans
            get_ans_step = f"{step}_r_qa"

        elif s_qa_ans != -1 and s_qa_conf >= s_conf_lower:
            answer = s_qa_ans
            get_ans_step = f"{step}_s_qa"

    elif ans_mode == "vote":
        if s_qa_ans == r_qa_ans:
            answer = s_qa_ans
            get_ans_step = f"{step}_s_r"
    
    elif (ans_mode == "vote_conf_and"):
        if (s_qa_ans == r_qa_ans) and \
           ((s_qa_conf >= s_conf_lower) and (r_qa_conf >= r_conf_lower)):
            answer = s_qa_ans
            get_ans_step = f"{step}_s_r"
    
    elif (ans_mode == "vote_conf_or"):
        if (s_qa_ans == r_qa_ans) and \
           ((s_qa_conf >= s_conf_lower) or (r_qa_conf >= r_conf_lower)):
            answer = s_qa_ans
            get_ans_step = f"{step}_s_r"

    else:
        raise KeyError
    
    return answer, get_ans_step



def select_process(formatted_question, sample_idx, sampled_caps, num_frames, step, 
                   args, all_sample_idx, caps, video_segments, select_fn):
    
    segment_des = {
        i + 1: f"{video_seg.start}-{video_seg.end}"
        for i, video_seg in enumerate(video_segments)
    }
    # segment_des: {1: '1-12', 2: '12-23', 3: '23-34', 4: '34-45', ...

    # LLM decides which segment from `segment_des` to use        
    candidate_descriptions = select_fn(formatted_question, sampled_caps, num_frames, segment_des, args.use_cache)
    if candidate_descriptions != None:
        parsed_candidate_descriptions = parse_json(candidate_descriptions)
        selected_descriptions = get_frames_descriptions(parsed_candidate_descriptions)

    # re-generate
    max_generate = 5
    generate_count = 0
    while candidate_descriptions == None or selected_descriptions == None:
        generate_count += 1
        if generate_count > max_generate:
            break
        candidate_descriptions = select_fn(formatted_question, sampled_caps, num_frames, segment_des, False)
        parsed_candidate_descriptions = parse_json(candidate_descriptions)
        selected_descriptions = get_frames_descriptions(parsed_candidate_descriptions)

    # extract `VideoSeg` instance base on selected_descriptions
    selected_video_segments = extract_videoseg_from_descriptions(selected_descriptions)

    video_segments = split_and_reconnect_segments(selected_video_segments, video_segments, args.for_seg_not_interested, num_frames)

    # extract visible frames from `selected_video_segments``
    sample_idx_set = set()
    for segment in video_segments:
        sample_idx_set.add(segment.start)  
        sample_idx_set.add(segment.end)   
    sample_idx = sorted(list(sample_idx_set))  
    
    return video_segments, sample_idx


def run_one_question(video_id, ann, caps, logs, args):

    logger.info(f"Start to process {video_id}")
    print(f"\nStart video: {video_id}")
    
    get_ans_step = None          # which step get the answer
    sample_idx_change_list = []  # change of `sample_idx`

    question = ann["question"]
    answers = [ann[f"option {i}"] for i in range(5)]
    formatted_question = (
        f"Here is the question: {question}\n"
        + "Here are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    num_frames = len(caps)

    # root node initialization
    sample_idx = list(range(1, num_frames + 1, args.init_interval))     
    sampled_caps = read_caption(caps, sample_idx)                       # e.g. {'frame 1': '#C C pours the water from the bowl', 'frame 45': '#C C puts the sponge in the sink', 'frame 90': '#C C scrubs the plate with the sponge', 'frame 135': '#C C puts the soap bottle on the sink', 'frame 180': '#C C opens the soap bottle'}
    all_sample_idx = sample_idx
    
    # video_segments initialization
    video_segments = []   
    for segment_id in range(1, len(sample_idx)):
        video_seg = VideoSeg(sample_idx[segment_id - 1], sample_idx[segment_id], segment_id, None)
        video_segments.append(video_seg)
    sample_idx_change_list.append(sample_idx)
    

    # main loop for tree search 
    # 1. LLM QA 
    # 2. node expansion
    for step in range(1, args.final_step + 1):

        # print(f"{video_id}: step {step}, sample_idx {sample_idx}")
        print_segment_list(video_segments)

        # 1. LLM QA
        if args.ans_mode == "s":
            s_qa_ans, s_qa_conf = summarize_and_qa(video_id, sampled_caps, ann, args)
            r_qa_ans, r_qa_conf = None, None
        elif args.ans_mode == "r":
            r_qa_ans, r_qa_conf = qa_and_reflect(formatted_question, sampled_caps, num_frames, args)
            s_qa_ans, s_qa_conf = None, None
        else:
            s_qa_ans, s_qa_conf = summarize_and_qa(video_id, sampled_caps, ann, args)
            r_qa_ans, r_qa_conf = qa_and_reflect(formatted_question, sampled_caps, num_frames, args)

        answer, get_ans_step = choose_ans(s_qa_ans, s_qa_conf, args.s_conf_lower, \
                                          r_qa_ans, r_qa_conf, args.r_conf_lower, \
                                          args.ans_mode, step)

        if answer != -1:
            break 
        
        # 2. node expansion
        if args.search_strategy == "bfs": 
            select_fn = bfs_select_segments
            video_segments, sample_idx = \
                select_process(formatted_question, sample_idx, sampled_caps, num_frames, 
                               step, args, all_sample_idx, caps, video_segments, select_fn)
        
        elif args.search_strategy == "gbfs": 
            select_fn = gbfs_select_one_segment
            for select_iter in range(args.beam_size):
                video_segments, sample_idx = \
                    select_process(formatted_question, sample_idx, sampled_caps, num_frames, 
                                   step, args, all_sample_idx, caps, video_segments, select_fn)
                
        elif args.search_strategy == "dijkstra":
            select_fn = dijkstra_select_one_segment
            for select_iter in range(args.beam_size):
                video_segments, sample_idx = \
                    select_process(formatted_question, sample_idx, sampled_caps, num_frames, 
                                   step, args, all_sample_idx, caps, video_segments, select_fn)
                
        elif args.search_strategy == "a_star":
            select_fn = a_star_select_one_segment
            for select_iter in range(args.beam_size):
                video_segments, sample_idx = \
                    select_process(formatted_question, sample_idx, sampled_caps, num_frames, 
                                   step, args, all_sample_idx, caps, video_segments, select_fn)
        else:
            raise KeyError
            
        # aggregate all `sample_idx` to `all_sample_idx`
        all_sample_idx = sorted(list(set(all_sample_idx + sample_idx)))
        sample_idx_change_list.append(sample_idx)
        sampled_caps = read_caption(caps, sample_idx)

    # print(video_id, "final sample num:", len(sample_idx))

    # Post Process
    if answer == -1:
        if args.post_resume_samples:
            sample_idx = list(range(1, num_frames + 1, args.init_interval))
        else:
            sample_idx = all_sample_idx

        sample_idx_change_list.append(sample_idx)
        sampled_caps = read_caption(caps, sample_idx) 


        if args.post_ans_mode == "s":
            s_qa_ans, s_qa_conf = summarize_and_qa(video_id, sampled_caps, ann, args)
            r_qa_ans, r_qa_conf = None, None
        elif args.post_ans_mode == "r":
            r_qa_ans, r_qa_conf = qa_and_reflect(formatted_question, sampled_caps, num_frames, args)
            s_qa_ans, s_qa_conf = None, None
        else:
            s_qa_ans, s_qa_conf = summarize_and_qa(video_id, sampled_caps, ann, args)
            r_qa_ans, r_qa_conf = qa_and_reflect(formatted_question, sampled_caps, num_frames, args)
        step = "post"
        answer, get_ans_step = choose_ans(s_qa_ans, s_qa_conf, args.post_s_conf_lower, \
                                          r_qa_ans, r_qa_conf, args.post_r_conf_lower, \
                                          args.post_ans_mode, step)

    # final direct QA
    if answer == -1:
        answer_str = generate_answer_direct(
            formatted_question, sampled_caps, num_frames, args.use_cache
        )
        answer = parse_text_find_number(answer_str, logger)
        get_ans_step = f"final_direct_qa"

    # no_ans
    if answer == -1:
        logger.info(f"No answer video id: {video_id}")                     
        print(f"\nNo answer video id: {video_id}\n")


    # print_nested_list(sample_idx_change_list)
    logger.info(f"Finished video: {video_id}/{answer}/{ann['truth']}")
    print(f"Finished video: {video_id}/{answer}/{ann['truth']}\n")


    label = int(ann["truth"])
    corr = int(label == answer)
    count_frame = len(all_sample_idx)           

    logs[video_id] = {
        "answer": answer,
        "label": label,
        "corr": corr,
        "count_frame": count_frame,
        "get_ans_step": get_ans_step,
    }


def main(args):

    output_result_file = os.path.join(args.output_base_path, f"{timestamp}.json")

    anns = json.load(open(args.anno_path, "r"))
    all_caps = json.load(open(args.cap_path, "r"))
    logs = {}

    process_video_ids = list(anns.keys())[:args.process_num]

    if args.specific_id != None:  
        specific_video_ids = [args.specific_id]
        process_video_ids = get_intersection(specific_video_ids, list(anns.keys()))
    
    if args.reprocess_log != None:
        # Load the log file and find video_ids with answer == -1
        reprocess_log = json.load(open(args.reprocess_log, "r"))
        process_video_ids = [
            video_id for video_id, log in reprocess_log.items() 
            # if log["answer"] == -1
            if log["answer"] != log["label"]
        ]
    
    if args.avoid_id != None:
        process_video_ids.remove(args.avoid_id)
    
    if args.specific_id_path != None:
        process_video_ids = json.load(open(args.specific_id_path, "r"))


    logger.info(f"{len(process_video_ids)} videos to process")
                                                                                  
    tasks = [
        (video_id, anns[video_id], all_caps[video_id], logs, args)
        for video_id in process_video_ids
    ]

    # parallel processing
    # with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    #     for _ in tqdm(executor.map(lambda p: run_one_question(*p), tasks), total=len(tasks), desc="Processing"):
    #         pass
    
    for task in tqdm(tasks):
        try:
            run_one_question(*task)
        except Exception as e:
            print(f"\nError -- main -- {e}\n")

    json.dump(logs, open(output_result_file, "w"))


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    main(args)
    
    print(args)
    print(f"\ntimestamp: {timestamp}\n")

    # eval
    os.system(f"python3 eval.py results/{args.dataset}/{timestamp}.json")