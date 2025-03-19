import openai
from openai import OpenAI
import pdb
from pprint import pprint
import time
import json
from util import get_from_cache, save_to_cache

LOG_MAX_LENGTH = 300

# one example for post_process_fn
def identity(res):
    return res


def get_model(args):
    # choose different LLM, e.g., GPT, TogetherAI
    model_name, temperature = args.model, args.temperature
    base_url = args.openai_proxy if hasattr(args, 'openai_proxy') else None
    print('base_url: ', base_url)
    
    if 'gpt' in model_name:
        model = GPT(args.api_key, model_name, temperature, base_url)
        return model
    else:
        raise KeyError(f"Model {model_name} not implemented")


class Model(object):
    def __init__(self):
        self.post_process_fn = identity
    
    def set_post_process_fn(self, post_process_fn):
        self.post_process_fn = post_process_fn


class GPT(Model):
    def __init__(self, api_key, model_name, temperature=1.0, base_url=None):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.badrequest_count = 0
        openai.api_key = api_key
        if base_url:
            openai.base_url = base_url
        self.client = OpenAI(api_key = api_key, base_url = base_url)
        
    # used in forward()
    def get_response(self, **kwargs):
        try:
            # res = openai.chat.completions.create(**kwargs)
            res = self.client.chat.completions.create(**kwargs)
            return res
        except openai.APIConnectionError as e:
            print('APIConnectionError')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.APIConnectionError as err:
            print('APIConnectionError')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.RateLimitError as e:
            print('RateLimitError')
            time.sleep(10)
            return self.get_response(**kwargs)
        except openai.APITimeoutError as e:
            print('APITimeoutError')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.BadRequestError as e:
            print('BadRequestError')
            self.badrequest_count += 1
            print('badrequest_count', self.badrequest_count)
            return None


    def forward(self, head=None, prompt=None, use_cache=True, logger=None, forward_use_logger=True, use_json_format=False):
        messages = []
        info = {} 

        if logger == None:
            forward_use_logger = False   

        if head != None:
            messages.append(
                {"role": "system", "content": head}
            )
        messages.append(
            {"role": "user", "content": prompt}
        )

        key = json.dumps([self.model_name, messages])
        
        if forward_use_logger == True:
            logger.info(f"Messages: {str(messages)[:LOG_MAX_LENGTH]}")
        
        if use_cache:
            cached_value = get_from_cache(key, logger, forward_use_logger)
            if cached_value is not None:
                if forward_use_logger == True:
                    logger.info("Cache Hit")
                else:
                    print("Cache Hit")
                return cached_value, None
        
        if forward_use_logger:
            logger.info("Cache Miss")
        else:
            print("Cache Miss")

        if use_json_format:
            response = self.get_response(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        else:
            response = self.get_response(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )

        if response is None:
            info['response'] = None
            info['message'] = None
            return None, info
        else:
            messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
            info = dict(response.usage)  # completion_tokens, prompt_tokens, total_tokens
            info['response'] = messages[-1]["content"]
            info['message'] = messages

            if forward_use_logger:
                logger.info(f"Response: {str(info['response'])[:LOG_MAX_LENGTH]}")
            # if use_cache:
            save_to_cache(key, info['response'], logger, forward_use_logger)
            if forward_use_logger:
                logger.info("Cache Saved")
            else:
                print("Cache Miss")

            return self.post_process_fn(info['response']), info