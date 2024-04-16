import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LogitsProcessorList
from RobustWatermark.watermark import WatermarkLogitsProcessor, WatermarkWindow, WatermarkContext
import argparse
import os
from transformers import LlamaTokenizer , AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pickle
import subprocess
import sys
import pandas as pd
import io


def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_stats = gpu_stats.decode('utf-8')
    gpu_df = pd.read_csv(io.StringIO(gpu_stats))
    gpu_df["memory.free"] = gpu_df[' memory.free [MiB]']
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]')).astype('float32')
    idx = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx
    
def detect_watermark(args ):
 

  
    device = torch.device("cuda:" + str(get_free_gpu()) if torch.cuda.is_available() else "cpu")

    model_path = args.llm_path
    
   # model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
       # model.config.pad_token_id = model.config.eos_token_id

    if args.watermark_type == "window": # use a window of previous tokens to hash, e.g. KGW
        watermark_model = WatermarkWindow(device, args.window_size, tokenizer)
        logits_processor = WatermarkLogitsProcessor(watermark_model)
        
        
    elif args.watermark_type == "context":
        print("Hello")
        watermark_model = WatermarkContext(device, args.chunk_size, tokenizer, delta = args.delta,transform_model_path=args.transform_model, embedding_model=args.embedding_model)
        logits_processor = WatermarkLogitsProcessor(watermark_model)
    else:
        watermark_model, logits_processor = None, None

    with torch.no_grad():
        z_score_generated = watermark_model.detect("To calculate true positives, true negatives, false positives, and false negatives, you can use a confusion matrix.") if watermark_model else 0
       # z_score_origin = watermark_model.detect(original_text_file) if watermark_model else 0
       # z_score_attacked = watermark_model.detect(attacked_text) if watermark_model else 0
        print(z_score_generated)




def generate_watermark_sir(args):

    device = torch.device("cuda:" + str(get_free_gpu()) if torch.cuda.is_available() else "cpu")

    model_path = args.llm_path
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    if args.watermark_type == "window": # use a window of previous tokens to hash, e.g. KGW
        watermark_model = WatermarkWindow(device, args.window_size, tokenizer)
        logits_processor = WatermarkLogitsProcessor(watermark_model)
    elif args.watermark_type == "context":
        watermark_model = WatermarkContext(device, args.chunk_size, tokenizer, delta = args.delta,transform_model_path=args.transform_model, embedding_model=args.embedding_model)
        logits_processor = WatermarkLogitsProcessor(watermark_model)
    else:
        watermark_model, logits_processor = None, None
    
    with open('./Dataset/Prompts/prompt_cn.pkl', "rb") as f:
        prompts = pickle.load(f)
    outputs = []
    for prompt in prompts:
        words = prompt.split()
        begin_text = ' '.join(words)
        inputs = tokenizer(begin_text, return_tensors="pt").to(device)
        generation_config = {
                "max_length": 650,
                "no_repeat_ngram_size": 4,
            }
        if args.decode_method == "sample":
            generation_config["do_sample"] = True
        elif args.decode_method == "beam":
            generation_config["num_beams"] = args.beam_size
            generation_config["do_sample"] = False
        
        if watermark_model is not None:
            generation_config["logits_processor"] = LogitsProcessorList([logits_processor])
        
        with torch.no_grad():
            o = model.generate(**inputs, **generation_config)
            outputs.append(tokenizer.decode(o[0], skip_special_tokens=True))
    with open('./Dataset/newlyGeneratedData/watermark_sir_cn.pkl' , 'wb') as f:
         pickle.dump(outputs , f)
        
            

from collections import namedtuple

args_dict = {
    'watermark_type': 'context',
    'base_model': 'llama',
    'llm_path': './llama-2-7b-hf',
    'window_size': 0,
    'generate_number': 200,
    'delta': 1.0,
    'chunk_size': 10,
    'max_new_tokens': 50,
    'data_path': 'data/dataset/prompt.pkl',
    'output_path': 'SIR.json',
    'transform_model': './RobustWatermark/model/transform_model_cbert.pth',
    'embedding_model': './RobustWatermark/data/compositional-bert-large-uncased',
    'decode_method': 'sample',
    'prompt_size': 30,
    'beam_size': 5
}

Args = namedtuple('Args', args_dict.keys())
args = Args(*args_dict.values())


generate_watermark_sir(args)


