# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models" 
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial
import pickle
import json

import numpy # for gradio hot reload

import torch

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)
from transformers import LlamaForCausalLM, LlamaTokenizer

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=True,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-6.7b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    args = parser.parse_args()
    return args

def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])
    if("llama" in args.model_name_or_path and args.load_fp16):
        model = LlamaForCausalLM.from_pretrained("../"  + args.model_name_or_path , torch_dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained("../" + args.model_name_or_path)
    elif("llama" in args.model_name_or_path and args.load_fp16):
        model = LlamaForCausalLM.from_pretrained("../"  + args.model_name_or_path )
        tokenizer = LlamaTokenizer.from_pretrained("../" + args.model_name_or_path)
    elif args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")
   

    if args.use_gpu:
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
        device = "cuda:" + str(get_free_gpu()) if torch.cuda.is_available() else "cpu"   
    else:
        device = "cpu"

    model = model.to(device)
    model.eval()

    return model, tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
    
    print(f"Generating with {args}")

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            args) 
            # decoded_output_with_watermark)

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float): 
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d

def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output, args

def main(args , input_text = None): 
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    # Generate and detect, report to stdout
    if not args.skip_model_load:
        if(input_text is None):
            input_text = (
         "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
        "species of turtle native to the brackish coastal tidal marshes of the "
        "Northeastern and southern United States, and in Bermuda.[6] It belongs "
        "to the monotypic genus Malaclemys. It has one of the largest ranges of "
        "all turtles in North America, stretching as far south as the Florida Keys "
        "and as far north as Cape Cod.[7] The name 'terrapin' is derived from the "
        "Algonquian word torope.[8] It applies to Malaclemys terrapin in both "
        "British English and American English. The name originally was used by "
        "early European settlers in North America to describe these brackish-water "
        "turtles that inhabited neither freshwater habitats nor the sea. It retains "
        "this primary meaning in American English.[8] In British English, however, "
        "other semi-aquatic turtle species, such as the red-eared slider, might "
        "also be called terrapins. The common name refers to the diamond pattern "
        "on top of its shell (carapace), but the overall pattern and coloration "
        "vary greatly. The shell is usually wider at the back than in the front, "
        "and from above it appears wedge-shaped. The shell coloring can vary "
        "from brown to grey, and its body color can be grey, brown, yellow, "
        "or white. All have a unique pattern of wiggly, black markings or spots "
        "on their body and head. The diamondback terrapin has large webbed "
        "feet.[9] The species is"
        )

        args.default_prompt = input_text

        term_width = 80
        print("#"*term_width)
        print("Prompt:")
        print(input_text)

        _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(input_text, 
                                                                                            args, 
                                                                                            model=model, 
                                                                                            device=device, 
                                                                                            tokenizer=tokenizer)
        without_watermark_detection_result = detect(decoded_output_without_watermark, 
                                                    args, 
                                                    device=device, 
                                                    tokenizer=tokenizer)
        with_watermark_detection_result = detect(decoded_output_with_watermark, 
                                                 args, 
                                                 device=device, 
                                                 tokenizer=tokenizer)

        print("#"*term_width)
        print("Output without watermark:")
        print(decoded_output_without_watermark)
        print("-"*term_width)
        print(f"Detection result @ {args.detection_z_threshold}:")
        pprint(without_watermark_detection_result)
        print("-"*term_width)

        print("#"*term_width)
        print("Output with watermark:")
        print(decoded_output_with_watermark)
        print("-"*term_width)
        print(f"Detection result @ {args.detection_z_threshold}:")
        pprint(with_watermark_detection_result)
        print("-"*term_width)

    return


def detect_watermark(args ):
    import tqdm
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None
    repharsed_sir = [None]*5
    for i in range(5):
        with open(f"../Dataset/Attacked/PivotTranslation/llm_watermarked_kwg_pivot_translated.pkl", 'rb') as f:
            repharsed_sir[i] = pickle.load(f)
    original_text = []
    with open("../Dataset/initial_prompts/llm_output_cn.pkl", "rb") as f:
        original_text = pickle.load(f)
    with open("../Dataset/Watermarked/cn/llm_watermarked_kwg.pkl" , 'rb') as f:
        watermarked_text = pickle.load(f)

    write_path = "../Dataset/Confidences/"
    
    torch.manual_seed(0)
    output  = []
    for i in range(1):
        for j in range(len(repharsed_sir[i])):
            attacked_text = repharsed_sir[i][j]
            original_text_file = original_text[j]
            watermarked_text_file = watermarked_text[j]

            with torch.no_grad():
                z_score_generated ,_ = detect(watermarked_text_file , args, 
                                                    device=device, 
                                                    tokenizer=tokenizer) 
                z_score_origin ,_ = detect(original_text_file , args, 
                                                    device=device, 
                                                    tokenizer=tokenizer) 
                z_score_attacked,_ = detect(attacked_text , args, 
                                                    device=device, 
                                                    tokenizer=tokenizer) 
        # if len(outputs[0]) > args.max_new_tokens: 
            if(i == 0) :
                output.append({
                    'original': z_score_origin[3][1],
                    f'watermarked': z_score_generated[3][1],
                    f'attacked{i}' : z_score_attacked[3][1]
                })
            else : 
                output[j][f'attacked{i}'] = z_score_attacked[3][1]
            print(j , end =" ")
        print()

    with open(write_path + "kwg_confidences_pivot_translate.json", 'w') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
if __name__ == "__main__":

    args = parse_args()
    print(args)
    #main(args)
    detect_watermark(args)