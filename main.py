import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torch.nn.parallel import DataParallel
import torch.nn.functional as F

from datasets import Dataset, load_dataset, load_from_disk

from transformers import Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, AutoModelForVisualQuestionAnswering,AutoModelForCausalLM, AutoModelForZeroShotObjectDetection
from transformers import pipeline

import argparse
from tqdm import tqdm
import os
import json
import numpy as np
from openai import OpenAI
import warnings
import pickle
from undecorated import undecorated
from types import MethodType
from PIL import Image


from dataset_process import load_compositionality_dataset

from utils import ComposeMetrics
from utils import prompt_system1_model, discover_concept_nodes, make_pipe, sys2_reasoner



# Caching



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vlm_model(model_type):
    # For smaller models (CLIP based)
    if model_type== "blip2":
        model_name = "Salesforce/blip2-flan-t5-xl"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name,device_map = "auto", cache_dir = cache_dir)
    if model_type == "instructblip":
        processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", device_map="auto", cache_dir = cache_dir)
    if model_type == "blip2-xxl":
        processor = InstructBlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-flan-t5-xxl", device_map="auto",cache_dir = cache_dir)
    
    # For medium models (below 8b)
    if model_type == "instructblip-vicuna":
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", device_map="auto",cache_dir = cache_dir)
    if model_type == "llava-1.5":
        model_name = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)
    if model_type == "llava-1.6":
        model_name = "llava-hf/llava-v1.6-vicuna-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)
    if model_type == "qwen-7b":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto",cache_dir = cache_dir)
    if model_type == "internvl-8b":
        processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL2_5-8B", device_map="auto",cache_dir = cache_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained("OpenGVLab/InternVL2_5-8B", device_map="auto",cache_dir = cache_dir, trust_remote_code=True)

    # For big models (above 13-15b)
    if model_type == "instructblip-xxl": #prob 13b
        processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
        model = AutoModelForImageTextToText.from_pretrained("Salesforce/instructblip-flan-t5-xxl", device_map="auto",cache_dir = cache_dir)
    if model_type == "instructblip-vicuna-13b":
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b", device_map="auto",cache_dir = cache_dir)
    if model_type == "llava-1.5-13b":
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
        model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-1.5-13b-hf", device_map="auto",cache_dir = cache_dir)
    if model_type == "llava-1.6-13b":
        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
        model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", device_map="auto",cache_dir = cache_dir)


    # For SOTA models (very big)s
    if model_type == "internvl-26b":
        processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL2_5-26B", device_map="auto",cache_dir = cache_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained("OpenGVLab/InternVL2_5-26B", device_map="auto",cache_dir = cache_dir, trust_remote_code=True)
    
    

    return model, processor

def load_llm_model(model_type):
    if model_type == "llama-3.1-8b":
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        language_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)
        language_tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)

    if model_type == "llama2-13b":
        model_name = "meta-llama/Llama-2-13b-hf"
        language_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)
        language_tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)

    elif model_type == "qwen-14b":
        language_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct", device_map="auto",cache_dir = cache_dir)
        language_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", device_map="auto",cache_dir = cache_dir)
    
    elif model_type == "qwen-1.5":
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        language_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)
        language_tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)

    return language_model, language_tokenizer

def calculate_prompt_prob(image_batch, prompt, decode = False):
    # Returns prob(yes), prob(no)
    image_batch = [image_batch[0].resize((94, 94))]
    # sz = image_batch[0].size
    # image_batch = [image_batch[0].resize((sz[0]//2,sz[1]//2))]
    yes_prompt = ["Does this figure show: %s? Please answer Yes or No."%(prompt)]
    out = prompt_system1_model(model, processor, image_batch, yes_prompt ,temp=0, decode = decode)
    # out = prompt_system1_model_with_context(model, processor, image_batch, yes_prompt ,temp=0, decode = decode) 

    req_gen_prompt = ["llava"]
    if any(m in model.__class__.__name__.lower() for m in req_gen_prompt):
        probs_yes = out[0].scores[-1][0][processor.tokenizer.convert_tokens_to_ids(["Yes"])]#+out[0].scores[-2][0][processor.tokenizer.convert_tokens_to_ids(["yes"])] # Index of Yes
        probs_no = out[0].scores[-1][0][processor.tokenizer.convert_tokens_to_ids(["No"])]#+out[0].scores[-2][0][processor.tokenizer.convert_tokens_to_ids(["no"])] # Index of No
    elif "qwen" in model.__class__.__name__.lower():
        probs_yes = out[0].scores[0][0][processor.tokenizer.convert_tokens_to_ids(["Yes"])]
        probs_no = out[0].scores[0][0][processor.tokenizer.convert_tokens_to_ids(["No"])]
    elif "internvl" in model.__class__.__name__.lower():
        probs_yes = out[0].scores[-1][0][processor.convert_tokens_to_ids(["Yes"])]# Index of Yes
        probs_no = out[0].scores[-1][0][processor.convert_tokens_to_ids(["No"])] # Index of No
    else:
        # print(out.scores[0])
        # out.scores = torch.nn.functional.normalize(out.scores[0],dim=-1)
        # probs_yes = 1*out.scores[0][processor.tokenizer.convert_tokens_to_ids(["Yes"])]+out.scores[0][processor.tokenizer.convert_tokens_to_ids(["yes"])] # Index of Yes
        # probs_no = 1*out.scores[0][processor.tokenizer.convert_tokens_to_ids(["No"])]+out.scores[0][processor.tokenizer.convert_tokens_to_ids(["no"])] # Index of No
        probs_yes = out[0][0][processor.tokenizer.convert_tokens_to_ids(["Yes"])]#+0*out[0][0][processor.tokenizer.convert_tokens_to_ids(["yes"])] # Index of Yes
        probs_no = out[0][0][processor.tokenizer.convert_tokens_to_ids(["No"])]#+0*out.scores[0][0][processor.tokenizer.convert_tokens_to_ids(["no"])] # Index of No
    return torch.nn.Softmax(dim=-1)(torch.tensor([probs_yes, probs_no]))#, out

def calculate_caption_label(image_batch, caption1, caption2):
    question_prompt = ["Choose only one of the choices. Does this figure show A. %s or B. %s?"%(caption1,caption2)]
    out, label = prompt_system1_model(model, processor, image_batch, question_prompt ,temp=0, decode = True)
    req_gen_prompt = ["llava"]
    if any(m in model.__class__.__name__.lower() for m in req_gen_prompt):
        probs_yes = out[1][0][0][processor.tokenizer.convert_tokens_to_ids(["A"])]#+out[0].scores[-2][0][processor.tokenizer.convert_tokens_to_ids(["yes"])] # Index of Yes
        probs_no = out[1][0][0][processor.tokenizer.convert_tokens_to_ids(["B"])]#+out[0].scores[-2][0][processor.tokenizer.convert_tokens_to_ids(["no"])] # Index of No
    elif "qwen" in model.__class__.__name__.lower():
        probs_yes = out[0].scores[0][0][processor.tokenizer.convert_tokens_to_ids(["A"])]
        probs_no = out[0].scores[0][0][processor.tokenizer.convert_tokens_to_ids(["B"])]
    elif "internvl" in model.__class__.__name__.lower():
        probs_yes = out[0].scores[-1][0][processor.convert_tokens_to_ids(["A"])]# Index of Yes
        probs_no = out[0].scores[-1][0][processor.convert_tokens_to_ids(["B"])] # Index of No
    else:
        # breakpoint()
        probs_yes = out[0][processor.tokenizer.convert_tokens_to_ids(["A"])] # Index of Yes
        probs_no = out[0][processor.tokenizer.convert_tokens_to_ids(["B"])] # Index of No
        # print([probs_yes, probs_no])
    return label, torch.nn.Softmax(dim=-1)(torch.tensor([probs_yes, probs_no]))#, out



def get_text_score(image0_batch,caption0_batch,image1_batch,caption1_batch, idx = None, beta=0, saved_outputs = False, sys2 = False):
    
    if not saved_outputs:
        sys1_predictions0_0 = calculate_prompt_prob(image0_batch,caption0_batch[0])[0]
        sys1_predictions0_1 = calculate_prompt_prob(image0_batch,caption1_batch[0])[0]
        if image1_batch:
            sys1_predictions1_0 = calculate_prompt_prob(image1_batch,caption0_batch[0])[0]
            sys1_predictions1_1 = calculate_prompt_prob(image1_batch,caption1_batch[0])[0]
        else:
            sys1_predictions1_0 = 0
            sys1_predictions1_1 = 0
    
    else:
        sys1_predictions0_0,sys1_predictions0_1,sys1_predictions1_0,sys1_predictions1_1 = saved_outputs[idx][0], saved_outputs[idx][1], saved_outputs[idx][2], saved_outputs[idx][3]

    if sys2:
        sys2_predictions0_0, sys2_predictions0_1, sys2_predictions1_0, sys2_predictions1_1 = sys2[0],sys2[1], sys2[2], sys2[3]

    # print(predictions0_0, predictions0_1, predictions1_0, predictions1_1)
    predictions0_0 = beta*sys1_predictions0_0 + (1-beta)*sys2_predictions0_0
    predictions0_1 = beta*sys1_predictions0_1 + (1-beta)*sys2_predictions0_1
    predictions1_0 = beta*sys1_predictions1_0 + (1-beta)*sys2_predictions1_0
    predictions1_1 = beta*sys1_predictions1_1 + (1-beta)*sys2_predictions1_1
    
    txt_score, img_score, grp_score = 0, 0, 0 # Flags not numbers

    if predictions0_0>predictions0_1:
        if image1_batch:
            if predictions1_0<predictions1_1:
                txt_score = 1
        else:
            txt_score = 1 # Sugarcrepe only has 1 txt score

    if predictions0_0>predictions1_0:
        if predictions0_1<predictions1_1:
            img_score = 1

    if txt_score == 1 and img_score == 1:
        grp_score = 1

    return [txt_score, img_score, grp_score], [sys1_predictions0_0,sys1_predictions0_1,sys1_predictions1_0,sys1_predictions1_1]


def get_caption_score(image0_batch,caption0_batch,image1_batch,caption1_batch, idx = None, saved_outputs = False, sys2 = False):
    beta=1
    txt_score, img_score, grp_score = 0, 0, 0
    if not saved_outputs:
        # breakpoint()
        if image1_batch:
            predictions0, prob_predictions0 = calculate_caption_label(image0_batch,caption0_batch[0],caption1_batch[0])
            predictions1, prob_predictions1 = calculate_caption_label(image1_batch,caption0_batch[0],caption1_batch[0])
            # breakpoint()
            print(predictions0, prob_predictions0,predictions1, prob_predictions1)
            if predictions0[0] in ["A","a"]:
                if predictions1[0] in ["B","b"]:
                    txt_score = 1

            if prob_predictions0[0] > prob_predictions1[0]: # prob of a1 > a2 for image 1
                if prob_predictions0[1] < prob_predictions1[1]: # prob of b2 > b1 for image 2
                    img_score = 1

            if txt_score == 1 and img_score == 1:
                grp_score = 1

            
            return [txt_score, img_score, grp_score], None
        
        else:
            predictions0, prob_predictions0 = calculate_caption_label(image0_batch,caption0_batch[0],caption1_batch[0])
            if predictions0[0] in ["A","a"]:
                txt_score = 1
            return [txt_score, img_score, grp_score], None
    else:
        sys1_predictions0_0,sys1_predictions0_1,sys1_predictions1_0,sys1_predictions1_1 = saved_outputs[idx][0], saved_outputs[idx][1], saved_outputs[idx][2], saved_outputs[idx][3]

    # if sys2:
    #     sys2_predictions0_0, sys2_predictions0_1, sys2_predictions1_0, sys2_predictions1_1 = sys2_reasoner(idx)

    # print(predictions0_0, predictions0_1, predictions1_0, predictions1_1)
    # predictions0_0 = beta*sys1_predictions0_0 #+ (1-beta)*sys2_predictions0_0
    # predictions0_1 = beta*sys1_predictions0_1 #+ (1-beta)*sys2_predictions0_1
    # predictions1_0 = beta*sys1_predictions1_0 #+ (1-beta)*sys2_predictions1_0
    # predictions1_1 = beta*sys1_predictions1_1 #+ (1-beta)*sys2_predictions1_1
    

    return [txt_score, img_score, grp_score], [sys1_predictions0_0,sys1_predictions0_1,sys1_predictions1_0,sys1_predictions1_1]


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for VLM/LLM evaluation pipeline")

    parser.add_argument('--mode', type=str, choices=["eval", "concept-tree", "preprocess", "sys2-tree"], required=True, help='Run mode')

    parser.add_argument('--vlm-name', type=str, required=False, help='Name of the vision-language model')
    parser.add_argument('--llm-name', type=str, required=False, default="llama-3.1-8b", help='Name of the language model')
    parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--dataset-split', type=str, required=False, default = "test", help='Dataset split - only used in sugarcrepe')
    parser.add_argument('--dataset-num-samples', type=int, required=False, help='Number of samples from the dataset')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value for evaluation')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta value for evaluation')
    parser.add_argument('--M', type=int, default=2, help='Number of morphological entities')
    parser.add_argument('--S', type=int, default=3, help='Parameter S splitting factor for the tree')
    parser.add_argument('--decoding_style', type=str, choices=["max", "beam"], default="max", help='Decoding style')
    parser.add_argument('--neurosymbolic_style', type=str, choices=["cnf", "dnf"], default="cnf", help='Neurosymbolic processing style')

    parser.add_argument('--load-save', type=bool, default=True, help='Load pre-computed inference runs')
    parser.add_argument('--sys1-save-folder', type=str, required=False, default = "./system1-outputs/" ,help='Name of the folder saving System-1 inference outputs')
    # parser.add_argument('--sys2-save-folder', type=str, required=False, default = "./system2-outputs/" ,help='Name of the folder saving System-2 inference outputs')
    parser.add_argument('--concept-tree-folder', type=str, required=False, default = "./concept-trees/" ,help='Name of the folder saving LLM generated concept trees')
    parser.add_argument('--sys2-concept-tree-folder', type=str, required=False, default = "./sys2-concept-trees/" ,help='Name of the folder saving LLM generated concept trees with VLM output')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # for args.dataset_split in ["add_att", "add_obj","replace_att","replace_obj","replace_rel","swap_att","swap_obj"]:
    # for args.dataset_split in ["test"]:
    g_t = []
    g_i = []
    g_g = []

    # for beta in [0.5,1]:
    #     for alpha in [0.8]:
    #         # for num_tree in [0,1,2,3,4,5,6,7,8,9,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]:
            # for num_tree in [0]:
                # print(alpha,beta)

                # alpha=0.1
                # for args.dataset_split in ["swap_obj"]:
                # for xxx in range(1):
    print(args.dataset_split)
    # Load the dataset
    test_dataloader, sub_dic = load_compositionality_dataset(args.dataset_name, args.dataset_split)

    if args.mode == "eval":
        final_score_preds = []
        # Intialize the metrics to 0
        metrics = ComposeMetrics(sub_dic)

        # Always store results at the same place
        if args.dataset_split == "test":
            sys1_save_location = args.sys1_save_folder + args.dataset_name+"-"+args.vlm_name+".pkl"
        else:
            sys1_save_location = args.sys1_save_folder + args.dataset_name+"-"+args.vlm_name+"-"+args.dataset_split+".pkl"
        
        # args.load_save = False

        if args.load_save:
            try:
                with open(sys1_save_location,"rb") as f:
                    final_score_preds = pickle.load(f)
            except:
                print("File is not found or corrupted, run inference again.")
            model, processor  = None, None

            try:
                sys2_save_location = args.sys2_concept_tree_folder + args.dataset_name+"-"+args.llm_name+"-"+args.vlm_name+".pkl"
                sys2_save_location_flip = "./sys2-concept-trees-flip/" + args.dataset_name+"-"+args.llm_name+"-"+args.vlm_name+".pkl"
                
                with open(sys2_save_location,"rb") as f:
                    tree1 = pickle.load(f)
                with open(sys2_save_location_flip,"rb") as f:
                    tree2 = pickle.load(f)
            except:
                print("Sys2 trees not found, run again!")
                exit(0)
            # breakpoint()
                
        else:
            # Load VLM Model
            model, processor  = load_vlm_model(args.vlm_name)
        
        # Keep track of index
        idx = 0
        for batch in tqdm(test_dataloader):
            breakpoint()
            if args.dataset_name == "sugarcrepe":
                image_batch, caption0_batch, caption1_batch, entity_batch = batch[0], batch[1], batch[2], batch[3]
                tag = None
            else:
                image0_batch, caption0_batch, image1_batch, caption1_batch, entity_batch, tag = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
                image0_batch[0].save("ttt.jpg")
            # System-1 Prediction
            if args.load_save:
                if args.dataset_name == "sugarcrepe":
                    score_preds, store_raw_values = get_text_score(image_batch,caption0_batch,None,caption1_batch,idx = idx, saved_outputs = final_score_preds)
                else:
                    sys2_scores = sys2_reasoner(alpha, idx, tree1, tree2, num_tree)
                    # score_preds, store_raw_values = get_text_score(image0_batch,caption0_batch,image1_batch,caption1_batch,idx = idx, beta=beta)
                    score_preds, store_raw_values = get_text_score(image0_batch,caption0_batch,image1_batch,caption1_batch,idx = idx, beta=beta, saved_outputs = final_score_preds, sys2=sys2_scores)
            else:
                if args.dataset_name == "sugarcrepe":
                    # score_preds, store_raw_values = get_caption_score(image_batch,caption0_batch,None,caption1_batch,False)
                    score_preds, store_raw_values = get_text_score(image_batch,caption0_batch,None,caption1_batch,False)
                else:
                    # score_preds, store_raw_values = get_caption_score(image0_batch,caption0_batch,image1_batch,caption1_batch,False)
                    score_preds, store_raw_values = get_text_score(image0_batch,caption0_batch,image1_batch,caption1_batch,False)
                # breakpoint()
                final_score_preds.append(store_raw_values)
            # System-2 Prediction
            # sys2_score_preds = get_text_score(image0_batch,caption0_batch,image1_batch,caption1_batch,total,True)
            metrics.update(score_preds, entity_batch, tag)
            # metrics.print_metrics()
            
            if not args.load_save:
                # if args.dataset_split == "test":
                #     sys1_save_location = args.sys1_save_folder + args.dataset_name+"-"+args.vlm_name+".pkl"
                # else:
                #     sys1_save_location = args.sys1_save_folder + args.dataset_name+"-"+args.vlm_name+"-"+args.dataset_split+".pkl"
                with open(sys1_save_location, "wb") as f:
                    pickle.dump(final_score_preds, f)

            idx += 1
            # metrics.print_metrics()
        print(args.dataset_split)
        metrics.print_metrics()

        g_t.append(metrics.txt_correct/metrics.total)
        g_i.append(metrics.img_correct/metrics.total)
        g_g.append(metrics.grp_correct/metrics.total)
    

    if args.mode == "concept-tree":
        # Load LLM model
        language_model, language_processor  = load_llm_model(args.llm_name)
        pipe = make_pipe(language_model, language_processor)
        # Keep track of index
        idx = 0
        all_trees = []
        for batch in tqdm(test_dataloader):
            if args.dataset_name == "sugarcrepe":
                image_batch, caption0_batch, caption1_batch, entity_batch = batch[0], batch[1], batch[2], batch[3]
                tag = None
            else:
                image0_batch, caption0_batch, image1_batch, caption1_batch, entity_batch, tag = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

            trees = discover_concept_nodes(pipe, language_model, language_processor, caption0_batch, caption1_batch, args.M, args.S)
            all_trees.append(trees)
            if args.dataset_split == "test":
                concept_tree_save_location = args.concept_tree_folder + args.dataset_name+"-"+args.llm_name+".pkl"
            else:
                concept_tree_save_location = args.concept_tree_folder + args.dataset_name+"-"+args.llm_name+"-"+args.dataset_split+".pkl"
            with open(concept_tree_save_location, "wb") as f:
                pickle.dump(all_trees, f)
            idx+=1


    if args.mode == "sys2-tree":
        sys_concept_tree_save_location = args.sys2_concept_tree_folder + args.dataset_name+"-"+args.llm_name+"-"+args.vlm_name+".pkl"
        # breakpoint()
        # if os.path.exists(sys_concept_tree_save_location):
        #     with open(sys_concept_tree_save_location, "rb") as f:
        #         sys2_concept_trees = pickle.load(f)
        #         idx = len(sys2_concept_trees)-1
        # else:
        #     sys2_concept_trees = []
        #     idx = 0

        sys2_concept_trees = []
        idx = 0
        model, processor  = load_vlm_model(args.vlm_name)

        concept_tree_save_location = args.concept_tree_folder + args.dataset_name+"-"+args.llm_name+".pkl"
        with open(concept_tree_save_location, "rb") as f:
            all_trees = pickle.load(f)


        # breakpoint()

        num = 0
        for batch in tqdm(test_dataloader):
            # if num < idx:
            #     print(num,idx)
            #     num+=1
            #     continue
            if args.dataset_name == "sugarcrepe":
                image_batch, caption0_batch, caption1_batch, entity_batch = batch[0], batch[1], batch[2], batch[3]
                tag = None
            else:
                image0_batch, caption0_batch, image1_batch, caption1_batch, entity_batch, tag = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
            
            id1, id2 = 1, 0

            img0_tree = []
            for ent in range(0,len(all_trees[idx][id1])):
                # preds = calculate_prompt_prob(image0_batch, all_trees[idx][0][ent][2])
                # breakpoint()
                try:
                    preds = calculate_prompt_prob(image0_batch, all_trees[idx][id1][ent][2])
                    print(all_trees[idx][id1][ent][2],preds[0].item())
                    ntree = all_trees[idx][id1][ent][:]
                    ntree.append(preds[0].item())
                    img0_tree.append(ntree)
                except:
                    breakpoint()
                    pass
            
            img1_tree = []
            for ent in range(0,len(all_trees[idx][id2])):
                try:
                    preds = calculate_prompt_prob(image1_batch, all_trees[idx][id2][ent][2])
                    print(all_trees[idx][id2][ent][2],preds[0].item())
                    ntree = all_trees[idx][id2][ent][:]
                    ntree.append(preds[0].item())
                    img1_tree.append(ntree)
                except:
                    breakpoint()
                    pass

            sys2_concept_trees.append([img0_tree,img1_tree])

            # sys_concept_tree_save_location = args.sys2_concept_tree_folder + args.dataset_name+"-"+args.llm_name+"-"+args.vlm_name+".pkl"
            sys_concept_tree_save_location = "./sys2-concept-trees-flip/" + args.dataset_name+"-"+args.llm_name+"-"+args.vlm_name+".pkl"
            
            with open(sys_concept_tree_save_location, "wb") as f:
                print(len(sys2_concept_trees))
                pickle.dump(sys2_concept_trees, f)

            idx +=1


    print(g_t)
    print(g_i)
    print(g_g)