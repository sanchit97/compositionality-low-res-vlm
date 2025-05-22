import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.nn.parallel import DataParallel

from datasets import Dataset, load_dataset, load_from_disk

from transformers import Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, AutoModelForVisualQuestionAnswering,AutoModelForCausalLM, AutoModelForZeroShotObjectDetection
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

from internvl_utils import get_conv_template

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def print_list(symbols):
    symbols = sorted(symbols)
    for i in range(len(symbols)):
        print(symbols[i][0],":",symbols[i][1],"-->",symbols[i][2], ":::",symbols[i][3])
        try:
            if symbols[i][0]!= symbols[i+1][0]:
                print("\n")
        except:
            print("END\n\n")


def prompt_system1_model(model, processor , image, text, temp=0, decode = True):
    # Models with agentic prompts like llava, qwen, internvl
    req_gen_prompt = ["llava","qwen"]
    if any(m in model.__class__.__name__.lower() for m in req_gen_prompt):
        conversation = [ {"role": "user", "content": [{"type": "text", "text": text[0]}, {"type": "image"},],},]
        final_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=final_prompt, return_tensors='pt').to(model.device)
        # breakpoint()
        # if not decode:
        # print(model.device)
        outputs = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, do_sample=False,temperature=temp, max_new_tokens=1)
        return outputs, [processor.decode(outputs[0][0][inputs.input_ids.size()[-1]:], skip_special_tokens=True)]

    elif "internvl" in model.__class__.__name__.lower():
        # conversation = [{"role": "user", "content": "<image> "+text[0]}]
        generation_config = {"num_beams":1,"max_new_tokens":1, "do_sample":False, "output_scores":True, "return_dict_in_generate":True}
        question =  "<image> "+text[0]
        pixel_values = load_image(image[0])
        history=None 
        return_history=False
        num_patches_list=None 
        IMG_START_TOKEN='<img>' 
        IMG_END_TOKEN='</img>'
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = processor.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id

        template = get_conv_template(model.template)
        template.system_message = model.system_message
        eos_token_id = processor.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()


        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = processor(query, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id

        if not decode:
            outputs = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
            return outputs, None #processor.decode(outputs[0][inputs.input_ids.size()[-1]:], skip_special_tokens=True)
        else:
            # TODO Does not work correctly right now - fix this to get correct decoding (not priority)
            outputs = model.generate(pixel_values=pixel_values,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    **generation_config, temperature=temp, 
                                    max_new_tokens=1)
            return [processor.decode(outputs[0][inputs.input_ids.size()[-1]:], skip_special_tokens=True)] 

    else:
        # Models where no agentic prompts are used (like instructblip)
        with torch.no_grad():
            inputs = processor(images = image, text = text, return_tensors="pt", padding=True).to(model.device)
            # Decoding converts the token id to word, usually not required
            # if not decode:
            outputs = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, do_sample=False,temperature=temp, max_new_tokens=1)
            # breakpoint()
            return outputs[1][0], processor.decode(outputs[0][0], skip_special_tokens=True) 
            # else:
            #     outputs = model.generate(**inputs,do_sample=False,temperature=temp, max_new_tokens=1)
            #     predictions = [processor.decode(output, skip_special_tokens=True) for output in outputs]
            #     predictions = [predictions[0].split(text[0])[-1].strip()]
            #     return predictions



def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=1):
    # image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def check_linguistic_probability(language_model, language_tokenizer, prompt, ogprompt):
    if "llama" in language_model.__class__.__name__.lower():
        final_prompt = "Given we observe %s. Is it possible %s?. Answer yes or no. Assistant: "%(prompt,ogprompt[0])
        messages = [
        {"role": "system", "content": "You are a helpful bot."},
        {"role": "user", "content": final_prompt }
        ]
        input_text = language_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = torch.tensor([input_text]).to(language_model.device)
        # Get logits from the model
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = language_model(input_ids, output_hidden_states=False, return_dict=True)
            logits = outputs.logits
        probs_yes = logits[0][-1][language_tokenizer.convert_tokens_to_ids(["yes"])]#+logits[0][-1][language_tokenizer.convert_tokens_to_ids(["yes"])]
        probs_no = logits[0][-1][language_tokenizer.convert_tokens_to_ids(["no"])]#+logits[0][-1][language_tokenizer.convert_tokens_to_ids(["no"])]
        return torch.nn.Softmax(dim=-1)(torch.tensor([probs_yes, probs_no]))
  

    elif "qwen" in language_model.__class__.__name__.lower():
        # breakpoint()
        # final_prompt = "Answer in a number between 0 and 1 only and nothing more. Given we observe - %s in an image. What is the probability that we observe %s in the image as well?."%(prompt,ogprompt[0])
        final_prompt = "Answer yes or no. Given we observe - %s in an image. Is it possible that in the image - %s?."%(prompt,ogprompt[0])
        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": final_prompt }
        ]
        input_text = language_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = language_tokenizer([input_text], return_tensors="pt").to(language_model.device)
        # with torch.no_grad():  # Disable gradient computation for inference
            # breakpoint()
        # output_raw = language_model.generate(**model_inputs, max_new_tokens=10)
        output_raw = language_model(**model_inputs, return_dict=True, max_new_tokens=20, temperature = 0.2)
        # generated_ids = [output_raw[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_raw)]
        # breakpoint()
        # generated_ids = output_raw[:, model_inputs.input_ids.shape[1]:]
        # response = language_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
        # probs_yes = float(response)
        # probs_no = 1 - float(response)
        logits = output_raw.logits
        probs_yes = logits[0][-1][language_tokenizer.convert_tokens_to_ids(["yes"])]
        probs_no = logits[0][-1][language_tokenizer.convert_tokens_to_ids(["no"])]*10 #Qwen
        return torch.nn.Softmax(dim=-1)(torch.tensor([probs_yes, probs_no]))
        # logits = output_raw.logits


    
## All tree based functions

def find_level_node(pipe, language_model, language_tokenizer, prompt, slist, depth, ogcaption, S):
    # print("At depth: ", depth)
    # This needs to be hard-coded to avoid circular dependencies (3) More than 3 would take exponentially bigger time/compute
    if depth==2:
        return slist

    # Note that entailment with OG caption is important to avoid hallucinations
    final_prompt = "List %d binary visual concepts to verify the %s. Ensure the outputs are possible for %s.\
                    Answer in small phrases and focus on verifiable things like objects, locations, actions, etc.\
                    Only include concepts which are not very semantic but focus on attributes in simple terms.\
                    Output format is: 1. xxx\n 2. xxx\n 3. xxx\n 4. xxx\n 5. xxx\n. Assistant: "%(S, prompt, ogcaption)
    messages = [{"role": "system", "content": "You are a helpful chatbot."},{"role": "user", "content": final_prompt}]
    # Temperature value is set to 1 here to force creativity
    if "llama" in language_model.__class__.__name__.lower():
        outputs = pipe(messages,max_new_tokens=30, pad_token_id= pipe.tokenizer.eos_token_id, do_sample=False, temperature=0.1)
    elif "qwen" in language_model.__class__.__name__.lower():
        outputs = pipe(messages,max_new_tokens=30, pad_token_id= pipe.tokenizer.eos_token_id, temperature=0.7)

    lvl = outputs[0]["generated_text"][-1]['content']
    lvl = lvl.split("\n")
    lvl = [l.strip(".").split(".")[-1].lower().strip() for l in lvl]
    
    for node in lvl:
        slist = find_level_node(pipe, language_model, language_tokenizer, node, slist, depth+1, ogcaption, S)
        slist.append([depth+1, \
                       prompt, \
                       node, \
                       check_linguistic_probability(language_model, language_tokenizer, node, ogcaption)[0].item()])
    return slist

def discover_concept_nodes(pipe, language_model, language_processor, caption1, caption2, M, S): 
    # Discovers the concept trees level-wise with M morphological entities and S concepts at each split
    trees = []
    for caption in [caption1, caption2]:
        final_prompt = "Divide the caption into %d smaller independent statements which combined together form the caption based on Subject and Object. Caption: %s. Output format is: 1. Subject Verb\n 2. Object Verb\n. Assistant: "%(M, caption)
        messages = [{"role": "system", "content": "You are a helpful chatbot."},{"role": "user", "content": final_prompt}]
        # This is an interesting part, for determinism we only use temp=0 and sampling False
        # However, for concept diversity, temp>0 can be an avenue
        outputs = pipe(messages,max_new_tokens=30, pad_token_id= pipe.tokenizer.eos_token_id, do_sample=False, temperature=0)
        cleaned_output = outputs[0]["generated_text"][-1]['content'].split("\n")
        cleaned_output = [cout.lower() for cout in cleaned_output if len(cout)>2]
        cleaned_output = [cout.strip(".").split(".")[-1].lower().strip() for cout in cleaned_output][-2:]
        print(cleaned_output)

        # This is a recursive function to populate slist, very prone to errors - be careful!
        slist = []
        slist = find_level_node(pipe, language_model, language_processor, cleaned_output[0], slist, 0, caption, S)
        print_list(slist)
        print("--"*5)

        trees.append(slist)

    return trees



def make_tree(tree,alpha,num_trees=0):
    # num_trees = 1
    depth_map = {}
    for node in tree:
        if node[0] not in depth_map:
            depth_map[node[0]] = [[node[2], node[3], node[4]]]
        else:
            depth_map[node[0]].append([node[2], node[3], node[4]])

    all_paths = [[x,y,z] for x in depth_map[1] for y in depth_map[2] for z in depth_map[3]]
    # all_paths = [[x,y] for x in depth_map[1] for y in depth_map[2] ]
    # all_paths = [[x] for x in depth_map[1]  ]

    path_weight = []
    for path in all_paths:
        llm_prob=0
        vlm_prob=0
        # llm_prob=1
        # vlm_prob=1
        path_symbols = []
        for symbol in path:
            llm_prob+=symbol[1]
            vlm_prob+=symbol[2]
            # llm_prob*=1/symbol[1]
            # vlm_prob*=1/symbol[2]
            path_symbols.append(symbol)
        llm_prob/=len(path)
        vlm_prob/=len(path)
        path_weight.append([alpha*llm_prob + (1-alpha)*vlm_prob,path_symbols])

    path_weight = sorted(path_weight, reverse = True)
    # mm = [p[0] for p in path_weight]
    # breakpoint()
    # print(path_weight)
    # breakpoint()
    # path_weight = path_weight[:num_trees]
    path_weight = [path_weight[num_trees]]
    # path_weight = [path_weight[0]]
    
    collect = []
    for p in path_weight:
        collect.append(p[0])
    print(path_weight)
    # print(path_symbols)
    # print(collect)
    # collect = [max(collect)]
    collect = [sum(collect)/len(collect)]
    return collect


def sys2_reasoner(alpha, idx, tree_pos, tree_neg, tree_num):
    
    s1list, s2list = tree_pos[idx][0], tree_pos[idx][1]

    collect = []
    collect = make_tree(s1list,alpha,tree_num)
    # for symbol in s1list:
    #     collect.append(alpha*symbol[3]+(1-alpha)*symbol[4])
    predictions0_0 = sum(collect)/len(collect)
    collect = []
    collect = make_tree(s2list,alpha,tree_num)
    # for symbol in s2list:
    #     collect.append(alpha*symbol[3]+(1-alpha)*symbol[4])
    predictions0_1 = sum(collect)/len(collect)

    s1list, s2list = tree_neg[idx][0], tree_neg[idx][1]

    collect = []
    collect = make_tree(s1list,alpha,tree_num)
    # for symbol in s1list:
    #     collect.append(alpha*symbol[3]+(1-alpha)*symbol[4])
    predictions1_0 = sum(collect)/len(collect)
    collect = []
    collect = make_tree(s2list,alpha,tree_num)
    # for symbol in s2list:
    #     collect.append(alpha*symbol[3]+(1-alpha)*symbol[4])
    predictions1_1 = sum(collect)/len(collect)

    return predictions0_0, predictions0_1, predictions1_0, predictions1_1




def make_pipe(language_model, language_tokenizer):
    pipe = pipeline(
    "text-generation",
    model=language_model, 
    tokenizer=language_tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",)
    return pipe


class ComposeMetrics:
    def __init__(self, orb_dic):
        self.txt_correct, self.img_correct, self.grp_correct = 0, 0, 0
        # self.orb_dic = {"Object":[0,0,0], "Relation":[0,0,0], "Both":[0,0,0], "Symbolic":[0,0,0], "Pragmatics":[0,0,0]}
        self.orb_dic = orb_dic
        # self.orb_total = {"Object":0, "Relation":0, "Both":0, "Symbolic":0, "Pragmatics":0}
        self.orb_total = {}
        if self.orb_dic:
            for i in self.orb_dic:
                self.orb_total[i] = 0
        self.total = 0

    def update(self, score_preds, entity_batch, tag):
        # For main scores
        self.txt_correct += score_preds[0]
        self.img_correct += score_preds[1]
        self.grp_correct += score_preds[2]

        if self.orb_dic:
            # For object, relation, both
            if entity_batch[0]!= "" and entity_batch[0] in self.orb_dic:
                self.orb_total[entity_batch[0]] += 1
                for i in range(len(self.orb_dic[entity_batch[0]])):
                    self.orb_dic[entity_batch[0]][i]+=score_preds[i]
            # For symbolic and pragmatic
            if tag[0]!= "" and tag[0] in self.orb_dic:
                self.orb_total[tag[0]] += 1
                for i in range(len(self.orb_dic[tag[0]])):
                    self.orb_dic[tag[0]][i]+=score_preds[i]

        self.total+=1

    
    def print_metrics(self,):
        print(f"Text Score: {self.txt_correct/self.total:.4f}")
        print(f"Image Score: {self.img_correct/self.total:.4f}")
        print(f"Group Score: {self.grp_correct/self.total:.4f}")
        if self.orb_dic:
            for name in self.orb_dic:
                if self.orb_total[name]!=0:
                    print(name+f" Text Score: {self.orb_dic[name][0]/self.orb_total[name]:.4f}")
                    print(name+f" Image Score: {self.orb_dic[name][1]/self.orb_total[name]:.4f}")
                    print(name+f" Group Score: {self.orb_dic[name][2]/self.orb_total[name]:.4f}")

