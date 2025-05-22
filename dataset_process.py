import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
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

# Caching


def collate_fn_wino(batch):
    # breakpoint()
    image0_batch = []
    image1_batch = []
    caption0_batch = []
    caption1_batch = []
    entity_batch = []
    final_tag_batch = []
    
    for idx in range(len(batch)):
        image0_batch.append(batch[idx]['image_0'])
        image1_batch.append(batch[idx]['image_1'])
        caption0_batch.append(batch[idx]['caption_0'])
        caption1_batch.append(batch[idx]['caption_1'])
        entity_batch.append(batch[idx]['collapsed_tag'])
        final_tag_batch.append(batch[idx]['secondary_tag'])


    return image0_batch, caption0_batch, image1_batch, caption1_batch, entity_batch, final_tag_batch

def collate_fn_eqben(batch):
    image0_batch = []
    image1_batch = []
    caption0_batch = []
    caption1_batch = []
    entity_batch = []
    final_tag_batch = []
    
    for idx in range(len(batch)):
        image0_batch.append(batch[idx]['image_0'])
        image1_batch.append(batch[idx]['image_1'])
        caption0_batch.append(batch[idx]['caption_0'])
        caption1_batch.append(batch[idx]['caption_1'])
        # entity_batch.append(None)
        # final_tag_batch.append(None)
    
    return image0_batch, caption0_batch, image1_batch, caption1_batch, entity_batch, final_tag_batch

def collate_fn_colorswap(batch):
    image0_batch = []
    image1_batch = []
    caption0_batch = []
    caption1_batch = []
    entity_batch = []
    final_tag_batch = []
    
    for idx in range(len(batch)):
        image0_batch.append(batch[idx]['image_1'])
        image1_batch.append(batch[idx]['image_2'])
        caption0_batch.append(batch[idx]['caption_1'])
        caption1_batch.append(batch[idx]['caption_2'])
        # entity_batch.append(None)
        # final_tag_batch.append(None)
    
    return image0_batch, caption0_batch, image1_batch, caption1_batch, entity_batch, final_tag_batch

def collate_fn_sugarcrepe(batch):
    image0_batch = []
    caption0_batch = []
    caption1_batch = []
    entity_batch = []
    
    for idx in range(len(batch)):
        image0_batch.append(batch[idx]['image'])
        caption0_batch.append(batch[idx]['caption_0'])
        caption1_batch.append(batch[idx]['caption_1'])
        entity_batch.append(batch[idx]['collapsed_tag'])

    return image0_batch, caption0_batch, caption1_batch, entity_batch



def load_compositionality_dataset(dataset_name, split = "test"):
    if dataset_name == "wino":
        dataset = load_dataset("facebook/winoground", cache_dir= '/bigtemp/ss7mu/datasets/')
        test_data = dataset[split]

        test_dataloader = DataLoader(test_data, batch_size=1, collate_fn = collate_fn_wino)

        # Level-2 dictionary, organized by scores (txt,img,grp)
        sub_dic = {"Object":[0,0,0], "Relation":[0,0,0], "Both":[0,0,0],"Symbolic":[0,0,0],"Pragmatics":[0,0,0]}

    elif dataset_name == "eqben":
        with open("/bigtemp/ss7mu/datasets/eqben_subset_10percent_final.json") as f:
            info = json.load(f)
        def gen():
            num = 0
            for i in range(len(info)):
                # print(i)
                if num == 400:
                    break
                try:
                    yield {
                        "id":i,
                        'image_0': Image.open("/bigtemp/ss7mu/datasets/image_subset/"+info[i]['image0']).convert("RGB"), 
                        'image_1': Image.open("/bigtemp/ss7mu/datasets/image_subset/"+info[i]['image1']).convert("RGB"), 
                        'caption_0': info[i]['caption0'], 
                        'caption_1': info[i]['caption1'],
                        'collapsed_tag': None
                    }
                    num+=1
                except:
                    print("error",i)
                    pass

        test_data = Dataset.from_generator(gen)
        # breakpoint()
        test_dataloader = DataLoader(test_data, batch_size=1, collate_fn = collate_fn_eqben)
        sub_dic = None

    if dataset_name == "colorswap":
        dataset = load_dataset("stanfordnlp/colorswap", cache_dir= '/bigtemp/ss7mu/datasets/')
        test_data = dataset[split]

        test_dataloader = DataLoader(test_data, batch_size=1, collate_fn = collate_fn_colorswap)

        # Level-2 dictionary, organized by scores (txt,img,grp)
        sub_dic = None


    elif dataset_name == "sugarcrepe":
        # split = "test"
        with open("/bigtemp/ss7mu/datasets/sugarcrepe/sugar-crepe/data/%s.json"%split) as f:
            info = json.load(f)

        # breakpoint()

        def gen():
            num = 0
            for i in range(len(info)):
                # print(i)
                if num == 400:
                    break
                try:
                    yield {
                        "id":i,
                        'image': Image.open("/bigtemp/ss7mu/datasets/val2017/"+info[str(i)]['filename']).convert("RGB"), 
                        'caption_0': info[str(i)]['caption'],
                        'caption_1': info[str(i)]['negative_caption'],
                        'collapsed_tag': split
                    }
                    num+=1
                except:
                    print("error",i)
                    pass

        test_data = Dataset.from_generator(gen)
        # breakpoint()
        test_dataloader = DataLoader(test_data, batch_size=1, collate_fn = collate_fn_sugarcrepe)
        sub_dic = None
    
    return test_dataloader, sub_dic



