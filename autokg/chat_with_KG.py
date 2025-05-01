import os
import matplotlib.pyplot as plt
import numpy as np
import openai
import yaml

from utils import *

import autoKG_full as AKG

with open("config.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


COMPLETIONS_MODEL = params["OPENAI_API_MODEL"]
EMBEDDING_MODEL = params["EMBEDDING_MODEL"]
my_api_key = params["OPENAI_API_KEY"]
openai.api_key = my_api_key

os.environ['OPENAI_API_KEY'] = my_api_key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


KG_class_chat = AKG.autoKG(texts=None,
                           source=None,
                           embedding_model=EMBEDDING_MODEL,
                           llm_model=COMPLETIONS_MODEL,
                           openai_api_key=OPENAI_API_KEY,
                           main_topic=None,
                           embedding=False)


KG_class_chat.load_data(os.path.join('KG_data1', 'KGtest0427.npy'), include_texts=True)


query = """
give me some literatures suggestion about wildfire and health risks 
"""
record = KG_class_chat.KG_prompt(query,
                                 search_nums=(15, 7, 3, 4, 2),
                                 search_mtd='pair_dist',
                                 use_u=False)
response, keywords_info, ref_info, all_tokens = KG_class_chat.completion_from_record(
                               record,
                               output_tokens=1024,
                               prompt_language='English',
                               show_prompt=False,
                               prompt_keywords=False,
                               include_source=False,
                               )


print("all_tokens: ", all_tokens)
print("response: ", response)


KG_class_chat.draw_graph_from_record(record,
                                     node_colors=([0, 1, 1], [0, 1, 0.5], [1, 0.7, 0.75]),
                                     node_shape='o',
                                     edge_color='black',
                                     edge_widths=(2, 0.5),
                                     node_sizes=(500, 150, 50),
                                     font_color='black',
                                     font_size=6,
                                     show_text=False,
                                     save_fig=True)
                                     #save_path='KG_outputs0427/Subgraph_vissoa.png')


KG_class_chat.draw_graph_from_record(record,
                                     node_colors=([0, 1, 1], [0, 1, 0.5], [1, 0.7, 0.75]),
                                     node_shape='o',
                                     edge_color='black',
                                     edge_widths=(2, 0.5),
                                     node_sizes=(500, 150, 50),
                                     font_color='black',
                                     font_size=6,
                                     show_text=True,
                                     save_fig=True)
                                     #save_path='KG_outputs0427/Subgraph_vis_textsoa.png')