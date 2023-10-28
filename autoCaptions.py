from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
import os
import shutil
import re


# Note: The default behavior now has injection attack prevention off.
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
torch.manual_seed(1)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
# print('model.generation_config',model.generation_config)

input_directory = 'F:/lora_training/simple_drawing/images/10_simple_drawing_empty'
output_directory = 'F:/lora_training/simple_drawing/images/10_simple_drawing_desc'

# create output directory if not exists
if not os.path.exists(output_directory):
  os.makedirs(output_directory)

descriptive_prompt = 'You are part of a team of bots that creates images. You work with an assistant bot that will draw anything.you say in double quote. For example, outputting "a sandcastle with a tall, pointed structure resembling a castle or a church steeple. The candles are placed around the sandcastle, with some closer to the base and others near the top of the structure. " will trigger your partner bot to output an image of a sandcastle, as described. You will be prompted by people looking to create detailed , amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.'
descriptive_hind = 'Please descript the image in detailed descriptive way.'


categories = [
  "religion",
  "animal",
  "portrait",
  "male",
  "female",
  "multiple_people",
  "anime",
  "landscape_only",
  "digital_painting",
  "2d_pixel_image",
  "food",
  "unrecognized"
]

MAX_CLASSIFICATION_LOOP = 3
RE_INIT_MODEL_WHILE_FILES = 1000

# create categories in output directory if not exists
# for category in categories:
#   category_directory = os.path.join(output_directory, category)
#   if not os.path.exists(category_directory):
#     os.makedirs(category_directory)

categories_string = ",".join(categories)
output_categories_string = ":scores,".join(categories)
output_categories_string = f'{output_categories_string}:scores  '

classification_prompt = f"You would classify prompt into different categorys including: {categories_string} You will be prompted by people looking to classify images for training. The way to accomplish this is to score the image for each category and detemine the prompt should go into which category. You should give each category a scores from 1 to 100. If you couldn't fit the image into above categories labeled it as unrecognized. You must assign the prompt to atleast one categories. You response must includes all categories"
classification_hint = f'Please classify the image in different categories resulting scores from 1-100. You should carefully scores each categories and give the reason for each scores. Higher scores is more related to the categories. Lower scores is not that much related to the categoreis. Follow format: {output_categories_string} Reason: reason'


def get_scores(response):
  response = response.lower().replace('_',' ')
  scores_string = response
  # find 'Reason:' index in response, get 0 to index string
  reason_index = response.find('Reason:')
  if reason_index != -1:
    scores_string = response[0:reason_index]
  
  reason_string = response[reason_index:]
  # print('response: ', response)
  # print('scores_string: ', scores_string)

  # define max_scores for looping to get max score
  max_scores = 0
  result = {}
  max_category = ''
  sum_scores = 0
  for index, category in enumerate(categories):
    result[category] = 0
    first_index = scores_string.find(category.replace('_', ' '))+len(category)
    rest_part = scores_string[first_index:]
    if rest_part[0] == ':':
        first_index+=1
        rest_part = scores_string[first_index:]
    second_index = rest_part.find(',')
    # no comma found
    if second_index == -1:
      second_index = rest_part.find(':')
      if second_index == -1:
        second_index = len(rest_part)
    second_index+=first_index
    # print('second_index',second_index)
    s = scores_string[first_index:second_index]
    # print('scores_string',scores_string)
    # print('category',category.replace('_', ' '))
    # print('search scores with: ', s)
    pattern = r'\d+' # a regular expression that matches one or more digits
    match = re.search(pattern, s) # search for the pattern in the string
    category_value = 0
    if match: # if a match is found
        category_value = int(match.group()) # convert the matched substring to an integer
        # print(category_value) # print the number
    else: # if no match is found
        print(category, ' category_value not found') # print a message
    
    result[category] = category_value
    sum_scores += category_value
    print('category: ', category, ' score: ', category_value)
    # get max score
    if int(category_value) > max_scores:
      max_scores = int(category_value)
      max_category = category
  # extra bias of image classification
  # if scores of food is greater than 50, set max_category to food

  average_scores = sum_scores / len(categories)
  # print('average_scores: ', average_scores)
  if result['food'] == max_scores and result['food'] > average_scores:
      max_category = 'food'
      max_scores = result['food']
  # if max category is landscape_only and people is greater than 50, set max_category to people
  if max_category == 'landscape_only' or max_category == 'unrecognized':
    if result['portrait'] > max_scores*0.8:
      max_category = 'portrait'
      max_scores = result['portrait']
    # elif result['human'] > max_scores*0.8:
    #   max_category = 'human'
    #   max_scores = result['human']
    elif result['animal'] > max_scores*0.8:
      max_category = 'animal'
      max_scores = result['animal']
    if max_category == 'unrecognized' and result['landscape_only'] > max_scores*0.8:
      max_category = 'landscape_only'
      max_scores = result['landscape_only']

  # print('result',result)
  print('max_category: ', max_category, ' max_scores: ', max_scores)  
  return max_category, max_scores

def write_text(filename,category_dir,content):
  output_path = os.path.join(category_dir, filename)
  # print('output_path: ', output_path)
  # create output_prompt_path file if not exists
  if not os.path.exists(output_path):
    open(output_path, 'a').close()
  with open(output_path, 'r+',encoding='utf-8') as output_file:
    output_file.truncate(0)
    output_file.write(content)

file_count = 0

def init_model(seed,model,tokenizer):
  del model
  del tokenizer
  torch.cuda.empty_cache()
  torch.manual_seed(seed)
  print('--------------manual_seed: ', seed)
  model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# loop through all files in input directory
for filename in os.listdir(input_directory):
  file_count+=1
  # re init model and tokenizer every 10 files
  # if file_count % RE_INIT_MODEL_WHILE_FILES == 0:
  #   print('--------------Re init model and tokenizer')
  #   init_model(file_count,model,tokenizer)
  print('--------------filename: ', filename)
  # skip if file is not jpg or png
  if not filename.lower().endswith('.jpg') and not filename.lower().endswith('.png'):
    continue

  # check if file is already exist in output directory
  # if exist, skip this file
  if os.path.exists(os.path.join(output_directory, filename)):
    print('--------------File already exist. Skip this file: ', filename)
    continue
  

  ori_filename = filename
  history = None
  image_path = os.path.join(input_directory, filename)
  print('--------------image_path: ', image_path)

  instruction_prompt = [{"text":descriptive_prompt}]
  instruction = tokenizer.from_list_format(instruction_prompt)
  response, history = model.chat(tokenizer, query=instruction, history=history)
  # print('instruction: ', response, '\n')
  query = tokenizer.from_list_format([
      {'image': image_path}, # Either a local path or an url
      {'text': descriptive_hind},
  ])
  response, history = model.chat(tokenizer, query=query, history=history)
  print('prompt: ', response)
  prompt = response

  filename_without_extention = os.path.splitext(filename)[0]

  # # classify image first
  # instruction_prompt = [{"text":classification_prompt}]

  # # # 1st dialogue turn
  # instruction = tokenizer.from_list_format(instruction_prompt)
  # response, history = model.chat(tokenizer, query=instruction, history=None)
  # # print('second instruction: ', response, '\n')

  # query = tokenizer.from_list_format([
  #     {'image': image_path}, # Either a local path or an url
  #     {'text': classification_hint},
  # ])
  # response, history = model.chat(tokenizer, query=query, history=history)
  # print('classification: ', response)
  # classification = response
  
  # classify_wrong = False
  # # try 5 times to classify image
  # classification_count = 0
  # # if get response is error, the response might be using unknown format
  # while not classify_wrong and classification_count < MAX_CLASSIFICATION_LOOP:
  #   try:
  #     query = tokenizer.from_list_format([
  #         {'image': image_path}, # Either a local path or an url
  #         {'text': classification_hint},
  #     ])
  #     response, history = model.chat(tokenizer, query=query, history=history)
  #     print('classification: ', response)
  #     classification = response
  #     max_category, max_scores = get_scores(response)
  #     classify_wrong = True
  #   except Exception as e:
  #     # Print the exception message
  #     print(f"An exception occurred: {str(e)}")
  #     classification_count+=1
  #     print('Classify Wrong. Re run classification.')
  #     init_model(file_count,model,tokenizer)
  
  # if classify_wrong and classification_count >= 5:
  #   print('Unable to classify image. Skip this image.')
  #   continue

  # category_dir = os.path.join(output_directory, max_category)

  category_dir = output_directory

  suffix = ''
  prompt_file = f'{filename_without_extention}{suffix}.txt'
  # write prompt to classified folder
  # replace filename extention with .txt, filename extension is .jpg or .png
  write_text(prompt_file,category_dir,prompt)


  # suffix = '_classification'
  # classification_file = f'{filename_without_extention}{suffix}.txt'
  # # write classified_log to classified folder
  # # replace filename extention with .txt, filename extension is .jpg or .png
  # write_text(classification_file,category_dir,classification)

  # copy image to classified folder
  output_image_path = os.path.join(category_dir, filename)
  print('output_image_path: ', output_image_path)
  # copy image from image_path to output_image_path, overwrite if exists
  shutil.copyfile(image_path, output_image_path)
  # copy image to classified folder


  print('--------------End: ', filename)
  # break
  