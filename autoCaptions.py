from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
import os
import shutil

torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
# print('model.generation_config',model.generation_config)

input_directory = 'F:\ImageSet\photoawards'
output_directory = 'F:\ImageSet\photoawards\output'

# create output directory if not exists
if not os.path.exists(output_directory):
  os.makedirs(output_directory)

descriptive_prompt = 'You are part of a team of bots that creates images. You work with an assistant bot that will draw anything.you say in double quote. For example, outputting "a sandcastle with a tall, pointed structure resembling a castle or a church steeple. The sandcastle is illuminated by candles, creating a beautiful and unique display. The candles are placed around the sandcastle, with some closer to the base and others near the top of the structure. The scene is set against a dark background, which further emphasizes the glowing candles and the intricate design of the sandcastle." will trigger your partner bot to output an image of a sandcastle, as described. You will be prompted by people looking to create detailed , amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.'
descriptive_hind = 'Please descript the image in amazingly detailed and descriptive way.'


categories = [
  "religion",
  "animal",
  "people",
  "landscape_only",
  "artwork",
  "2d_pixel_image",
  "digital_illustration",
  "food",
  "unrecognized"
]

# create categories in output directory if not exists
for category in categories:
  category_directory = os.path.join(output_directory, category)
  if not os.path.exists(category_directory):
    os.makedirs(category_directory)

categories_string = ",".join(categories)
output_categories_string = ":scores,".join(categories)
output_categories_string = f'{output_categories_string}:scores  '

classification_prompt = f"You would classify prompt into different categorys including: {categories_string} You will be prompted by people looking to classify images for training. The way to accomplish this is to score the image for each category and detemine the prompt should go into which category. You should give each category a scores from 1 to 100. If you couldn't fit the image into above categories labeled it as unrecognized. You must assign the prompt to atleast one categories. You response must includes all categories"
classification_hint = f'Please classify the image in different categories resulting scores from 1-100. You should carefully scores each categories and give the reason for each scores. Higher scores is more related to the categories. Lower scores is not that much related to the categoreis. Follow format: {output_categories_string} Reason: reason'


def get_scores(response):
  # find 'Reason:' index in response, get 0 to index string
  reason_index = response.find('Reason:')
  scores_string = response[0:reason_index].lower()
  print('response: ', response)
  print('scores_string: ', scores_string)

  split_char = ','
  if scores_string.find(',') != -1:
    # split string by ','
    scores = scores_string.split(',')
  else:
    first_category_index = scores_string.find(f'{categories[0]}:')
    second_category_index = scores_string.find(f'{categories[1]}:')
    score_part = scores_string[first_category_index:second_category_index]
    # remove empty space
    score_part = score_part.replace(' ','')
    # remain part should like: '10|' 
    # get the last part to find the split char
    split_char = score_part[-1]
    scores = scores_string.split(split_char)

  # define max_scores for looping to get max score
  max_scores = 0
  result = {}
  for category in categories:
    result[category] = 0
  print('result init:', result)
  # loop through scores, get score and category
  for score in scores:
    # split score by ':'
    score_split = score.strip().split(':')
    # get category
    category = score_split[0].replace(' ','_')
    # try to parse score to int
    try:
      # get score
      value = score_split[1]
      value = int(value)
    except:
      print('category error: ', category)
      value = 0
    result[category] = value
    print('category: ', category, ' score: ', value)
    # get max score
    if int(value) > max_scores:
      max_scores = int(value)
      max_category = category

  # extra bias of image classification
  # if scores of food is greater than 50, set max_category to food
  if result['food'] >= 50:
    max_category = 'food'
    max_scores = result['food']

  # if max category is landscape_only and people is greater than 50, set max_category to people
  if max_category == 'landscape_only' and result['people'] > 50:
    max_category = 'people'
    max_scores = result['people']
  print('result',result)
  print('max_category: ', max_category, ' max_scores: ', max_scores)  
  return max_category, max_scores

def write_text(filename,category_dir,content):
  output_path = os.path.join(category_dir, filename)
  # print('output_path: ', output_path)
  # create output_prompt_path file if not exists
  if not os.path.exists(output_path):
    open(output_path, 'a').close()
  with open(output_path, 'r+') as output_file:
    output_file.truncate(0)
    output_file.write(content)

# loop through all files in input directory
for filename in os.listdir(input_directory):
  print('--------------filename: ', filename)
  # skip if file is not jpg or png
  if not filename.endswith('.jpg') and not filename.endswith('.png'):
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

  # classify image first
  instruction_prompt = [{"text":classification_prompt}]

  # # 1st dialogue turn
  instruction = tokenizer.from_list_format(instruction_prompt)
  response, history = model.chat(tokenizer, query=instruction, history=None)
  # print('second instruction: ', response, '\n')

  query = tokenizer.from_list_format([
      {'image': image_path}, # Either a local path or an url
      {'text': classification_hint},
  ])
  response, history = model.chat(tokenizer, query=query, history=history)
  print('classification: ', response)
  classification = response
  
  max_category, max_scores = get_scores(response)
  category_dir = os.path.join(output_directory, max_category)

  suffix = ''
  prompt_file = f'{filename_without_extention}{suffix}.txt'
  # write prompt to classified folder
  # replace filename extention with .txt, filename extension is .jpg or .png
  write_text(prompt_file,category_dir,prompt)


  suffix = '_classification'
  classification_file = f'{filename_without_extention}{suffix}.txt'
  # write classified_log to classified folder
  # replace filename extention with .txt, filename extension is .jpg or .png
  write_text(classification_file,category_dir,classification)

  # copy image to classified folder
  output_image_path = os.path.join(category_dir, filename)
  print('output_image_path: ', output_image_path)
  # copy image from image_path to output_image_path, overwrite if exists
  shutil.copyfile(image_path, output_image_path)
  # copy image to classified folder


  print('--------------End: ', filename)
  # break
  