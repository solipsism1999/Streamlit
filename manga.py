import json
from PIL import Image, ImageDraw, ImageFont
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
import io
import os
import warnings
import random
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import streamlit as st

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
STABILITY_API_KEY = st.secrets["STABILITY_API_KEY"]
def add_text_to_panel(text, panel_image):
    text_image = generate_text_image(text)
    result_image = Image.new('RGB', (panel_image.width, panel_image.height + text_image.height))
    result_image.paste(panel_image, (0, 0))
    result_image.paste(text_image, (0, panel_image.height))
    return result_image

def generate_text_image(text):
    # Define image dimensions
    width = 1024
    height = 128

    # Create a white background image
    image = Image.new('RGB', (width, height), color='white')

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Choose a font (Pillow's default font)
    font = ImageFont.truetype(font="manga-font.ttf", size=15)

    # Calculate text size
    text_width, text_height = draw.textsize(text, font=font)

    # Calculate the position to center the text horizontally and vertically
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # Define text color (black in this example)
    text_color = (0, 0, 0)

    # Add text to the image
    draw.text((x, y), text, fill=text_color, font=font)

    return image

def resize_and_add_border(image, target_size, border_size):
    resized_image = Image.new("RGB", target_size, "black")
    resized_image.paste(image, ((target_size[0] - image.width) // 2, (target_size[1] - image.height) // 2))
    return resized_image

def create_strip(images):
    # Desired grid size
    columns, rows = 2, 3

    # Calculate the size of the output image
    output_width = columns * images[0].width + (columns - 1) * 10  # 10 is the black border width
    output_height = rows * images[0].height + (rows - 1) * 10  # 10 is the black border width

    # Create a new image with the calculated size
    result_image = Image.new("RGB", (output_width, output_height), "white")

    # Combine images into a grid with black borders
    for i, img in enumerate(images):
        x = (i % columns) * (img.width + 10)  # 10 is the black border width
        y = (i // columns) * (img.height + 10)  # 10 is the black border width

        resized_img = resize_and_add_border(img, (images[0].width, images[0].height), 10)
        result_image.paste(resized_img, (x, y))

    return result_image.resize((1024, 1536))

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

template = """
You are a manga artist.

Your task is to take a brief storyline and divide it into six distinct manga panels. 
Each manga panel requires a detailed description, including the characters and the panel's background. 
Character descriptions should be precise, and avoid using character names within the panel descriptions. 
You must also provide concise text for each panel, consisting of no more than two short sentences that start with the character's name.

Example Input:
Characters: Lily is a cheerful girl with a ribbon in her hair. Max is a mischievous boy with a backpack.
Lily and Max set off on a thrilling journey, venturing deep into the mysterious cave of wonders.

Example Output:

# Panel 1
description: Two companions, a girl with a ribbon in her hair and a mischievous boy with a backpack, stand at the cave entrance, surrounded by eerie stalactites.

text:
```
Lily: The cave looks so mysterious!
Max: Let's explore and see what's inside.
```
# end

Short Scenario:
{scenario}

Split the scenario in 6 parts:
"
"""

def generate_panels(scenario):
    model = ChatOpenAI(model_name='gpt-3.5-turbo')

    human_message_prompt = HumanMessagePromptTemplate.from_template(template)

    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

    chat_prompt.format_messages(scenario=scenario)

    result = model(chat_prompt.format_messages(scenario=scenario))

    print(result.content)

    return extract_panel_info(result.content)

def extract_panel_info(text):
    panel_info_list = []
    panel_blocks = text.split('# Panel')

    for block in panel_blocks:
        if block.strip() != '':
            panel_info = {}
            
            # Extracting panel number
            panel_number = re.search(r'\d+', block)
            if panel_number is not None:
                panel_info['number'] = panel_number.group()
            
            # Extracting panel description
            panel_description = re.search(r'description: (.+)', block)
            if panel_description is not None:
                panel_info['description'] = panel_description.group(1)
            
            # Extracting panel text
            panel_text = re.search(r'text:\n```\n(.+)\n```', block, re.DOTALL)
            if panel_text is not None:
                panel_info['text'] = panel_text.group(1)
            
            panel_info_list.append(panel_info)
    return panel_info_list

os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'

seed = random.randint(0, 1000000000)

# Set up our connection to the API.
stability_api = client.StabilityInference(
    key="STABILITY_API_KEY",
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0"
)

def text_to_image(prompt):
    # Set up our initial generation parameters.
    answers = stability_api.generate(
        # prompt="rocket ship launching from forest with flower garden under a blue sky, masterful, ghibli",
        prompt=prompt,
        seed=seed, # If a seed is provided, the resulting generated image will be deterministic.
                        # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                        # Note: This isn't quite the case for CLIP Guided generations, which we tackle in the CLIP Guidance documentation.
        steps=30, # Amount of inference steps performed on image generation. Defaults to 30.
        cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
        width=1024, # Generation width, defaults to 1024 if not included.
        height=1024, # Generation height, defaults to 1024 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                    # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                    # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, display generated image.
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                global img
                img = Image.open(io.BytesIO(artifact.binary))
                return img 

def edit_image(input_image_path, prompt, output_image_name):
    img = Image.open(input_image_path)

    # Set up our initial generation parameters.
    answers = stability_api.generate(
        # prompt="crayon drawing of rocket ship launching from forest",
        prompt=prompt,
        init_image=img, # Assign our previously generated img as our Initial Image for transformation.
        start_schedule=0.6, # Set the strength of our prompt in relation to our initial image.
        seed=123463446, # If attempting to transform an image that was previously generated with our API,
                        # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
        steps=50, # Amount of inference steps performed on image generation. Defaults to 30.
        cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
        width=512, # Generation width, defaults to 512 if not included.
        height=512, # Generation height, defaults to 512 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                    # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                    # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, save generated image.
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                global img2
                img2 = Image.open(io.BytesIO(artifact.binary))
                img2.save(output_image_name + ".png") # Save our generated image with its seed number as the filename and the img2img suffix so that we know this is our transformed image.
                
