import os 
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print('Current Env:', os.getenv("CONDA_DEFAULT_ENV"))

import av
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from PIL import Image
from nanonetOCR import NanonetOCR

start_time = time.time()

# Load the Model

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", use_fast=True)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    device_map='auto'
)


# print the config
# model.config

# Preparing the video and image inputs
# In order to read the video we'll use av and sample 8 frames. You can try to sample more frames 
# if the video is long. The model was trained with 32 frames, but can ingest more as long as we're 
# in the LLM backbone's max sequence length range.

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# Set Video Path
video_path = '/home/aritrad/MSR-Project/samples/black-screen.mp4'
container = av.open(video_path)

# Frame Sampling
# Sample uniformly 8 frames from the video (we can sample more for longer videos)
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 16).astype(int)
clip = read_video_pyav(container, indices)


# Perform OCR on each frame
ocr = NanonetOCR()
ocr_outputs_list = []
for i, frame in enumerate(clip):
    ocr_output = ocr.run_ocr(Image.fromarray(frame))
    ocr_outputs_list.append(f'Frame {i}: {ocr_output}')

    # print
    print(f'\n\nOCR for frame {indices[i]}:')
    print(ocr_output)


# Visualize the sampled frames
def visualize_frames():
    
    #print("Selected indices:", indices)
    cols = 8
    rows = int(np.ceil(len(indices) / cols))
    
    plt.figure(figsize=(20, 6))
    for i in range(len(indices)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(clip[i])
        plt.title(f"Frame {indices[i]}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig('/home/aritrad/MSR-Project/framework/sampled_frames.jpeg')
    #plt.show()

visualize_frames()

# Prepare a prompt and generate.

# In the prompt, you can refer to video using the special <video> or <image> token. To indicate which text comes from a human vs. the model, one uses USER and ASSISTANT respectively (note: it's true only for this checkpoint). The format looks as follows:
# USER: <video>\n<prompt> ASSISTANT:
# In other words, you always need to end your prompt with ASSISTANT:.
# Manually adding USER and ASSISTANT to your prompt can be error-prone since each checkpoint has its own prompt format expected, depending on the backbone language model. Luckily we can use apply_chat_template to make it easier.
# Chat templates are special templates written in jinja and added to the model's config. Whenever we call apply_chat_template, the jinja template in filled in with your text instruction.

def create_chat_message(prompt):
    
    message = [
        {
            "role": "system",
            "content": (
                "You are a vision-language model analyzing lecture videos.\n\n"
                "IMPORTANT:\n"
                "Text visible in the video (slides, blackboards, handwritten content) "
                "has ALREADY been extracted using a dedicated OCR system and will be "
                "provided to you explicitly.\n\n"
                "You MUST follow these rules:\n"
                "1. DO NOT perform OCR yourself or infer unreadable text from visuals.\n"
                "2. Use ONLY the provided OCR text as the authoritative source of written content.\n"
                "3. Use video frames ONLY for visual context (e.g., diagrams, figures, gestures, pointing).\n"
                "4. Explicitly list the provided OCR text before reasoning.\n"
                "5. Answer the question using ONLY:\n"
                "   - the provided OCR text, and\n"
                "   - clearly visible non-textual visual evidence.\n"
                "6. If the answer cannot be derived strictly from the provided OCR text "
                "and visible visuals, respond exactly with:\n"
                "\"The answer cannot be determined from the video.\"\n"
                "7. Do NOT guess, infer missing text, or use outside knowledge."
            )
        },
        {
              "role": "user",
              "content": [
                      {
                          "type": "text", 
                          "text": f"{prompt}"
                      },
                      {
                          "type": "video"
                      },
                  ],
          },
    ]
    return message


# Question 1: 4 min sample
ocr_outputs = '\n'.join(ocr_outputs_list)
question = 'What is the runtime of insertion sort algorithm as shown in the video?'
prompt = 'We supply the following OCR outputs from the video frames:\n' + ocr_outputs + '\nStrictly refuse to answer in one-line if the OCR outputs is mostly empty or not relvant to the question, now based on these OCR outputs and the visuals answer to the point: ' + question
message = create_chat_message(prompt)

prompt = processor.apply_chat_template(message, add_generation_prompt=True)
inputs = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)
generate_kwargs = {
    "max_new_tokens": 32,
    "do_sample": False,
    "temperature": 0.0
}


# Generate Output.
output = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print('\n\n****************Final Answer*****************\n\n', generated_text[0].split('ASSISTANT:')[1].strip())

end_time = time.time()
print(f"\n\nExecution time: {end_time - start_time} seconds\n")