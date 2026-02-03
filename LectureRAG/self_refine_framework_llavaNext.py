import os 
import time
import pickle
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print('Current Env:', os.getenv("CONDA_DEFAULT_ENV"))

import warnings
warnings.filterwarnings("ignore")

import av
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nanonetOCR import NanonetOCR
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

start_time = time.time()


# Intialize Prompts

P_GEN = (
    "You are a vision-language model answering questions about lecture videos. "
    "You are given:\n"
    "1. A question\n"
    "2. OCR text extracted from video frames\n"
    "3. The video frames themselves\n\n"
    "Answer the question using the OCR text and clearly visible visual evidence. "
    "If the answer cannot be determined, say exactly: "
    "\"The answer cannot be determined from the video.\""
)

P_FB = (
    "You are verifying an answer based on OCR text and video frames.\n\n"
    "Classify the answer into ONE of the following categories:\n\n"
    "SUPPORTED:\n"
    "- The answer is directly supported by the OCR text or clear visual evidence.\n\n"
    "PARTIALLY_SUPPORTED:\n"
    "- Some evidence exists, but it may be incomplete or implicit (e.g., diagram labels).\n\n"
    "UNSUPPORTED:\n"
    "- No relevant OCR text or visual evidence supports the answer.\n\n"
    "Rules:\n"
    "- Do NOT use outside knowledge\n"
    "- Respond with ONLY one word: SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED"
)

P_REFINE = (
    "You are refining a previous answer using feedback.\n\n"
    "Rules:\n"
    "1. Use ONLY the OCR text and visible visual evidence.\n"
    "2. If feedback indicates the answer is not grounded, remove or correct unsupported claims.\n"
    "3. If grounding is impossible, respond exactly:\n"
    "\"The answer cannot be determined from the video.\"\n\n"
    "Produce ONLY the refined answer."
)

# Helper Methods

def build_chat_template(system_prompt, user_prompt):
    return [
        {
            "role": "system", 
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": user_prompt
                },
                {
                    "type": "video"
                }
            ]
        }
    ]


# Load the Model

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", use_fast=True)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    device_map='auto'
)

# Method to run the model

def run_model(prompt_message, clip, max_new_tokens=256):
    prompt = processor.apply_chat_template(prompt_message, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        videos=clip,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0
    )

    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return decoded.split("ASSISTANT:")[-1].strip()


# print the config
# model.config

# Preparing the video and image inputs
# In order to read the video we'll use av and sample 8 frames. You can try to sample more frames 
# if the video is long. The model was trained with 32 frames, but can ingest more as long as we're 
# in the LLM backbone's max sequence length range.

def read_video_pyav(container, indices):
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
video_path = '/home/aritrad/MSR-Project/random/insertion_sort.mp4'
container = av.open(video_path)

# Frame Sampling
# Sample uniformly 32 frames from the video (we can sample more for longer videos)
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 32).astype(int)
clip = read_video_pyav(container, indices)


# Visualize the sampled frames

def visualize_frames():
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

# Call the function to visualize frames.
visualize_frames()


# Perform OCR on each frame

def run_ocr(clip):
    print("\n--- Running OCR on Sampled Frames ---")
    ocr = NanonetOCR()
    ocr_outputs_list = []
    for i, frame in enumerate(clip):
        ocr_output = ocr.run_ocr(Image.fromarray(frame))
        ocr_outputs_list.append(f'Frame {i}: {ocr_output}')

        # print
        print(f'\n\nOCR for frame {indices[i]}:')
        print(ocr_output)

    with open('/home/aritrad/MSR-Project/framework/frameworkocr_insertionsort_outputs.pkl', 'wb') as f:
        pickle.dump(ocr_outputs_list, f)

    return ocr_outputs_list


# Self Refinement Code

def self_refine_video_qa(question, ocr_outputs_list, clip, max_iters=3):
    print("\n--- Starting Self-Refinement Process ---")

    ocr_text = "\n".join(ocr_outputs_list)

    # Initial Generation (y0) 
    print("\nStep 1: Initial Answer Generation")
    gen_prompt = f"""
                OCR TEXT:
                {ocr_text}

                QUESTION:
                {question}
                """

    y = run_model(build_chat_template(P_GEN, gen_prompt), clip)
    print(f"Initial Answer (y0): {y}")

    for t in range(max_iters):
        print(f"\n--- Iteration {t+1}/{max_iters} ---")

        # Feedback (fb_t)
        print(f"Step 2: Generating Feedback for answer: '{y}'")
        fb_user_prompt = f"""
                    OCR TEXT:
                    {ocr_text}

                    QUESTION:
                    {question}

                    MODEL ANSWER:
                    {y}
                    """

        fb = run_model(build_chat_template(P_FB, fb_user_prompt), clip)
        print(f"Feedback (fb_{t}):\n{fb}")

        # Stop condition (as in paper)
        if "PARTIALLY_SUPPORTED" in fb or "SUPPORTED" in fb:
            print("Stop Condition Met: Answer is grounded. Returning current answer.")
            return y, True

        print("Answer not sufficiently grounded. Proceeding to refinement.")
        # Refinement (y_{t+1})
        print("Step 3: Refining Answer based on Feedback")
        refine_prompt = f"""
                        OCR TEXT:
                        {ocr_text}

                        QUESTION:
                        {question}

                        PREVIOUS ANSWER:
                        {y}

                        FEEDBACK:
                        {fb}
                        """

        y = run_model(build_chat_template(P_REFINE, refine_prompt), clip)
        print(f"Refined Answer (y_{t+1}): {y}")

        if y.strip() == "cannot":
            print("Refined answer is 'cannot'. Stopping refinement.")
            return y, False

    print(f"\n--- Max iterations ({max_iters}) reached. Returning final answer. ---")
    return y, False


# Question 1: 4 min sample
question = 'UML is adapted by which group and in which year?'

# Driving Code:

ocr_outputs_list = run_ocr(clip)
#print('\n\nLoaded OCR outputs from pickle file.', ocr_outputs_list)  # print first 2 for verification

final_answer, grounded = self_refine_video_qa(question, ocr_outputs_list, clip)
print('\n\n****************Final Answer*****************\n\n', final_answer)

end_time = time.time()
print(f"\n\nExecution time: {end_time - start_time} seconds\n")