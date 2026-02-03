import av
import os 
import cv2
import time
import torch
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nanonetOCR import NanonetOCR
from decord import VideoReader, cpu
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from hybrid_search import hybrid_search

os.environ["TOKENIZERS_PARALLELISM"] = "false"
print('Current Env:', os.getenv("CONDA_DEFAULT_ENV"))

import warnings
warnings.filterwarnings("ignore")

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
    "Your task is to verify accuracy of the answer, count, relevance to the user query, strictly do not repeat the answer."
    "Classify the answer into ONE of the following categories:\n\n"
    "SUPPORTED:\n"
    "- The answer is supported by the OCR text by semantic or some keyword match not exact or clear visual evidence.\n\n"
    "DERIVABLE_FROM_STEPS:"
    "- The answer is not explicitly stated, but can be logically derived"
    " from the algorithm steps, equations, or procedures shown in the OCR.\n\n"
    "PARTIALLY_SUPPORTED:\n"
    "- Some evidence exists, but it may be incomplete or implicit (e.g., diagram labels).\n\n"
    "UNSUPPORTED:\n"
    "- No relevant OCR text or visual evidence supports the answer.\n\n"
    "Rules:\n"
    "- Do NOT use outside knowledge\n"
    "- Respond with ONLY: SUPPORTED - REASON, PARTIALLY_SUPPORTED - REASON, or UNSUPPORTED - REASON"
)

P_REFINE = (
    "You are refining and rewriting a previous answer using the last feedback.\n\n"
    "Rules:\n"
    "1. Use ONLY the OCR text and visible visual evidence.\n"
    "2. If feedback indicates the answer is not grounded, remove or correct unsupported claims, fix count if its a counting question.\n"
    "3. If grounding is impossible, respond exactly:\n"
    "\"The answer cannot be determined from the video.\"\n\n"
    "4. If feedback is DERIVABLE_FROM_STEPS:"
    "- Rewrite the answer by explicitly referring to the steps or equations shown in the OCR text."
    "- Do NOT introduce theory beyond the shown steps."
    "Produce ONLY the refined answer."
)


# Load the Qwen2.5-VL Model and Processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels=256*28*28,
    max_pixels=512*28*28
)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


# Helper Methods

def getManualFPS(video_path):

    # Calculates Duration of Video
    # And returns the fps value as desired by the user to sample that.
    # many frames only from the video.
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / source_fps
    video.release()

    # Consider only 32 frames regardless of the video length.
    # fps = target no. of frames / tot. len. of the Video
    fps = 64 / duration
    return fps


def read_video(video_path, num_frames):

    # We will sample 32 frames uniformly from the video for OCR purpose.
    # For other models the same 32 frames go into main model and the OCR.
    # But for Qwen we need to do the uniform sampling manually for the OCR purpose.
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return frames

# Create Chat Message

def create_chat_message(fps, video_path, system_prompt, user_prompt):

    message = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f'file://{video_path}',
                    "max_pixels": 360 * 420,
                    "fps": fps,
                },
                {
                    "type": "text", 
                    "text": user_prompt
                },
            ],
        }
    ]
    return message


# Method to run the model:

def run_model(chat_message, max_new_tokens=256):
    text = processor.apply_chat_template(
        chat_message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(chat_message)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def run_ocr(clip):
    print("\n--- Running OCR on Sampled Frames ---")
    ocr = NanonetOCR()
    ocr_outputs_list = []
    for i, frame in enumerate(clip):
        ocr_output = ocr.run_ocr(Image.fromarray(frame))
        ocr_outputs_list.append(f'Frame {i}: {ocr_output}')

        # print
        print(f'\n\nOCR for frame {i}:')
        print(ocr_output)

    with open('/home/aritrad/MSR-Project/frameworkocr_outputs.pkl', 'wb') as f:
        pickle.dump(ocr_outputs_list, f)

    return ocr_outputs_list


# Self Refinement Code

def self_refine_video_qa(ocr_outputs_list, max_iters=3):

    print("\n--- Starting Self-Refinement Process ---")

    # Get manually set fps for Qwen video input
    fps = getManualFPS(video_path)

    ocr_text = "\n".join(ocr_outputs_list)

    # Initial Generation (y0) 
    print("\n\nStep 1: Initial Answer Generation")
    gen_prompt = f"""
                OCR TEXT:
                {ocr_text}

                QUESTION:
                {question}
                """

    y = run_model(create_chat_message(fps, video_path, P_GEN, gen_prompt))
    print(f"\n\nInitial Answer (y0): {y}")

    for t in range(max_iters):
        print(f"\n--- Iteration {t+1}/{max_iters} ---")

        # Feedback (fb_t)
        print(f"\nStep 2: Generating Feedback for answer: '{y}'")
        fb_user_prompt = f"""
                    OCR TEXT:
                    {ocr_text}

                    QUESTION:
                    {question}

                    MODEL ANSWER:
                    {y}
                    """

        fb = run_model(create_chat_message(fps, video_path, P_FB, fb_user_prompt))
        print(f"\nFeedback (fb_{t}):\n{fb}")

        # Stop condition.
        if "PARTIALLY_SUPPORTED" in fb or "SUPPORTED" in fb or "DERIVABLE_FROM_STEPS" in fb and "UNSUPPORTED" not in fb:
            print("\nStop Condition Met: Answer is grounded. Returning current answer.")
            return y, True

        print("\nAnswer not sufficiently grounded. Proceeding to refinement.")
        # Refinement (y_{t+1})
        print("\nStep 3: Refining Answer based on Feedback")
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

        y = run_model(create_chat_message(fps, video_path, P_REFINE, refine_prompt))
        print(f"\nRefined Answer (y_{t+1}): {y}")

        if y.strip() == "cannot":
            print("\nRefined answer is 'cannot'. Stopping refinement.")
            return y, False

    print(f"\n--- Max iterations ({max_iters}) reached. Returning final answer. ---")
    return y, False


# Set Video Path & Question
# Test From EduVidQA
# https://www.youtube.com/watch?v=yndgIDO0zQQ (Linear Sorting)

video_path = '/home/aritrad/MSR-Project/samples/black-screen.mp4'
question = 'What are different object modelling technique in UML?'
num_frames_to_sample = 32

# Driving Code:
# Since, qwen has its own way to sample video frames which has been abstracted in the 
# create_chat_message function, we implement our own uniform frame sampling only for OCR purpose.
# OCR has been pre-run and outputs saved.
clip = read_video(video_path, num_frames_to_sample)
ocr_outputs_list = run_ocr(clip)

# Take only relevant OCR texts related to the question.
relevant_ocr_texts_list = hybrid_search(question, ocr_outputs_list, top_k=8)
print('\n\nLength of the relevant OCR texts:', len(relevant_ocr_texts_list))
print('\n\nRelevant OCR texts:', relevant_ocr_texts_list)
    
final_answer, grounded = self_refine_video_qa(relevant_ocr_texts_list)
print('\n\n**************** Final Answer *****************\n\n', final_answer.strip())

end_time = time.time()
print(f"\n\nExecution time: {end_time - start_time} seconds\n")