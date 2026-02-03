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
from decord import VideoReader, cpu
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer, AutoProcessor
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

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
    "- The answer is directly supported by the OCR text or clear visual evidence.\n\n"
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
    "Produce ONLY the refined answer."
)

# Helper Methods

def create_chat_message(system_prompt, user_prompt):

    message = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"<|video|> {user_prompt}"
        },
        {
            "role": "assistant",
            "content": ""
        }
    ]
    return message



# Load the Model

model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path, 
    torch_dtype=torch.half, 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)
_ = model.eval().cuda()

# Method to run the model

def run_model(prompt_message, clip, max_new_tokens=256):
    inputs = processor(
        prompt_message, 
        images=None, 
        videos=clip
    )

    inputs.to('cuda')
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens':512,
        'decode_text':True,
    })

    answer = model.generate(**inputs)
    return answer[0]

# print the config
# model.config

# Preparing the video inputs

MAX_NUM_FRAMES=16

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    #print(frames)
    print('num frames:', len(frames))
    return frames


# Set Video Path

video_path = ['/home/aritrad/MSR-Project/random/insertion_sort.mp4']
video_frames = [encode_video(_) for _ in video_path]


# Visualize the sampled frames

def visualize_frames():
    frames = video_frames[0]
    cols = 8
    rows = int(np.ceil(len(frames) / cols))
    plt.figure(figsize=(20, 6))
    
    for i in range(len(frames)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(frames[i])
        plt.title(f"Frame {i}")
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

    with open('/home/aritrad/MSR-Project/frameworkocr_outputs.pkl', 'wb') as f:
        pickle.dump(ocr_outputs_list, f)

    return ocr_outputs_list


def hybrid_search(query, documents, top_k=4):
    print("\n--- Running Hybrid Search ---")
    if not documents:
        return []

    # BM25
    tokenized_corpus = [doc.split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Semantic Search
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
    
    # Reciprocal Rank Fusion
    k = 60
    rrf_scores = {}
    
    # Rank by BM25
    bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
    for rank, idx in enumerate(bm25_ranked_indices):
        if idx not in rrf_scores: rrf_scores[idx] = 0
        rrf_scores[idx] += 1 / (k + rank + 1)
        
    # Rank by Cosine
    cosine_ranked_indices = np.argsort(cosine_scores.cpu().numpy())[::-1]
    for rank, idx in enumerate(cosine_ranked_indices):
        if idx not in rrf_scores: rrf_scores[idx] = 0
        rrf_scores[idx] += 1 / (k + rank + 1)
        
    # Sort by RRF score
    sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    
    top_docs = [documents[i] for i in sorted_indices[:top_k]]
    return top_docs


# Self Refinement Code

def self_refine_video_qa(question, ocr_outputs_list, clip, max_iters=3):
    print("\n--- Starting Self-Refinement Process ---")

    ocr_text = "\n".join(ocr_outputs_list)

    # Initial Generation (y0) 
    print("\n\nStep 1: Initial Answer Generation")
    gen_prompt = f"""
                OCR TEXT:
                {ocr_text}

                QUESTION:
                {question}
                """

    y = run_model(create_chat_message(P_GEN, gen_prompt), clip)
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

        fb = run_model(create_chat_message(P_FB, fb_user_prompt), clip)
        print(f"\nFeedback (fb_{t}):\n{fb}")

        # Stop condition (as in paper)
        if "PARTIALLY_SUPPORTED" in fb or "SUPPORTED" in fb and "UNSUPPORTED" not in fb:
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

        y = run_model(create_chat_message(P_REFINE, refine_prompt), clip)
        print(f"\nRefined Answer (y_{t+1}): {y}")

        if y.strip() == "cannot":
            print("\nRefined answer is 'cannot'. Stopping refinement.")
            return y, False

    print(f"\n--- Max iterations ({max_iters}) reached. Returning final answer. ---")
    return y, False


# Question 1: 4 min sample
question =  'How many element is present in the array shown?'

# Driving Code:
# OCR Disabled for MplugOwl due to env dependencies.
# OCR has been pre-run and outputs saved.
# ocr_outputs_list = run_ocr(clip)

# Read pre-saved OCR outputs
# Or, run OCR using nanonetOCR on the fly.
with open('/home/aritrad/MSR-Project/framework/frameworkocr_insertionsort_outputs.pkl', 'rb') as f:
    ocr_outputs_list = pickle.load(f)

print('\n\nLoaded OCR outputs from pickle file.', ocr_outputs_list)

# Take only relevant OCR texts related to the question.
relevant_ocr_texts = hybrid_search(question, ocr_outputs_list, top_k=5)
print('\n\nRelevant OCR texts:', relevant_ocr_texts)
    
final_answer, grounded = self_refine_video_qa(question, relevant_ocr_texts, video_frames)
print('\n\n**************** Final Answer *****************\n\n', final_answer.strip())

end_time = time.time()
print(f"\n\nExecution time: {end_time - start_time} seconds\n")