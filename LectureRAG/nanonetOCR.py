#!/usr/bin/env python
# coding: utf-8

# Reference: https://huggingface.co/nanonets/Nanonets-OCR-s

from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText


import os 

class NanonetOCR:
    def __init__(self, model_path="nanonets/Nanonets-OCR-s"):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, 
            dtype="auto", 
            device_map="auto", 
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        #self.prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
        self.prompt = """
        You are extracting text from lecture slides for question answering.

        Instructions:
        1. Extract ONLY visible, meaningful text that conveys educational content.
        2. Do NOT add descriptions of images, diagrams, or figures.
        3. Do NOT include page numbers, watermarks, headers, footers, logos, hash symbol, asterisk or decorative text.
        4. Do NOT add HTML tags or any formatting.
        5. Do NOT rewrite, paraphrase, summarize, or infer missing text.
        6. Preserve bullet points, lists, and line breaks exactly as shown.
        7. If a slide contains a diagram with labeled text, extract ONLY the visible labels.
        8. If no meaningful text is visible on the slide, return an "No visuals".

        Output format:
        - Plain text only
        - One bullet or line per concept
        - No explanations
        - No metadata
        """

    def run_ocr(self, image):
        if isinstance(image, str):
            image_path = image
            image = Image.open(image_path)
            image_uri = f"file://{image_path}"
        else:
            image_uri = "file://image.png"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": self.prompt},
            ]},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=True,
            temperature=0.7
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]

if __name__ == "__main__":
    print(os.getenv("CONDA_DEFAULT_ENV"))
    ocr = NanonetOCR()
    image_path = "/home/aritrad/test_images/electric.PNG"
    print(ocr.run_ocr(image_path))
