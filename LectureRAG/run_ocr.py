from nanonetOCR import NanonetOCR

ocr = NanonetOCR()
image_path = "/home/aritrad/test_images/electric.PNG"
print('\n\n\n', ocr.run_ocr(image_path))