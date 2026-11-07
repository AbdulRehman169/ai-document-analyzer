import pytesseract
from PIL import Image

image = Image.open("/home/abdulrehman/Pictures/Screenshots/Screenshot_2026-11-07_11-34-48.png")
text = pytesseract.image_to_string(image)
print(text)