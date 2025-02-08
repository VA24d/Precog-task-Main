import os
from PIL import Image, ImageDraw, ImageFont
from wonderwords import RandomWord
import string

# ===============================
# Configuration parameters
# ===============================
output_dir = "easy_dataset"
num_samples = 1000        # Number of images to generate
image_width = 200         # Width of the generated image
image_height = 50         # Height of the generated image
font_size = 32            # Font size for the text
text_color = (0, 0, 0)    # Black text
bg_color = (255, 255, 255)  # White background
font_path = "fonts/opensans.ttf"

dir_path = output_dir+" "+str(num_samples)

# Ensure the output directory exists
os.makedirs(dir_path, exist_ok=True)

# Initialize the RandomWord generator
rw = RandomWord()

# Load the font
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    raise IOError(f"Cannot open font file {font_path}. Make sure the path is correct and the file exists.")

def contains_punctuation(word):
    """Check if a word contains any punctuation."""
    return any(char in string.punctuation for char in word)


def generate_image(word, image_width, image_height, font, text_color, bg_color):
    # Create a new image with a white background
    image = Image.new("RGB", (image_width, image_height), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    # Use textbbox to calculate the size of the text
    # textbbox returns (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate the position to center the text
    x = (image_width - text_width) // 2
    y = (image_height - text_height) // 2
    
    # Draw the text onto the image
    draw.text((x, y), word, fill=text_color, font=font)
    return image

# ===============================
# Main dataset generation loop
# ===============================
for i in range(num_samples):
    # Generate a random word
    word = rw.word(word_min_length=3, word_max_length=12).title()

    # Skip words with punctuation
    if contains_punctuation(word):
        continue
    
    # Generate the image containing the word
    image = generate_image(word, image_width, image_height, font, text_color, bg_color)
    
    # Save the image to the output directory.

    filename = os.path.join(dir_path, f"{i}_{word}.png")
    image.save(filename)

    # Optional: Print progress
    if (i + 1) % 100 == 0:
        print(f"Generated {i + 1}/{num_samples} images.")

print("Dataset generation complete!")
