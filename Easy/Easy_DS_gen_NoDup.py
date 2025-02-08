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

dir_path = output_dir + " " + str(num_samples)
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
    """Generate an image with the given word."""
    image = Image.new("RGB", (image_width, image_height), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    # Calculate text bounding box using textbbox (available in Pillow 10+)
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
unique_words = set()
generated_count = 0

while generated_count < num_samples:
    # Generate a random word with length between 3 and 12 characters.
    word = rw.word(word_min_length=3, word_max_length=12).title()
    
    # Reject words with punctuation
    if contains_punctuation(word):
        continue
    
    # Skip the word if it has already been generated
    if word in unique_words:
        continue
    
    # Add the unique word to our set
    unique_words.add(word)
    
    # Generate the image containing the word
    image = generate_image(word, image_width, image_height, font, text_color, bg_color)
    
    # Save the image to the output directory with a unique filename.
    filename = os.path.join(dir_path, f"{generated_count}_{word}.png")
    image.save(filename)
    
    generated_count += 1
    
    # Optional: Print progress
    if generated_count % 100 == 0:
        print(f"Generated {generated_count}/{num_samples} images.")

print("Dataset generation complete!")
