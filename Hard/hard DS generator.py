import os
import random
import string
from PIL import Image, ImageDraw, ImageFont
from wonderwords import RandomWord

# ===============================
# Configuration parameters
# ===============================
output_dir = "hard_dataset"
num_samples = 100        
image_width = 400        
image_height = 100        
initial_font_size = 32   # Starting font size

# Boolean toggle for applying rotation/distortion.
enable_distortion = False

# Get the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
fonts_dir = os.path.join(script_dir, "fonts")

# Dynamically load all .ttf fonts from the fonts directory.
def load_fonts():
    fonts = []
    try:
        for file in os.listdir(fonts_dir):
            if file.lower().endswith('.ttf'):
                font_path = os.path.join(fonts_dir, file)
                fonts.append(font_path)
        if not fonts:
            raise Exception("No .ttf fonts found in fonts directory")
        return fonts
    except Exception as e:
        print(f"Error loading fonts: {e}")
        exit(1)

fonts_list = load_fonts()

dir_path = os.path.join(output_dir, f"{num_samples}_images")
os.makedirs(dir_path, exist_ok=True)

# Initialize the RandomWord generator
rw = RandomWord()

# ===============================
# Helper Functions
# ===============================
def contains_punctuation(word):
    """Return True if the word contains any punctuation."""
    return any(char in string.punctuation for char in word)

def random_capitalize(word):
    """Randomly choose a capitalization for the word."""
    option = random.choice(["upper", "lower", "title", "random"])
    if option == "upper":
        return word.upper()
    elif option == "lower":
        return word.lower()
    elif option == "title":
        return word.title()
    elif option == "random":
        # Randomize each character's case
        return ''.join(random.choice([c.upper(), c.lower()]) for c in word)

def generate_background(bg_type, width, height, font_color=None):
    """
    Generate a background image.
      - bg_type: "clean", "black_dots", or "font_dots"
      - font_color: used for dots if bg_type == "font_dots"
    """
    # Start with a white background
    bg = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(bg)
    
    if bg_type == "clean":
        return bg

    # For the dot textures, choose a random dot size (tested between 1 and 5 pixels)
    dot_size = random.choice([1, 2, 3, 4, 5])
    # Choose a random number of dots
    num_dots = random.randint(20, 50)
    
    # Set the dot color: black for "black_dots", otherwise use the provided font_color.
    dot_color = (0, 0, 0) if bg_type == "black_dots" else (font_color if font_color else (0, 0, 0))
    
    for _ in range(num_dots):
        # Random location for each dot
        x = random.randint(0, width - dot_size)
        y = random.randint(0, height - dot_size)
        draw.ellipse((x, y, x + dot_size, y + dot_size), fill=dot_color)
    
    return bg

def generate_text_image(word, font_path, text_color, width, height, initial_font_size):
    """
    Create an RGBA image with the given word drawn in the center.
    This function adjusts the font size so that the text fits within (width, height).
    """
    # Start with the initial font size.
    font_size = initial_font_size
    # Create a temporary image to measure the text.
    temp_img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(temp_img)
    
    # Loop until the text fits or the font size becomes too small.
    while font_size > 5:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), word, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= width and text_height <= height:
            break
        font_size -= 1  # Reduce font size if text doesn't fit.
    
    # Create the final text image.
    text_img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_img)
    bbox = draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), word, fill=text_color, font=font)
    return text_img

def distort_image(image):
    """Apply a random distortion (rotation) to the image."""
    angle = random.uniform(-10, 10)  # Random rotation angle between -10 and 10 degrees
    return image.rotate(angle, expand=1, fillcolor=(255, 255, 255))

# ===============================
# Main Generation Loop
# ===============================
unique_words = set()
generated_count = 0

# To avoid infinite loops, set a maximum number of attempts.
max_attempts = num_samples * 10
attempts = 0

while generated_count < num_samples and attempts < max_attempts:
    attempts += 1

    # Generate a random word with length between 3 and 12 characters
    raw_word = rw.word(word_min_length=3, word_max_length=12)
    if raw_word is None:
        print("RandomWord returned None. Skipping.")
        continue
    
    # Reject words with punctuation
    if contains_punctuation(raw_word):
        continue

    # Apply random capitalization
    word = random_capitalize(raw_word)
    
    # Ensure the word is unique
    if word in unique_words:
        continue
    unique_words.add(word)
    
    # Randomly choose one of the fonts
    chosen_font_path = random.choice(fonts_list)
    
    # Randomize font color
    font_color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )
    
    # Choose a background type: 1/3 "clean", 1/3 "black_dots", 1/3 "font_dots"
    r_val = random.random()
    if r_val < 1/3:
        bg_type = "clean"
    elif r_val < 2/3:
        bg_type = "black_dots"
    else:
        bg_type = "font_dots"
    
    # Generate the background image based on the selected type.
    background = generate_background(bg_type, image_width, image_height, font_color)
    
    # Generate the text image (with transparent background) ensuring it fits.
    text_img = generate_text_image(word, chosen_font_path, font_color, image_width, image_height, initial_font_size)
    
    # Apply distortion if enabled
    if enable_distortion:
        text_img = distort_image(text_img)
    
    # Composite the (possibly distorted) text image on top of the background.
    background.paste(text_img, (0, 0), text_img)
    
    # The final composite image
    final_img = background
    
    # Save the final image with a filename that includes a unique count and the word.
    filename = os.path.join(dir_path, f"{generated_count}_{word}.png")
    try:
        final_img.save(filename)
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        continue
    
    generated_count += 1
    print(f"Generated {generated_count}/{num_samples} images (attempts: {attempts}).")

if generated_count < num_samples:
    print(f"Stopped after {attempts} attempts. Only generated {generated_count} unique images.")
else:
    print("Dataset generation complete!")
