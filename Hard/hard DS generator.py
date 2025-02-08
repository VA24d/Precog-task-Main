import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from wonderwords import RandomWord

# ===============================
# Configuration Parameters (modifiable)
# ===============================
OUTPUT_DIR = "hard_dataset"            # Base output directory for generated images
NUM_SAMPLES = 100                      # Total number of images to generate
IMAGE_WIDTH = 800                      # Width of each generated image
IMAGE_HEIGHT = 200                     # Height of each generated image
INITIAL_FONT_SIZE = 64                 # Starting font size (used as a reference for scaling)

ENABLE_DISTORTION = False              # If True, applies random rotation/distortion to text
USE_NOISE = True                       # If True, noise background is available
USE_DOTS = True                        # If True, dots background is available

FILENAME_FORMAT = "{word}.png" # Format for saving files

# Text rendering configuration
MARGIN_RATIO = 0.9                     # Fraction of image dimensions used for text boundaries (to avoid overflow)
MIN_FONT_SIZE = 10                     # Minimum font size allowed when scaling text

# ===============================
# End of Configuration Parameters
# ===============================

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

# Create output directory
dir_path = os.path.join(OUTPUT_DIR, f"{NUM_SAMPLES}_images")
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
    """Randomize the capitalization for the word."""
    return ''.join(random.choice([c.upper(), c.lower()]) for c in word)

def generate_plain_background(width, height):
    """Generate a plain white background."""
    return Image.new("RGB", (width, height), color=(255, 255, 255))

def generate_noise_background(width, height):
    """
    Generate a noise background.
    With a 50% chance, generate full noise (each pixel is random).
    Otherwise, generate subtle noise (a white base with small random variations).
    The subtle noise amplitude is chosen randomly between 1% and 5% of 255.
    """
    if random.random() < 0.5:
        # Full noise: every pixel is random.
        noise_arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(noise_arr, 'RGB')
    else:
        # Subtle noise: white base with small variations using a Gaussian distribution.
        amp = random.uniform(0.01, 0.05) * 255  # amplitude between 1% and 5% of 255
        base = np.full((height, width, 3), 255, dtype=np.float32)
        # Use a standard deviation of half the amplitude for subtle variation.
        noise = np.random.normal(0, amp / 2, (height, width, 3))
        noisy_arr = np.clip(base + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_arr, 'RGB')

def generate_dots_background(width, height, font_color=None):
    """
    Generate a background with random dots.
    Uses a higher number of small dots for a finer appearance.
    """
    bg = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(bg)
    dot_size = random.choice([1, 2, 3])
    num_dots = random.randint(100, 200)  # More dots for a finer texture.
    # Use provided font_color if available; otherwise, a default dark gray.
    dot_color = font_color if font_color else (50, 50, 50)
    for _ in range(num_dots):
        x = random.randint(0, width - dot_size)
        y = random.randint(0, height - dot_size)
        draw.ellipse((x, y, x + dot_size, y + dot_size), fill=dot_color)
    return bg

def generate_background(width, height, font_color=None):
    """
    Generate a background image based on the configuration.
    If both USE_NOISE and USE_DOTS are enabled, randomly choose one.
    If neither is enabled, return a plain white background.
    """
    if USE_NOISE and USE_DOTS:
        bg_type = random.choice(["noise", "dots"])
    elif USE_NOISE:
        bg_type = "noise"
    elif USE_DOTS:
        bg_type = "dots"
    else:
        bg_type = "plain"

    if bg_type == "noise":
        return generate_noise_background(width, height)
    elif bg_type == "dots":
        return generate_dots_background(width, height, font_color)
    else:
        return generate_plain_background(width, height)

def generate_text_image(word, font_path, text_color, width, height, initial_font_size,
                        margin_ratio=MARGIN_RATIO, min_font_size=MIN_FONT_SIZE):
    """
    Create an RGBA image with the given word drawn in the center.
    The text is sized (via a binary search) to be as large as possible while
    fitting within a margin of the image dimensions (default 90% of width/height),
    and not falling below a minimum font size.
    
    To avoid cutoffs due to negative offsets in the bounding box, the drawing
    position is adjusted by the bbox offset.
    """
    # Allowed dimensions for text (with margin)
    allowed_width = int(width * margin_ratio)
    allowed_height = int(height * margin_ratio)

    # Use binary search to find the maximum font size that fits
    low = min_font_size
    high = 1000  # an arbitrary high limit
    best = low

    dummy_img = Image.new("RGBA", (width, height))
    dummy_draw = ImageDraw.Draw(dummy_img)

    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(font_path, mid)
        bbox = dummy_draw.textbbox((0, 0), word, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= allowed_width and text_height <= allowed_height:
            best = mid  # this font size fits; try a larger one
            low = mid + 1
        else:
            high = mid - 1

    # Use the best (largest) font size that fits
    font = ImageFont.truetype(font_path, best)
    bbox = dummy_draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    # Adjust the drawing position by subtracting the bbox offsets
    x = (width - text_width) // 2 - bbox[0]
    y = (height - text_height) // 2 - bbox[1]

    text_img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_img)
    draw.text((x, y), word, fill=text_color, font=font)
    return text_img

def distort_image(image):
    """Apply a random rotation (distortion) to the image."""
    angle = random.uniform(-10, 10)  # Rotation angle between -10 and 10 degrees.
    return image.rotate(angle, expand=1, fillcolor=(255, 255, 255))

# ===============================
# Main Generation Loop
# ===============================
unique_words = set()
generated_count = 0

# To avoid infinite loops, set a maximum number of attempts.
max_attempts = NUM_SAMPLES * 10
attempts = 0

while generated_count < NUM_SAMPLES and attempts < max_attempts:
    attempts += 1

    # Generate a random word (length between 3 and 12 characters).
    raw_word = rw.word(word_min_length=3, word_max_length=12)
    if raw_word is None:
        print("RandomWord returned None. Skipping.")
        continue

    if contains_punctuation(raw_word):
        continue

    # Randomize capitalization.
    word = random_capitalize(raw_word)

    # Ensure uniqueness.
    if word in unique_words:
        continue
    unique_words.add(word)

    # Choose a random font.
    chosen_font_path = random.choice(fonts_list)

    # Randomize font color.
    font_color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

    # Generate the background based on the selected configuration.
    background = generate_background(IMAGE_WIDTH, IMAGE_HEIGHT, font_color)

    # Generate the text image (with transparent background).
    text_img = generate_text_image(word, chosen_font_path, font_color,
                                   IMAGE_WIDTH, IMAGE_HEIGHT, INITIAL_FONT_SIZE)

    # Apply distortion if enabled.
    if ENABLE_DISTORTION:
        text_img = distort_image(text_img)

    # Composite the text onto the background.
    background.paste(text_img, (0, 0), text_img)
    final_img = background

    # Save the final image.
    filename = os.path.join(dir_path, FILENAME_FORMAT.format(count=generated_count, word=word))
    try:
        final_img.save(filename)
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        continue

    generated_count += 1
    print(f"Generated {generated_count}/{NUM_SAMPLES} images (attempts: {attempts}).")

if generated_count < NUM_SAMPLES:
    print(f"Stopped after {attempts} attempts. Only generated {generated_count} unique images.")
else:
    print("Dataset generation complete!")
