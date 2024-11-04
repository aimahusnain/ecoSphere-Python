from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import requests
from io import BytesIO

app = FastAPI()

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")


# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_transparency(image, threshold=100):
    """Remove semi-transparent pixels around the main object."""
    img_array = np.array(image)
    alpha = img_array[:, :, 3]
    semi_transparent = alpha < threshold
    kernel_close = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(semi_transparent.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
    kernel_erode = np.ones((2, 2), np.uint8)
    mask = cv2.erode(mask, kernel_erode)
    img_array[:, :, 3][mask == 1] = 0
    fully_transparent = img_array[:, :, 3] == 0
    img_array[:, :, :3][fully_transparent] = 0
    return Image.fromarray(img_array)

def calculate_full_size(text, font_path, image_width, image_height):
    """Calculate the largest font size that fits within the image dimensions."""
    font_size = 1
    target_width = image_width * 0.95
    target_height = image_height * 0.2
    while True:
        try:
            font = ImageFont.truetype(font_path, font_size)
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            if text_width > target_width or text_height > target_height:
                return font_size - 1
            font_size += 1
        except Exception:
            return max(1, font_size - 1)

def get_text_position(alignment, text_width, text_height, image_width, image_height):
    """Calculate text position based on alignment."""
    positions = {
        '1': ((image_width - text_width) // 2, 50),  # Top Center
        '2': ((image_width - text_width) // 2, (image_height - text_height) // 2),  # Center
        '3': ((image_width - text_width) // 2, image_height - text_height - 50),  # Bottom Center
        '4': (50, (image_height - text_height) // 2),  # Left Center
        '5': (image_width - text_width - 50, (image_height - text_height) // 2),  # Right Center
    }
    return positions.get(alignment, (0, 0))

def download_permanent_marker_font():
    """Download the Permanent Marker font if not already present."""
    font_url = "https://github.com/google/fonts/raw/main/apache/permanentmarker/PermanentMarker-Regular.ttf"
    font_path = "PermanentMarker-Regular.ttf"
    if not os.path.exists(font_path):
        response = requests.get(font_url)
        if response.status_code == 200:
            with open(font_path, 'wb') as f:
                f.write(response.content)
        else:
            raise HTTPException(status_code=404, detail="Font file could not be downloaded.")
    return font_path

@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...), 
    texty: str = Form("This is a default text."),
    text_size: str = Form("full"), 
    alignment: str = Form("1"), 
    text_color: str = Form("#ffffff")
):
    print(f"Received parameters: text={texty}, text_size={text_size}, alignment={alignment}, text_color={text_color}")

    try:
        # Load image
        input_image = Image.open(BytesIO(await file.read()))
        if input_image.mode != 'RGBA':
            input_image = input_image.convert('RGBA')
        
        # Prepare background as a copy of the original image
        background = input_image.copy()
        
        # Step 1: Draw text on a separate transparent layer
        text_layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        
        # Font handling
        font_path = download_permanent_marker_font()
        if text_size.lower() == 'full':
            font_size = calculate_full_size(texty, font_path, background.width, background.height)
        else:
            font_size = int(text_size) if text_size.isdigit() else 80  # Default to 80 if not specified
        
        font = ImageFont.truetype(font_path, font_size)
        bbox = font.getbbox(texty)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x, y = get_text_position(alignment, text_width, text_height, background.width, background.height)
        
        # Draw text on the transparent text layer
        draw.text((x, y), texty, font=font, fill=text_color)
        
        # Step 2: Remove background from the original input image to create transparent_image
        transparent_image = remove(input_image)
        transparent_image = clean_transparency(transparent_image)
        
        # Step 3: Composite the text layer with the transparent image layer on top
        combined_layer = Image.alpha_composite(background, text_layer)
        final_image = Image.alpha_composite(combined_layer, transparent_image)
        
        # Save final image
        output_path = f"output/processed_{file.filename}"
        os.makedirs("output", exist_ok=True)
        final_image.save(output_path, "PNG")
        
        return FileResponse(output_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
