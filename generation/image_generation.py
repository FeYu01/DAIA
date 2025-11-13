#!.venv/bin python3
"""
generate_nano_banana_image.py

Calls the Gemini 2.5 Flash Image ("Nano Banana") API and saves the generated
image(s) as PNG files in the current directory.
"""

import os
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image


def main():
    # Either set GEMINI_API_KEY in your environment, or replace below:
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyCMfgff61wveo5lZlEZOKp6wVD7JYqqHZ0")

    client = genai.Client(api_key=api_key)

    # Change this prompt to whatever you want Nano Banana to generate
    prompt = "A hyper-realistic 3D toy figurine of a banana astronaut on the moon"

    # Call the Nano Banana / Gemini 2.5 Flash Image model
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",  # Nano Banana model ID 
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                # Optional: "1:1", "16:9", "9:16", etc.
                aspect_ratio="1:1",
            ),
        ),
    )

    # Extract and save any image parts from the response
    img_count = 0
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                img_count += 1
                img_bytes = part.inline_data.data
                img = Image.open(BytesIO(img_bytes))

                filename = f"./nano_banana_{img_count}.png"
                img.save(filename)
                print(f"Saved image #{img_count} to {filename}")

    if img_count == 0:
        print("No image data returned by the model.")


if __name__ == "__main__":
    main()
