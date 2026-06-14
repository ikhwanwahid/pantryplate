from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def detect_ingredients_from_image(image_path: str) -> list[str]:
    """Send fridge image to Gemini Vision, return list of detected ingredients."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[
            types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
            "List every food ingredient you can see in this fridge image. Reply with only a comma-separated list of ingredient names, nothing else. Example: eggs, milk, carrots, cheese"
        ]
    )

    raw = response.text
    return [item.strip().lower() for item in raw.split(",") if item.strip()]