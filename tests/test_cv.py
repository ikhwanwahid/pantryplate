import sys
sys.path.insert(0, ".")

from src.vision.cv_inference import detect_ingredients_from_image

def test_detect_ingredients():
    result = detect_ingredients_from_image("data/test_fridge.jpg")
    print("Detected ingredients:", result)
    assert isinstance(result, list)