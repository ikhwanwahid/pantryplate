from src.cv_inference import detect_ingredients_from_image
from src.ingredient_normalizer import normalize

st.header("What's in your fridge?")

input_mode = st.radio("Input mode", ["Type ingredients", "Upload fridge photo"])

if input_mode == "Upload fridge photo":
    uploaded = st.file_uploader("Upload a fridge photo", type=["jpg", "png"])
    if uploaded:
        with open("temp_fridge.jpg", "wb") as f:
            f.write(uploaded.read())
        raw_labels = detect_ingredients_from_image("temp_fridge.jpg")
        ingredients = normalize(raw_labels)
        st.success(f"Detected: {', '.join(ingredients)}")
        # pass `ingredients` into your recommender as usual