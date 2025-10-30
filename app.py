"""
🐕 SINGLE BREED PREDICTION - SIMPLIFIED VERSION
================================================
Shows only the top predicted breed (no top 5)
Uses YOUR exact breed names from training!
"""

# Install packages
print("📦 Installing packages...")
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("gradio")
print("✅ Packages installed!\n")

# ============================================
# UPLOAD YOUR MODEL
# ============================================
print("="*60)
print("📤 UPLOAD YOUR MODEL FILE")
print("="*60)
from google.colab import files
uploaded = files.upload()
model_filename = list(uploaded.keys())[0]
print(f"✅ Model uploaded: {model_filename}\n")

# ============================================
# SETUP
# ============================================
print("🔧 Setting up the AI model...")

import numpy as np
from PIL import Image
import gradio as gr
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = keras.models.load_model(model_filename)
print("✅ Model loaded!")

# YOUR EXACT BREED NAMES FROM YOUR TRAINING (in the correct order!)
breed_names = [
    "affenpinscher",
    "afghan_hound",
    "african",
    "airedale",
    "american_terrier",
    "appenzeller",
    "australian_terrier",
    "basenji",
    "basset_hound",
    "beagle",
    "bedlington_terrier",
    "bernese_mountain",
    "blenheim_spaniel",
    "blood_hound",
    "bluetick",
    "border_collie",
    "border_terrier",
    "borzoi",
    "boston_bulldog",
    "bouvier",
    "boxer",
    "brabancon",
    "briard",
    "brittany_spaniel",
    "bull_mastiff",
    "cairn_terrier",
    "cardigan_corgi",
    "chesapeake_retriever",
    "chihuahua",
    "chow",
    "clumber",
    "cocker_spaniel",
    "coonhound",
    "curly_retriever",
    "dandie_terrier",
    "dhole",
    "dingo",
    "doberman",
    "english_hound",
    "english_setter",
    "english_sheepdog",
    "english_springer",
    "entlebucher",
    "eskimo",
    "flatcoated_retriever",
    "fox_terrier",
    "french_bulldog",
    "german_pointer",
    "germanshepherd",
    "golden_retriever",
    "gordon_setter",
    "great_dane",
    "groenendael",
    "husky",
    "ibizan_hound",
    "irish_setter",
    "irish_spaniel",
    "irish_terrier",
    "irish_wolfhound",
    "italian_greyhound",
    "japanese_spaniel",
    "keeshond",
    "kelpie",
    "kerryblue_terrier",
    "komondor",
    "kuvasz",
    "labrador",
    "lakeland_terrier",
    "leonberg",
    "lhasa",
    "malamute",
    "malinois",
    "maltese",
    "mexicanhairless",
    "miniature_pinscher",
    "miniature_poodle",
    "mix",
    "newfoundland",
    "norfolk_terrier",
    "norwegian_elkhound",
    "norwich_terrier",
    "otterhound",
    "papillon",
    "pekinese",
    "pembroke",
    "pomeranian",
    "pug",
    "pyrenees",
    "redbone",
    "rottweiler",
    "saluki",
    "samoyed",
    "schipperke",
    "scottish_deerhound",
    "scottish_terrier",
    "sealyham_terrier",
    "shetland_sheepdog",
    "shihtzu",
    "silky_terrier",
    "staffordshire_bullterrier",
    "standard_poodle",
    "stbernard",
    "sussex_spaniel",
    "swiss_mountain",
    "tibetan_mastiff",
    "tibetan_terrier",
    "toy_poodle",
    "toy_terrier",
    "vizsla",
    "walker_hound",
    "weimaraner",
    "welsh_spaniel",
    "westhighland_terrier",
    "wheaten_terrier",
    "whippet",
    "yorkshire_terrier"
]

print(f"✅ Loaded {len(breed_names)} dog breeds\n")

# Breed descriptions
DESCRIPTIONS = {
    "golden_retriever": "Friendly, intelligent, and devoted. Great family dogs with a beautiful golden coat.",
    "germanshepherd": "Confident, courageous, and smart. Excellent working dogs and loyal companions.",
    "labrador": "Outgoing, even-tempered, and gentle. One of the most popular family dogs.",
    "beagle": "Merry, friendly, and curious. Known for their excellent sense of smell.",
    "french_bulldog": "Calm, courageous, and friendly. Distinctive bat ears and compact build.",
    "pug": "Charming, loving, and mischievous. Compact with a wrinkled face.",
    "rottweiler": "Loyal, loving, and confident guardian. Strong and powerful working dog.",
    "chihuahua": "Tiny but mighty! Bold, confident, and devoted to their owners.",
    "husky": "Energetic, friendly, and mischievous. Beautiful blue or multi-colored eyes.",
    "boxer": "Fun-loving, bright, and active. Great with families and children.",
    "yorkshire_terrier": "Feisty, affectionate, and brave. Small dog with big personality.",
    "doberman": "Loyal, fearless, and alert. Excellent guard dogs and companions.",
    "shihtzu": "Affectionate, playful, and outgoing. Originally bred as Chinese royal dogs.",
    "pomeranian": "Bold, lively, and inquisitive. Fluffy coat and fox-like face.",
    "border_collie": "Highly intelligent, energetic, and athletic. Best herding dog breed.",
    "stbernard": "Gentle giant, patient, and friendly. Famous for mountain rescue work.",
}

def get_description(breed_name):
    return DESCRIPTIONS.get(breed_name, f"A wonderful dog breed! {breed_name.replace('_', ' ').title()}s make excellent companions.")

# ============================================
# PREDICTION FUNCTION - SINGLE BREED ONLY
# ============================================
def predict_dog_breed(image):
    """Predict single dog breed from image"""
    if image is None:
        return "No image", "Please upload a dog image to get started! 📸", None

    try:
        # Preprocess image
        img = Image.fromarray(image).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        predictions = model.predict(img_array, verbose=0)

        # Get top prediction only
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100

        # Get breed name
        breed_name = breed_names[predicted_idx]
        breed_display = breed_name.replace("_", " ").title()

        # Create description
        description = get_description(breed_name)

        # Confidence bar color
        if confidence >= 70:
            conf_color = "🟢"
            conf_text = "High Confidence"
        elif confidence >= 50:
            conf_color = "🟡"
            conf_text = "Medium Confidence"
        else:
            conf_color = "🔴"
            conf_text = "Low Confidence"

        # Full output
        full_description = f"""
### {breed_display}

{conf_color} **{conf_text}: {confidence:.1f}%**

---

{description}
"""

        if confidence < 50:
            full_description += "\n\n⚠️ **Note:** Low confidence - the image might be unclear or the dog might be a mixed breed."

        return breed_display, full_description, confidence

    except Exception as e:
        return "Error", f"❌ Error processing image: {str(e)}\n\nPlease try a different image.", 0

# ============================================
# CREATE BEAUTIFUL WEB INTERFACE
# ============================================
print("🎨 Creating web interface...")

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
h1 {
    background: linear-gradient(90deg, #a020f0, #ff00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
"""

# Create interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple"), css=custom_css) as demo:

    gr.Markdown("""
    # 🐕 AI Dog Breed Identifier
    ### Upload a photo and discover the breed instantly!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📸 Upload Dog Image",
                type="numpy",
                height=400
            )
            predict_btn = gr.Button(
                "🔍 Identify Breed",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            breed_output = gr.Textbox(
                label="🏆 Predicted Breed",
                interactive=False,
                scale=1
            )
            confidence_output = gr.Slider(
                label="📊 Confidence Level",
                minimum=0,
                maximum=100,
                interactive=False,
                scale=1
            )
            description_output = gr.Markdown(
                label="📋 Breed Information"
            )

    gr.Markdown("""
    ---
    **Tips for best results:**
    - Use clear, well-lit photos
    - Show the dog's face or full body clearly
    - Avoid blurry or low-quality images

    ---
    **Created by:** Surya Kathyakeya | **Model:** MobileNetV2 | **Year:** 2025
    """)

    # Connect the button
    predict_btn.click(
        fn=predict_dog_breed,
        inputs=image_input,
        outputs=[breed_output, description_output, confidence_output]
    )

print("✅ Interface ready!\n")

# ============================================
# LAUNCH PUBLIC APP
# ============================================
print("="*70)
print("🚀 LAUNCHING YOUR AI DOG BREED IDENTIFIER!")
print("="*70)
print()
print("⏳ Creating public URL (takes 5-10 seconds)...")
print()

# Launch with public URL
demo.launch(
    share=True,
    debug=False,
    show_error=True
)

print()
print("="*70)
print("✅ YOUR APP IS LIVE!")
print("="*70)
print()
print("📱 Features:")
print("   ✓ Shows ONLY the top predicted breed")
print("   ✓ Confidence percentage with visual indicator")
print("   ✓ Breed description and information")
print("   ✓ Clean, simple interface")
print()
print("⏰ The public link expires in 72 hours")
print("💡 Re-run this cell anytime to get a new link!")
print()
print("="*70)