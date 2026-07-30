from google import genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_recommendation(condition):

    prompt = f"""
You are an expert dermatologist.

The AI model predicted this skin condition:

{condition}

Explain it in simple language.

Return your response using these headings:

About this Condition

Possible Causes

Recommended Skin Care

Ingredients to Look For

Things to Avoid

When to Visit a Dermatologist

Keep the explanation beginner-friendly.
Do not mention AI.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text

# User uploads image

# ↓

# predict_skin()

# ↓

# Prediction:
# Redness

# ↓

# get_recommendation("Redness")

# ↓

# Gemini

# ↓

# Returns

# About the condition
# Morning routine
# Night routine
# Ingredients
# Products
# Diet
# Lifestyle
# Doctor advice