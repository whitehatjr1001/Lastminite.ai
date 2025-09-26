import os
from dotenv import load_dotenv

# Make sure this import path is correct for your project structure
from src.lastminute_api.infrastructure.nano_bannana.openai import create_openai_client

# --- Configuration ---
PROMPT = "create a image of muscle fiber with labels"  
OUTPUT_FILENAME = "output_image.png"
# -------------------

def generate_and_save_image():
    """
    Uses your OpenAI client to generate an image and save it to a file.
    """
    print("--- Running Standalone Image Generation Test ---")

    # Load environment variables from the .env file
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("🔴 ERROR: OPENAI_API_KEY not found. Check your .env file.")
        return

    print("✅ Environment key loaded. Initializing client...")

    try:
        # Initialize the client, forcing it to use the environment variable
        client = create_openai_client({})

        image_model = os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")
        max_images = int(os.getenv("OPENAI_IMAGE_MAX_N", "1"))

        print(f"✅ Client initialized. Sending prompt to OpenAI:\n   '{PROMPT}'")
        print("   (This may take up to 30 seconds...)")

        # Call the generate function, specifying you want an image
        result = client.generate(
            prompt=PROMPT,
            response_modalities=["image"],
            model=image_model,
            max_images=max_images,
        )

        # Check if the result contains any images
        if not result.images:
            print("🔴 ERROR: OpenAI did not return an image. The response was empty.")
            return

        # Get the binary data of the first image
        image_data = result.images[0].data

        # Save the image data to a file
        with open(OUTPUT_FILENAME, "wb") as f:
            f.write(image_data)

        print(f"\n🎉 SUCCESS! Image saved as '{OUTPUT_FILENAME}' in your project directory.")

    except Exception as e:
        print(f"\n🔴 An unexpected error occurred: {e}")
        print("   Please check your API key, OpenAI account status, and internet connection.")

if __name__ == "__main__":
    generate_and_save_image()
