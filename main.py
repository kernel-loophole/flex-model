import os
import random
import torch
from huggingface_hub import login
from diffusers import DiffusionPipeline

# Your Hugging Face token
hf_token = "your_hf_token_here"  # 🔁 Replace this

# Login
login(token=hf_token)
os.environ["HUGGINGFACE_TOKEN"] = hf_token

# Model setup
model_id = "black-forest-labs/FLUX.1-dev"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

# Data
car_models = [
    "Toyota Camry sedan", "Honda Civic hatchback", "BMW X5 SUV", "Nissan Altima sedan",
    "Maserati Ghibli sedan", "Hyundai Tucson SUV", "Toyota Land Cruiser", "Camry",
    "Hyundai Elantra"
]

colors = [
    "midnight black", "pearl white", "crimson red", "navy blue", "metallic silver",
    "forest green", "sunset orange", "charcoal gray", "bronze brown", "electric yellow"
]

environments = [
    "daytime in a sunny parking lot", "rainy street at night", "urban alley under streetlights",
    "suburban driveway during the day", "desert road under a clear sky", "coastal road at sunset",
    "forested area in the morning", "city street during rush hour", "rural road at dusk",
    "industrial area under overcast skies"
]

# Image generation function
def generate_image(prompt, output_path):
    result = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5)
    image = result.images[0]
    image.save(output_path)
    print(f"✅ Saved image: {output_path}")

# Main function
def generate_and_store_images(n=10):
    output_dir = "flux_generated_images"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n):
        color = random.choice(colors)
        car = random.choice(car_models)
        env = random.choice(environments)

        prompt = (
            f"A high-resolution, photorealistic image of a {color} {car} with a visibly damaged hood, "
            f"captured in a {env}. Realistic lighting, cinematic angle, detailed textures, reflections, "
            f"35mm DSLR style."
        )
        filename = os.path.join(output_dir, f"flux_damaged_car_{i+1}.png")
        generate_image(prompt, filename)

if __name__ == "__main__":
    generate_and_store_images(n=100)
