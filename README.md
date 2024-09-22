# Stable Diffusion Image Generation Project

This project demonstrates how to use the Stable Diffusion model for image generation tasks. It includes setup instructions, usage examples, and troubleshooting tips for common issues.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/stable-diffusion-project.git
   cd stable-diffusion-project
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install torch diffusers transformers
   ```

4. Set up your Hugging Face account and obtain an access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Usage

Here's a basic example of how to generate an image using the Stable Diffusion model:

```python
from diffusers import StableDiffusionPipeline
import torch

# Initialize the pipeline
model_id = "stabilityai/stable-diffusion-2-1"
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token='YOUR_HUGGING_FACE_TOKEN'
)
pipeline = pipeline.to("cuda")

# Generate an image
prompt = "A beautiful sunset over a calm ocean"
image = pipeline(prompt).images[0]

# Save the image
image.save("generated_image.png")
```

Replace `'YOUR_HUGGING_FACE_TOKEN'` with your actual Hugging Face token.

## Troubleshooting

### Missing Files Error

If you encounter an error about missing files, try clearing the cache and re-downloading the model:

```python
import shutil
import os

# Clear the cache
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Cleared cache directory: {cache_dir}")
else:
    print(f"Cache directory not found: {cache_dir}")

# Re-run your model initialization code
```

### Module Not Found Error

If you get a "No module named 'diffusers'" error, make sure you've installed the required packages:

```
pip install diffusers transformers
```

### CUDA Out of Memory Error

If you're running out of CUDA memory, try using a smaller model or reducing the image size:

```python
pipeline = pipeline.to("cuda")
image = pipeline(prompt, height=512, width=512).images[0]  # Adjust size as needed
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
