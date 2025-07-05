from mlx_lm.utils import load
from mlx_lm.generate import generate
import argparse

# enum for models, using gemma 3 4 up to 27b
# note: 1b is not available as a text model and will generate slop
from enum import Enum
class GemmaModels(Enum):
    GEMMA_3_4B_IT_4BIT = "mlx-community/gemma-3-text-4b-it-4bit" # context windows 128k
    GEMMA_3_7B_IT_4BIT = "mlx-community/gemma-3-text-7b-it-4bit-DWQ" # context windows 128k
    GEMMA_3_27B_IT_4BIT = "mlx-community/gemma-3-text-27b-it-4bit-DWQ" # context windows 128k

# Choose a small, fast model for testing
model_id = GemmaModels.GEMMA_3_4B_IT_4BIT.value

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate text using MLX language model")
parser.add_argument("prompt", help="The prompt to send to the model")
args = parser.parse_args()

# Load the model and tokenizer
model, tokenizer = load(model_id)

# Format your prompt using the chat template
if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
else:
    prompt = args.prompt

response = generate(model, tokenizer, prompt=prompt)
print(response)