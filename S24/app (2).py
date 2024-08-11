import gradio as gr
import torch
# region_offset = torch.tensor(region_offset).int()

from utils import gen_image_as_per_prompt

styles = ["depthmap", "cosmicgalaxy", "concept-art", "Marc Allante", "midjourney-style", "No style"]
styleValues = ["learned_embeds_depthmap.bin",
               "learned_embeds_cosmic-galaxy-characters-style.bin",
               "learned_embeds_sd_concept-art.bin",
               "learned_embeds_style-of-marc-allante.bin",
               "learned_embeds_midjourney.bin",
               ""]
seed_values = [30, 24, 35, 47, 78, 42]

styles_dict = dict(zip(styles, styleValues))
seed_dict = dict(zip(styles, seed_values))


# Custom loss function
def reduce_highlight(images):
    """Calculates the mean absolute error for amber color.

    Args:
      images: A tensor of shape (batch_size, channels, height, width).
      target_red: Target red value for amber.
      target_green: Target green value for amber.
      target_blue: Target blue value for amber.

    Returns:
      The mean absolute error.
      #target_red=0.8, target_green=0.6, target_blue=0.4
    """

    red_error = torch.abs(images[:, 0] - 0.12).mean()
    green_error = torch.abs(images[:, 1] - 0.2).mean()
    blue_error = torch.abs(images[:, 2] - 0.15).mean()

    # You can adjust weights for each channel if needed
    amber_error = (red_error + green_error + blue_error) / 3
    return amber_error

def output(text, style, use_loss=False):
    if use_loss:
        image = gen_image_as_per_prompt(text, styles_dict[style], seed_dict[style], reduce_highlight)
    else:
        image = gen_image_as_per_prompt(text, styles_dict[style], seed_dict[style])
    return image


title = "Stable Diffusion with different styles"
description = "Explore the versatility of artistic styles by transforming your prompts. This demo takes the promt and applies a unique style of your choice, giving you a fresh visual interpretation."
examples = [["A majestic lion with the playful expression of a puppy", "depthmap", True],
            ["A futuristic robot designed in space suit", "midjourney", True],
            ["A serene forest scene, with animals that resemble puppies", "cosmicgalaxy", False],
             ["A warrior in an ancient battlefield, with a hint of puppy charm", "concept-art", False]]

demo = gr.Interface(
    output,
    inputs=[
        gr.Textbox(placeholder="Prompt", container=False, scale=7),
        gr.Radio(styles, label="Select a Style"),
        gr.Checkbox(label="Use custom loss")
    ],
    outputs=[
        gr.Image(width=512, height=512, label="output")
    ],
    title=title,
    description=description,
    examples=examples,
    cache_examples=False
)
demo.launch(debug=True)