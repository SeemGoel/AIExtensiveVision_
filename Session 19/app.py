import torch
from utils import BigramLanguageModel, decode
import gradio as gr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BigramLanguageModel()
model.load_state_dict(torch.load("./shakespeare_gpt.pth", map_location=device))

def generate_text(max_new_tokens):
    context = torch.zeros((1, 1), dtype=torch.long)
    return decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())

# Create a Gradio interface
iface = gr.Interface(
    title = "Shakespeare Poem Generation",
    fn=generate_text,  # Function to be called on user input
    inputs=gr.Slider("Token numbers", value=1, step=1, maximum=1000),  # Input slider with a range of 0 to 1000
    outputs="text"  # Output element to display text
)

# Launch the Gradio app
iface.launch()
