import clip_model
from clip_model import clip_image_search
from clip_model import get_image_embeddings
from clip_model import make_train_valid_dfs
import gradio as gr
import os
import pandas as pd
import subprocess
import zipfile


image_path = "./Images"
captions_path = "."
data_source = 'flickr8k.zip'

print("\n\n")
print("Going to unzip dataset")
with zipfile.ZipFile(data_source, 'r') as zip_ref:
    zip_ref.extractall('.')
print("unzip of dataset is done")

#=============================================

cmd = "pwd"
output1 = subprocess.check_output(cmd, shell=True).decode("utf-8")
print("result of pwd command")
print(output1) 

print("Going to prepare captions.csv")
df = pd.read_csv("captions.txt")
df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
df.to_csv("captions.csv", index=False)
df = pd.read_csv("captions.csv")
print("Finished in preparing captions.csv")
print("\n\n")

print("Going to invoke make_train_valid_dfs")
_, valid_df = make_train_valid_dfs()
print("Going to invoke make_train_valid_dfs")
model, image_embeddings = get_image_embeddings(valid_df, "best.pt")

def generate_images(text, num_images=6):
    generated_images = clip_image_search(model, 
             image_embeddings,
             text,
             image_filenames=valid_df['image'].values,
             n=6)

    return generated_images

# Gradio interface
def create_demo():
      with gr.Blocks() as demo:
        text_input = gr.Textbox(label="Enter text")
        submit_button = gr.Button("Generate Images")
        image_gallery = gr.Gallery(label="Generated Images")

        def generate_and_update(text):
            if text:
                generated_images = generate_images(text)
            else:
                generated_images = []  # Handle empty input
            return generated_images

        submit_button.click(fn=generate_and_update, inputs=text_input, outputs=image_gallery)
      return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
