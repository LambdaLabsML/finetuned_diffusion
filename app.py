from diffusers import StableDiffusionPipeline
import gradio as gr
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained("nitrosocke/Arcane-Diffusion", torch_dtype=torch.float16)
if torch.cuda.is_available():
  pipe = pipe.to("cuda")

def inference(prompt, guidance, steps):    
    all_images = []
    images = pipe([prompt] * 1, num_inference_steps=int(steps), guidance_scale=guidance, width=512, height=512).images
    all_images.extend(images)
    return all_images

with gr.Blocks() as demo:
    with gr.Row():
        
        with gr.Column():
            prompt = gr.Textbox(label="prompt")
            guidance = gr.Slider(label="guidance scale", value=7.5, maximum=15)
            steps = gr.Slider(label="steps", value=50, maximum=100, minimum=2)
            run = gr.Button(value="Run")
            gr.Markdown(f"Running on: {device}")
        with gr.Column():
            gallery = gr.Gallery()

    run.click(inference, inputs=[prompt, guidance, steps], outputs=gallery)
    gr.Examples([
        ["jason bateman disassembling the demon core, arcane style", 7.5, 50],
        ["portrait of dwayne johnson, arcane style", 7.0, 75],
        ["portrait of a beautiful alyx vance half life, volume lighting, concept art, by greg rutkowski!!, colorful, xray melting colors!!, arcane style", 7, 50],
        ["Aloy from Horizon: Zero Dawn, half body portrait, videogame cover art, highly detailed, digital painting, artstation, concept art, smooth, detailed armor, sharp focus, beautiful face, illustration, art by Artgerm and greg rutkowski and alphonse mucha, arcane style", 7, 50],
        ["fantasy portrait painting, digital art, arcane style", 4, 30],
    ], [prompt, guidance, steps], gallery, inference, cache_examples=torch.cuda.is_available())

demo.queue()
demo.launch()