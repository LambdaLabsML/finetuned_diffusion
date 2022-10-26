from diffusers import StableDiffusionPipeline
import gradio as gr
import torch

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

pipe = StableDiffusionPipeline.from_pretrained("nitrosocke/Arcane-Diffusion", torch_dtype=torch.float16)
if torch.cuda.is_available():
  pipe = pipe.to("cuda")

def inference(prompt, guidance, steps):
    prompt = prompt + " , arcane style"
    image = pipe(prompt, num_inference_steps=int(steps), guidance_scale=guidance, width=512, height=512).images[0]
    return image

with gr.Blocks() as demo:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 700px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Arcane Diffusion
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
               Demo for a fine-tuned Stable Diffusion model trained on images from the TV Show Arcane.
              </p>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Column():
            prompt = gr.Textbox(label="prompt", placeholder="' , arcane style' is appended automatically")
            guidance = gr.Slider(label="guidance scale", value=7.5, maximum=15)
            steps = gr.Slider(label="steps", value=50, maximum=100, minimum=2)
            run = gr.Button(value="Run")
            gr.Markdown(f"Running on: {device}")
        with gr.Column():
            image_out = gr.Image(height=512)

    run.click(inference, inputs=[prompt, guidance, steps], outputs=image_out)
    gr.Examples([
        ["jason bateman disassembling the demon core, arcane style", 7.5, 50],
        ["portrait of dwayne johnson, arcane style", 7.0, 75],
        ["portrait of a beautiful alyx vance half life, volume lighting, concept art, by greg rutkowski!!, colorful, xray melting colors!!, arcane style", 7, 50],
        ["Aloy from Horizon: Zero Dawn, half body portrait, videogame cover art, highly detailed, digital painting, artstation, concept art, smooth, detailed armor, sharp focus, beautiful face, illustration, art by Artgerm and greg rutkowski and alphonse mucha, arcane style", 7, 50],
        ["fantasy portrait painting, digital art, arcane style", 4, 30],
    ], [prompt, guidance, steps], image_out, inference, cache_examples=torch.cuda.is_available())
    gr.HTML('''
        <div>
            <p>Model by <a href="https://huggingface.co/nitrosocke" style="text-decoration: underline;" target="_blank">@nitrosocke</a> ❤️</p>
        </div>
        <div>Space by 
            <a href="https://twitter.com/hahahahohohe">
              <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/hahahahohohe?label=%40anzorq&style=social">
            </a>
        </div>
        ''')

demo.queue()
demo.launch()