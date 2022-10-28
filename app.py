from diffusers import StableDiffusionPipeline
import gradio as gr
import torch

models = [
  "nitrosocke/Arcane-Diffusion",
  "nitrosocke/archer-diffusion",
  "nitrosocke/elden-ring-diffusion",
  "nitrosocke/spider-verse-diffusion",
  "nitrosocke/modern-disney-diffusion"
]

prompt_prefixes = {
  models[0]: "arcane style ",
  models[1]: "archer style ",
  models[2]: "elden ring style ",
  models[3]: "spiderverse style ",
  models[4]: "modern disney style "
}

current_model = models[0]
pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torch.float16)
if torch.cuda.is_available():
  pipe = pipe.to("cuda")

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

def on_model_change(model):

    global current_model
    global pipe
    if model != current_model:
        current_model = model
        pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

def inference(prompt, guidance, steps):

    prompt = prompt_prefixes[current_model] + prompt
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
                  Finetuned Diffusion
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
               Demo for multiple fine-tuned Stable Diffusion models, trained on different styles: Arcane, Archer, Elden Ring, Spiderverse.
              </p>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Column():
            model = gr.Dropdown(label="Model", choices=models, value=models[0])
            prompt = gr.Textbox(label="Prompt", placeholder="{} is added automatically".format(prompt_prefixes[current_model]))
            guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
            steps = gr.Slider(label="Steps", value=50, maximum=100, minimum=2)
            run = gr.Button(value="Run")
            gr.Markdown(f"Running on: {device}")
        with gr.Column():
            image_out = gr.Image(height=512)

    model.change(on_model_change, inputs=model, outputs=[])
    run.click(inference, inputs=[prompt, guidance, steps], outputs=image_out)
    gr.Examples([
        ["jason bateman disassembling the demon core", 7.5, 50],
        ["portrait of dwayne johnson", 7.0, 75],
        ["portrait of a beautiful alyx vance half life", 10, 50],
        ["Aloy from Horizon: Zero Dawn, half body portrait, smooth, detailed armor, beautiful face, illustration", 7, 45],
        ["fantasy portrait painting, digital art", 4, 30],
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