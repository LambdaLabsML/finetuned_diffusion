from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import gradio as gr
import torch

models = [
  "nitrosocke/Arcane-Diffusion",
  "nitrosocke/archer-diffusion",
  "nitrosocke/elden-ring-diffusion",
  "nitrosocke/spider-verse-diffusion",
  "nitrosocke/modern-disney-diffusion",
  "hakurei/waifu-diffusion",
  "lambdalabs/sd-pokemon-diffusers",
  "yuk/fuyuko-waifu-diffusion",
  "AstraliteHeart/pony-diffusion",
  "IfanSnek/JohnDiffusion",
  "nousr/robo-diffusion"
]

prompt_prefixes = {
  models[0]: "arcane style ",
  models[1]: "archer style ",
  models[2]: "elden ring style ",
  models[3]: "spiderverse style ",
  models[4]: "modern disney style ",
  models[5]: "",
  models[6]: "",
  models[7]: "",
  models[8]: "",
  models[9]: "",
  models[10]: "",
}

current_model = models[0]
pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torch.float16)
if torch.cuda.is_available():
  pipe = pipe.to("cuda")

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

def inference(model, prompt, img, guidance, steps):
  if img is not None:
    return img_inference(model, prompt, img, guidance, steps)
  else:
    return text_inference(model, prompt, guidance, steps)

def text_inference(model, prompt, guidance, steps):

    global current_model
    global pipe
    if model != current_model:
        current_model = model
        pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torch.float16)
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

    prompt = prompt_prefixes[current_model] + prompt
    image = pipe(prompt, num_inference_steps=int(steps), guidance_scale=guidance, width=512, height=512).images[0]
    return image

def img_inference(model, prompt, img, guidance, steps):

    global current_model
    global pipe
    if model != current_model:
        current_model = model
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(current_model, torch_dtype=torch.float16)
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

    prompt = prompt_prefixes[current_model] + prompt
    img.resize((512, 512))
    image = pipe(
        prompt,
        init_image=img,
        num_inference_steps=int(steps),
        strength=0.75,
        guidance_scale=guidance,
        width=512,
        height=512).images[0]
    return image


css = """
  <style>
  .finetuned-diffusion-div {
      text-align: center;
      max-width: 700px;
      margin: 0 auto;
    }
    .finetuned-diffusion-div div {
      display: inline-flex;
      align-items: center;
      gap: 0.8rem;
      font-size: 1.75rem;
    }
    .finetuned-diffusion-div div h1 {
      font-weight: 900;
      margin-bottom: 7px;
    }
    .finetuned-diffusion-div p {
      margin-bottom: 10px;
      font-size: 94%;
    }
    .finetuned-diffusion-div p a {
      text-decoration: underline;
    }
  </style>
"""
with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Finetuned Diffusion</h1>
              </div>
              <p>
               Demo for multiple fine-tuned Stable Diffusion models, trained on different styles: <br>
               <a href="https://huggingface.co/nitrosocke/Arcane-Diffusion">Arcane</a>, <a href="https://huggingface.co/nitrosocke/archer-diffusion">Archer</a>, <a href="https://huggingface.co/nitrosocke/elden-ring-diffusion">Elden Ring</a>, <a href="https://huggingface.co/nitrosocke/spider-verse-diffusion">Spiderverse</a>, <a href="https://huggingface.co/nitrosocke/modern-disney-diffusion">Modern Disney</a>, <a href="https://huggingface.co/hakurei/waifu-diffusion">Waifu</a>, <a href="https://huggingface.co/lambdalabs/sd-pokemon-diffusers">Pokemon</a>, <a href="https://huggingface.co/yuk/fuyuko-waifu-diffusion">Fuyuko Waifu</a>, <a href="https://huggingface.co/AstraliteHeart/pony-diffusion">Pony</a>, <a href="https://huggingface.co/IfanSnek/JohnDiffusion">John</a>, <a href="https://huggingface.co/nousr/robo-diffusion">Robo</a>.
              </p>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Column():
            model = gr.Dropdown(label="Model", choices=models, value=models[0])
            prompt = gr.Textbox(label="Prompt", placeholder="Style prefix is applied automatically")
            img = gr.Image(label="img2img (optional)", type="pil", height=256, tool="editor")
            guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
            steps = gr.Slider(label="Steps", value=50, maximum=100, minimum=2)
            run = gr.Button(value="Run")
            gr.Markdown(f"Running on: {device}")
        with gr.Column():
            image_out = gr.Image(height=512)

    run.click(inference, inputs=[model, prompt, img, guidance, steps], outputs=image_out)
    gr.Examples([
        [models[0], "jason bateman disassembling the demon core", 7.5, 50],
        [models[3], "portrait of dwayne johnson", 7.0, 75],
        [models[4], "portrait of a beautiful alyx vance half life", 10, 50],
        [models[5], "Aloy from Horizon: Zero Dawn, half body portrait, smooth, detailed armor, beautiful face, illustration", 7, 45],
        [models[4], "fantasy portrait painting, digital art", 4, 30],
    ], [model, prompt, guidance, steps], image_out, text_inference, cache_examples=torch.cuda.is_available())
    gr.HTML('''
        <div>
            <p>Model by <a href="https://huggingface.co/nitrosocke" target="_blank">@nitrosocke</a> ❤️</p>
        </div>
        <div>Space by 
            <a href="https://twitter.com/hahahahohohe">
              <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/hahahahohohe?label=%40anzorq&style=social">
            </a>
        </div>
        ''')

demo.queue()
demo.launch(debug=True)