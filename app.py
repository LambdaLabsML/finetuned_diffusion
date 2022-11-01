from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
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
  "nousr/robo-diffusion",
  "DGSpitzer/Cyberpunk-Anime-Diffusion",
  "sd-dreambooth-library/herge-style"
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
  models[10]: "dgs illustration style ",
  models[11]: "herge_style ",
}

current_model = models[0]
pipes = []
vae = AutoencoderKL.from_pretrained(current_model, subfolder="vae", torch_dtype=torch.float16)
for model in models:
  unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet", torch_dtype=torch.float16)
  pipe = StableDiffusionPipeline.from_pretrained(model, unet=unet, vae=vae, torch_dtype=torch.float16)
  pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(model, unet=unet, vae=vae, torch_dtype=torch.float16)
  pipes.append({"name":model, "pipeline":pipe, "pipeline_i2i":pipe_i2i})


device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

def inference(model, img, strength, prompt, neg_prompt, guidance, steps, width, height, seed):

  generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None
  
  if img is not None:
    return img_to_img(model, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator)
  else:
    return txt_to_img(model, prompt, neg_prompt, guidance, steps, width, height, generator)

def txt_to_img(model, prompt, neg_prompt, guidance, steps, width, height, generator=None):

    global current_model
    global pipe
    if model != current_model:
        current_model = model 
        pipe = pipe.to("cpu")
        
    for pipe_dict in pipes:
          if(pipe_dict["name"] == current_model):
            pipe = pipe_dict["pipeline"]
          
          if torch.cuda.is_available():
            pipe = pipe.to("cuda")

    prompt = prompt_prefixes[current_model] + prompt
    results = pipe(
      prompt,
      negative_prompt=neg_prompt,
      num_inference_steps=int(steps),
      guidance_scale=guidance,
      width=width,
      height=height,
      generator=generator)
    
    image = results.images[0] if not results.nsfw_content_detected[0] else Image.open("nsfw.png")
    return image

def img_to_img(model, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator):

    global current_model
    global pipe
    if model != current_model:
        current_model = model
        pipe = pipe.to("cpu")
        
    for pipe_dict in pipes:
          if(pipe_dict["name"] == current_model):
            pipe = pipe_dict["pipeline_i2i"]
          
          if torch.cuda.is_available():
            pipe = pipe.to("cuda")

    prompt = prompt_prefixes[current_model] + prompt
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    results = pipe(
        prompt,
        negative_prompt=neg_prompt,
        init_image=img,
        num_inference_steps=int(steps),
        strength=strength,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator)
        
    image = results.images[0] if not results.nsfw_content_detected[0] else Image.open("nsfw.png")
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
               <a href="https://huggingface.co/nitrosocke/Arcane-Diffusion">Arcane</a>, <a href="https://huggingface.co/nitrosocke/archer-diffusion">Archer</a>, <a href="https://huggingface.co/nitrosocke/elden-ring-diffusion">Elden Ring</a>, <a href="https://huggingface.co/nitrosocke/spider-verse-diffusion">Spiderverse</a>, <a href="https://huggingface.co/nitrosocke/modern-disney-diffusion">Modern Disney</a>, <a href="https://huggingface.co/hakurei/waifu-diffusion">Waifu</a>, <a href="https://huggingface.co/lambdalabs/sd-pokemon-diffusers">Pokemon</a>, <a href="https://huggingface.co/yuk/fuyuko-waifu-diffusion">Fuyuko Waifu</a>, <a href="https://huggingface.co/AstraliteHeart/pony-diffusion">Pony</a>, <a href="https://huggingface.co/sd-dreambooth-library/herge-style">Hergé (Tintin)</a>, <a href="https://huggingface.co/nousr/robo-diffusion">Robo</a>, <a href="https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion">Cyberpunk Anime</a>
              </p>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Column():
            model = gr.Dropdown(label="Model", choices=models, value=models[0])
            prompt = gr.Textbox(label="Prompt", placeholder="Style prefix is applied automatically")
            run = gr.Button(value="Run")
            gr.Markdown(f"Running on: {device}")
            
            with gr.Tab("Options"):

                neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")
                guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
                steps = gr.Slider(label="Steps", value=50, maximum=100, minimum=2)
                width = gr.Slider(label="Width", value=512, maximum=1024, minimum=64)
                height = gr.Slider(label="Height", value=512, maximum=1024, minimum=64)
                seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)
            with gr.Tab("Image to image"):
                image = gr.Image(label="Image", height=256, tool="editor", type="pil")
                strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)

        with gr.Column():
            image_out = gr.Image(height=512)

    inputs = [model, image, strength, prompt, neg_prompt, guidance, steps, width, height, seed]
    prompt.submit(inference, inputs=inputs, outputs=image_out)
    run.click(inference, inputs=inputs, outputs=image_out)
    gr.Examples([
        [models[0], "jason bateman disassembling the demon core", 7.5, 50],
        [models[3], "portrait of dwayne johnson", 7.0, 75],
        [models[4], "portrait of a beautiful alyx vance half life", 10, 50],
        [models[5], "Aloy from Horizon: Zero Dawn, half body portrait, smooth, detailed armor, beautiful face, illustration", 7, 45],
        [models[4], "fantasy portrait painting, digital art", 4, 30],
    ], [model, prompt, guidance, steps], image_out, img_to_img, cache_examples=False)
    gr.Markdown('''
      Models by [@nitrosocke](https://huggingface.co/nitrosocke), [@Helixngc7293](https://twitter.com/DGSpitzer) and others. ❤️<br>
      Space by: [![Twitter Follow](https://img.shields.io/twitter/follow/hahahahohohe?label=%40anzorq&style=social)](https://twitter.com/hahahahohohe)
  
      ![visitors](https://visitor-badge.glitch.me/badge?page_id=anzorq.finetuned_diffusion)
    ''')

demo.queue()
demo.launch()