from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import gradio as gr
import torch
from PIL import Image
import utils

is_colab = utils.is_google_colab()

class Model:
    def __init__(self, name, path, prefix):
        self.name = name
        self.path = path
        self.prefix = prefix
        self.pipe_t2i = None
        self.pipe_i2i = None

models = [
     Model("Custom model", "", ""),
     Model("Arcane", "nitrosocke/Arcane-Diffusion", "arcane style "),
     Model("Archer", "nitrosocke/archer-diffusion", "archer style "),
     Model("Elden Ring", "nitrosocke/elden-ring-diffusion", "elden ring style "),
     Model("Spider-Verse", "nitrosocke/spider-verse-diffusion", "spiderverse style "),
     Model("Modern Disney", "nitrosocke/modern-disney-diffusion", "modern disney style "),
     Model("Classic Disney", "nitrosocke/classic-anim-diffusion", ""),
     Model("Waifu", "hakurei/waifu-diffusion", ""),
     Model("Pokémon", "lambdalabs/sd-pokemon-diffusers", ""),
     Model("Pony Diffusion", "AstraliteHeart/pony-diffusion", ""),
     Model("Robo Diffusion", "nousr/robo-diffusion", ""),
     Model("Cyberpunk Anime", "DGSpitzer/Cyberpunk-Anime-Diffusion", "dgs illustration style "),
     Model("Tron Legacy", "dallinmackay/Tron-Legacy-diffusion", "trnlgcy")
]

last_mode = "txt2img"
current_model = models[1]
current_model_path = current_model.path
pipe = None

if is_colab:
  pipe = StableDiffusionPipeline.from_pretrained(current_model.path, torch_dtype=torch.float16)

else: # download all models
  vae = AutoencoderKL.from_pretrained(current_model.path, subfolder="vae", torch_dtype=torch.float16)
  for model in models[1:]:
    unet = UNet2DConditionModel.from_pretrained(model.path, subfolder="unet", torch_dtype=torch.float16)
    model.pipe_t2i = StableDiffusionPipeline.from_pretrained(model.path, unet=unet, vae=vae, torch_dtype=torch.float16)
    model.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(model.path, unet=unet, vae=vae, torch_dtype=torch.float16)
  pipe = models[1].pipe_t2i
  
if torch.cuda.is_available():
  pipe = pipe.to("cuda")

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

def custom_model_changed(path):
  models[0].path = path
  global current_model
  current_model = models[0]

def inference(model_name, prompt, guidance, steps, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt=""):

  global current_model
  for model in models:
    if model.name == model_name:
      current_model = model
      model_path = current_model.path

  generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None

  if img is not None:
    return img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator)
  else:
    return txt_to_img(model_path, prompt, neg_prompt, guidance, steps, width, height, generator)

def txt_to_img(model_path, prompt, neg_prompt, guidance, steps, width, height, generator=None):

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "txt2img":
        current_model_path = model_path

        if is_colab:
          pipe = StableDiffusionPipeline.from_pretrained(current_model_path, torch_dtype=torch.float16)
        else:
          pipe = pipe.to("cpu")
          pipe = current_model.pipe_t2i

        if torch.cuda.is_available():
          pipe = pipe.to("cuda")
        last_mode = "txt2img"

    prompt = current_model.prefix + prompt
    result = pipe(
      prompt,
      negative_prompt = neg_prompt,
      # num_images_per_prompt=n_images,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator)
    
    return replace_nsfw_images(result)

def img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator=None):

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "img2img":
        current_model_path = model_path

        if is_colab:
          pipe = StableDiffusionImg2ImgPipeline.from_pretrained(current_model_path, torch_dtype=torch.float16)
        else:
          pipe = pipe.to("cpu")
          pipe = current_model.pipe_t2i
        
        if torch.cuda.is_available():
              pipe = pipe.to("cuda")
        last_mode = "img2img"

    prompt = current_model.prefix + prompt
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe(
        prompt,
        negative_prompt = neg_prompt,
        # num_images_per_prompt=n_images,
        init_image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        width = width,
        height = height,
        generator = generator)
        
    return replace_nsfw_images(result)

def replace_nsfw_images(results):
    for i in range(len(results.images)):
      if results.nsfw_content_detected[i]:
        results.images[i] = Image.open("nsfw.png")
    return results.images[0]

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
    .tabs {
      margin-top: 0px;
      margin-bottom: 0px;
    }
    #gallery {
      min-height: 20rem;
    }
  </style>
"""
with gr.Blocks(css=css) as demo:
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Finetuned Diffusion</h1>
              </div>
              <p>
               Demo for multiple fine-tuned Stable Diffusion models, trained on different styles: <br>
               <a href="https://huggingface.co/nitrosocke/Arcane-Diffusion">Arcane</a>, <a href="https://huggingface.co/nitrosocke/archer-diffusion">Archer</a>, <a href="https://huggingface.co/nitrosocke/elden-ring-diffusion">Elden Ring</a>, <a href="https://huggingface.co/nitrosocke/spider-verse-diffusion">Spiderverse</a>, <a href="https://huggingface.co/nitrosocke/modern-disney-diffusion">Modern Disney</a>, <a href="https://huggingface.co/hakurei/waifu-diffusion">Waifu</a>, <a href="https://huggingface.co/lambdalabs/sd-pokemon-diffusers">Pokemon</a>, <a href="https://huggingface.co/yuk/fuyuko-waifu-diffusion">Fuyuko Waifu</a>, <a href="https://huggingface.co/AstraliteHeart/pony-diffusion">Pony</a>, <a href="https://huggingface.co/sd-dreambooth-library/herge-style">Hergé (Tintin)</a>, <a href="https://huggingface.co/nousr/robo-diffusion">Robo</a>, <a href="https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion">Cyberpunk Anime</a> + any other custom Diffusers 🧨 SD model hosted on HuggingFace 🤗.
              </p>
              <p>Don't want to wait in queue? <a href="https://colab.research.google.com/gist/qunash/42112fb104509c24fd3aa6d1c11dd6e0/copy-of-fine-tuned-diffusion-gradio.ipynb"><img data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a></p>
               Running on <b>{device}</b>
              </p>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Group():
            model_name = gr.Dropdown(label="Model", choices=[m.name for m in models], value=current_model.name)
            custom_model_path = gr.Textbox(label="Custom model path", placeholder="Path to model, e.g. nitrosocke/Arcane-Diffusion", visible=False, interactive=True)
            
            with gr.Row():
              prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,placeholder="Enter prompt. Style applied automatically").style(container=False)
              generate = gr.Button(value="Generate").style(rounded=(False, True, True, False))


            image_out = gr.Image(height=512)
            # gallery = gr.Gallery(
            #     label="Generated images", show_label=False, elem_id="gallery"
            # ).style(grid=[1], height="auto")

        with gr.Tab("Options"):
          with gr.Group():
            neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")

            # n_images = gr.Slider(label="Images", value=1, minimum=1, maximum=4, step=1)

            with gr.Row():
              guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
              steps = gr.Slider(label="Steps", value=50, minimum=2, maximum=100, step=1)

            with gr.Row():
              width = gr.Slider(label="Width", value=512, minimum=64, maximum=1024, step=8)
              height = gr.Slider(label="Height", value=512, minimum=64, maximum=1024, step=8)

            seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)

        with gr.Tab("Image to image"):
            with gr.Group():
              image = gr.Image(label="Image", height=256, tool="editor", type="pil")
              strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)

    model_name.change(lambda x: gr.update(visible = x == models[0].name), inputs=model_name, outputs=custom_model_path)
    custom_model_path.change(custom_model_changed, inputs=custom_model_path, outputs=None)
    # n_images.change(lambda n: gr.Gallery().style(grid=[2 if n > 1 else 1], height="auto"), inputs=n_images, outputs=gallery)

    inputs = [model_name, prompt, guidance, steps, width, height, seed, image, strength, neg_prompt]
    prompt.submit(inference, inputs=inputs, outputs=image_out)
    generate.click(inference, inputs=inputs, outputs=image_out)

    ex = gr.Examples([
        [models[1].name, "jason bateman disassembling the demon core", 7.5, 50],
        [models[4].name, "portrait of dwayne johnson", 7.0, 75],
        [models[5].name, "portrait of a beautiful alyx vance half life", 10, 50],
        [models[6].name, "Aloy from Horizon: Zero Dawn, half body portrait, smooth, detailed armor, beautiful face, illustration", 7.0, 45],
        [models[5].name, "fantasy portrait painting, digital art", 4.0, 30],
    ], [model_name, prompt, guidance, steps, seed], image_out, inference, cache_examples=not is_colab and torch.cuda.is_available())
    # ex.dataset.headers = [""]

    gr.Markdown('''
      Models by [@nitrosocke](https://huggingface.co/nitrosocke), [@Helixngc7293](https://twitter.com/DGSpitzer) and others. ❤️<br>
      Space by: [![Twitter Follow](https://img.shields.io/twitter/follow/hahahahohohe?label=%40anzorq&style=social)](https://twitter.com/hahahahohohe)
  
      ![visitors](https://visitor-badge.glitch.me/badge?page_id=anzorq.finetuned_diffusion)
    ''')

if not is_colab:
  demo.queue(concurrency_count=4)
demo.launch(debug=is_colab, share=is_colab)