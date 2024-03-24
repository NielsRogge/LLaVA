import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import numpy as np

import requests
from PIL import Image
from io import BytesIO
import re


def denormalize_image(pixel_values, mean, std):
    unnormalized_image = (pixel_values.cpu().numpy() * np.array(std)[:, None, None]) + np.array(mean)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

    return unnormalized_image


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map="auto",
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # load images and prompts
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image1 = Image.open(requests.get(url, stream=True).raw)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image2 = Image.open(requests.get(url, stream=True).raw)

    images = [image1, image2]
    prompts = ["[INST] <image>\nWhat is shown in this image? [/INST]", "[INST] <image>\nHow many cats are there? [/INST]"]

    # process images
    image_sizes = [x.size for x in images]
    image_tensor = process_images(images, image_processor, model.config)
    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]

    print("Image sizes:", image_sizes)

    # denormalize the pixel values for visualization
    # mean = image_processor.image_mean
    # std = image_processor.image_std

    # for idx, image in enumerate(image_tensor):
    #     patch_images = [denormalize_image(i, mean=mean, std=std) for i in image]
    #     for patch_idx, image in enumerate(patch_images):
    #         Image.fromarray(image).save(f"patch_{idx}_{patch_idx}.png") 

    # process prompts
    convs = []
    for prompt in prompts:
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        convs.append(conv.get_prompt())

    input_tokens = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in convs]
    input_tokens_padded = torch.nn.utils.rnn.pad_sequence(input_tokens, batch_first=True, padding_value=tokenizer.pad_token_id).to(model.device)
    attention_mask = input_tokens_padded != tokenizer.pad_token_id

    with torch.inference_mode():
        generated_ids = model.generate(
                input_tokens_padded.to("cuda"),
                attention_mask=attention_mask.to("cuda"),
                images=image_tensor,
                image_sizes=image_sizes,
                # do_sample=False,
                # temperature=None,
                # top_p=None,
                # num_beams=1,
                max_new_tokens=100,
                use_cache=True,
                # output_logits=True,
                # return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
        )

    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)