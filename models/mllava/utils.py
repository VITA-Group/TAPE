import PIL
import torch
from .modeling_llava import LlavaForConditionalGeneration
from .processing_llava import MLlavaProcessor
# from ..conversation import conv_mllava_v1_mmtag as default_conv
from ..conversation import conv_mllava_v1 as default_conv, conv_templates

from typing import List, Tuple, Union, Tuple
from io import BytesIO
from PIL import Image
import requests


def chat_mllava(
    text:str, 
    images: List[Union[PIL.Image.Image, str]], 
    model:LlavaForConditionalGeneration, 
    processor:MLlavaProcessor, 
    max_input_length:int=None, 
    history:List[dict]=None, 
    **kwargs) -> Tuple[str, List[dict]]:
    """
    Chat with the Mllava model
    Args:
        text: str, the text to be sent to the model, where <image> will be the placeholder for the image
        images: List[PIL.Image.Image], the images to be sent to the model, or None  
        model: LlavaForConditionalGeneration, the model to be used
        processor: MLlavaProcessor, the processor to be used
        max_input_length: int, the maximum input length
        history: List[dict], list of messages in the conversation as history. Each message is a dictionary {"role": "ASSISTANT/USER", "text": "the message"}. If None, the conversation will start from scratch
        kwargs: dict, the generation kwargs
    Returns:
        Tuple[str, List[dict]], the generated text and the history of the conversation
        

    """
    if "llama-3" in model.language_model.name_or_path.lower():
        conv = conv_templates['llama_3']
        terminators = [
            processor.tokenizer.eos_token_id,
            processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        conv = default_conv
        terminators = None
    kwargs["eos_token_id"] = terminators
    conv = conv.copy()
    conv.messages = []
    if history is not None:
        for message in history:
            assert message["role"] in conv.roles
            conv.append_message(message["role"], message["text"])
        if text:
            assert conv.messages[-1][0] == conv.roles[1], "The last message in the history should be the assistant, if the given text is not empty"
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], "")
            history.append({"role": conv.roles[0], "text": text})
            history.append({"role": conv.roles[1], "text": ""})
        else:
            if conv.messages[-1][0] == conv.roles[1]:
                assert conv.messages[-1][1] == "", "No user message should be provided"
            else:
                assert conv.messages[-1][0] == conv.roles[0], "The last message in the history should be the user, if the given text is empty"
                conv.append_message(conv.roles[0], "")
                history.append({"role": conv.roles[0], "text": ""})
    else:
        history = []
        history.append({"role": conv.roles[0], "text": text})
        history.append({"role": conv.roles[1], "text": ""})
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], "")
    assert conv.messages[-1][0] == conv.roles[1] and conv.messages[-1][1] == "", "Format check"
    assert history[-1]["role"] == conv.roles[1] and history[-1]["text"] == "", "Format check"
    
    prompt = conv.get_prompt()
    if images:
        for i in range(len(images)):
            if isinstance(images[i], str):
                images[i] = PIL.Image.open(images[i]).convert("RGB")
    
    inputs = processor(images=images, text=prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    for k, v in inputs.items():
        if v is not None:
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)
            elif isinstance(v, list):
                inputs[k] = [x.to(model.device) for x in v]
            else:
                raise ValueError(f"Invalid input type: {type(v)}")
    

    output_ids = model.generate(**inputs, **kwargs)
    output_ids = output_ids[0]
    
    # remove the input tokens
    generated_ids = output_ids[inputs["input_ids"].shape[-1]:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)

    history[-1]["text"] = generated_text
    
    return generated_text, history


def chat_mllava_stream(
    text:str, 
    images: List[Union[PIL.Image.Image, str]], 
    model:LlavaForConditionalGeneration, 
    processor:MLlavaProcessor, 
    max_input_length:int=None, 
    history:List[dict]=None, 
    **kwargs) -> Tuple[str, List[dict]]:
    """
    Chat with the Mllava model
    Args:
        text: str, the text to be sent to the model, where <image> will be the placeholder for the image
        images: List[PIL.Image.Image], the images to be sent to the model, or None  
        model: LlavaForConditionalGeneration, the model to be used
        processor: MLlavaProcessor, the processor to be used
        max_input_length: int, the maximum input length
        history: List[dict], list of messages in the conversation as history. Each message is a dictionary {"role": "ASSISTANT/USER", "text": "the message"}. If None, the conversation will start from scratch
        kwargs: dict, the generation kwargs
    Returns:
        Tuple[str, List[dict]], the generated text and the history of the conversation
        

    """
    if "llama-3" in model.language_model.name_or_path.lower():
        conv = conv_templates['llama_3']
        terminators = [
            processor.tokenizer.eos_token_id,
            processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        conv = default_conv
        terminators = None
    kwargs["eos_token_id"] = terminators
    conv = conv.copy()
    conv.messages = []
    if history is not None:
        for message in history:
            assert message["role"] in conv.roles
            conv.append_message(message["role"], message["text"])
        if text:
            assert conv.messages[-1][0] == conv.roles[1], "The last message in the history should be the assistant, if the given text is not empty"
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], "")
            history.append({"role": conv.roles[0], "text": text})
            history.append({"role": conv.roles[1], "text": ""})
        else:
            if conv.messages[-1][0] == conv.roles[1]:
                assert conv.messages[-1][1] == "", "No user message should be provided"
            else:
                assert conv.messages[-1][0] == conv.roles[0], "The last message in the history should be the user, if the given text is empty"
                conv.append_message(conv.roles[0], "")
                history.append({"role": conv.roles[0], "text": ""})
    else:
        history = []
        history.append({"role": conv.roles[0], "text": text})
        history.append({"role": conv.roles[1], "text": ""})
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], "")
    assert conv.messages[-1][0] == conv.roles[1] and conv.messages[-1][1] == "", "Format check"
    assert history[-1]["role"] == conv.roles[1] and history[-1]["text"] == "", "Format check"
    
    prompt = conv.get_prompt()
    if images:
        for i in range(len(images)):
            if isinstance(images[i], str):
                images[i] = PIL.Image.open(images[i])
    
    inputs = processor(images=images, text=prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    for k, v in inputs.items():
        if v is not None:
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)
            elif isinstance(v, list):
                inputs[k] = [x.to(model.device) for x in v]
            else:
                raise ValueError(f"Invalid input type: {type(v)}")
    
    from transformers import TextIteratorStreamer
    from threading import Thread
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    kwargs["streamer"] = streamer
    inputs.update(kwargs)
    thread = Thread(target=model.generate, kwargs=inputs)
    thread.start()
    for _output in streamer:
        history[-1]["text"] += _output
        yield history[-1]["text"], history


def load_image(image_file):
    if image_file.startswith("http"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        import os
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        if isinstance(image_file, Image.Image):
            image = image_file.convert("RGB")
        else:
            image = load_image(image_file)
        out.append(image)
    return out

def merge_images(image_links: List = []):
        """Merge multiple images into one image

        Args:
            image_links (List, optional): List of image links. Defaults to [].

        Returns:
            [type]: [description]
        """
        if len(image_links) == 0:
            return None
        images = load_images(image_links)
        if len(images) == 1:
            return images[0]
        widths, heights = zip(*(i.size for i in images))
        average_height = sum(heights) // len(heights)
        for i, im in enumerate(images):
            # scale in proportion
            images[i] = im.resize((int(im.size[0] * average_height / im.size[1]), average_height))
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new("RGB", (total_width + 10 * (len(images) - 1), max_height))
        x_offset = 0
        for i, im in enumerate(images):
            if i > 0:
                # past a column of 1 pixel starting from x_offset width being black, 8 pixels being white, and 1 pixel being black
                new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
                x_offset += 1
                new_im.paste(Image.new("RGB", (8, max_height), (255, 255, 255)), (x_offset, 0))
                x_offset += 8
                new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
                x_offset += 1
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        return new_im
