import panel as pn
from transformers import AutoTokenizer, TextStreamer
import transformers
import torch

pn.extension()

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    messages = [{"role": "user", "content": contents}]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = pipeline(prompt, streamer=streamer, max_new_tokens=4000, 
                       
                        do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
    message = ""
    for token in outputs[0]["generated_text"]:
        message += token
        yield message
        
# model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model, max_length=8000,
                                          
                                          )
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
)
chat_interface = pn.chat.ChatInterface(callback=callback, callback_user="Mixtral")
chat_interface.send(
    "Send a message to get a reply from Mixtral!", user="System", respond=False
)
chat_interface.servable()

