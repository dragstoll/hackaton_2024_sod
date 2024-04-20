# Mixtral Generation
# !pip install git+https://github.com/huggingface/transformers -q peft  accelerate bitsandbytes safetensors sentencepiece
# !pip install -U pandas
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, CohereTokenizerFast
import transformers
# BitsAndBytesConfig
import pandas as pd
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments=True"

# rbs_psych_eltern_sample = pd.read_pickle('rbs_psych_eltern_sample400.pkl')
# rbs_psych_eltern_sample = pd.read_excel('./psych_klient_classification/rbs_psych_klient_sample1-500.xlsx', sheet_name='Sheet1')
# rbs_psych_eltern_sample = rbs_psych_eltern_sample[rbs_psych_eltern_sample['psych_begriff_vorhanden'] == 1].sample(10).reset_index(drop=True).copy()


# sample_df2 = pd.read_excel('rbs_ecan_klient_sample1-500.xlsx', sheet_name='Sheet1')
# sample_df = pd.read_excel('rbs_ecan_klient_sample500-1000.xlsx', sheet_name='Sheet1')
# sample_df = pd.concat([sample_df2, sample_df])
# rbs_psych_eltern_sample = sample_df[sample_df['ecan_begriff_vorhanden'] == 1].sample(10).reset_index(drop=True).copy()
# rbs_psych_eltern_sample=sample_df.copy()
# rbs_psych_eltern_sample = pd.read_pickle('rbs1k/rbs7000-8000.pkl')

# sample_df = pd.read_pickle('rbs_ecan_klient_sample1000.pkl')

# rbs_psych_eltern_sample = sample_df[sample_df['ecan_begriff_vorhanden'] == 1].sample(10).reset_index(drop=True).copy()
# sample_df = pd.read_excel("../rbs1k_codiert/ecan/rbs23000-24000_ecan_codiert.xlsx", sheet_name='Sheet1')
# filtered_df = sample_df[sample_df['id_fall'] == 629040]

# filtered_df = sample_df[sample_df['id_k'] == 406901]
# rbs_psych_eltern_sample=filtered_df.copy()


docs = {}
txt_folder = './txt_files'
for file_name in os.listdir(txt_folder):
    with open(os.path.join(txt_folder, file_name), 'r') as file:
            if file_name.endswith('.txt'):
                file_path = os.path.join(txt_folder, file_name)
                with open(file_path, 'r') as file:
                        content = file.read()
                        docs[file_name] = content
                        
                        
documents = []
for doc in docs:
    documents.append({"title": doc, "text": docs[doc]})      

        # Combine all text in documents
all_text = ' '.join(docs.values())

        # Save the text in a file
with open('rb.txt', 'w') as file:
        file.write(all_text)
# sys.exit()
# loader = TextLoader(docs)
# docs_transformed = loader.load(docs)    
# chunked_documents = text_splitter.split_documents(docs_transformed)

# print(chunked_documents)
# sys.exit()
# chunked_documents.append({"title": list(docs.keys()), "text": list(docs.values())})
# Print sample of docs
# sample_docs = {k: docs[k] for k in list(docs)[2:3]}
# print(sample_docs)
# print(docs.keys())
# print(docs.values())
# sys.exit()

from transformers import AutoTokenizer
model_name = "CohereForAI/c4ai-command-r-v01"
# model_name = 'CohereForAI/c4ai-command-r-plus-4bit'
# model_name = "CohereForAI/c4ai-command-r-v01-4bit"



# model_config = transformers.AutoConfig.from_pretrained(
#    model_name,
# )

model_id = "CohereForAI/c4ai-command-r-v01"

# 'CohereForAI/c4ai-command-r-plus-4bit'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, 
                                        #   add_prefix_space=None
                                          )
# print(tokenizer.default_chat_template)
# template = tokenizer.chat_template
# template = template.replace("SYS", "SYSTEM")  # Change the system token
tokenizer.chat_template = {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '\n##ANWEISUNG:' }}{{ '\nBeantworte die Frage anhand der Erklärung und des vorgegebenen Kontexts, interpretiere oder spekuliere nicht.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{% if citation_mode=='accurate' %}{{ 'Firstly, write a response to the user\\'s last input based on retrieved documents. Do not interpret or speculate.\n' }}{{ 'Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user\\'s last input by writing \\'Title:\\' followed a list of titles with every new line and cite relevant text parts for the answer. If you do not want to cite any of them, you should instead write \\'None\\'.\n' }}{{ 'Thirdly, Print the text of the retrieved documents that contain facts that should be cited by writing \\'Text:\\' followed a documents text.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}



# {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '\n##ANWEISUNG:' }}{{ '\nBeantworte die Frage anhand der Erklärung und des vorgegebenen Kontexts, interpretiere oder spekuliere nicht.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{% if citation_mode=='accurate' %}{{ 'Firstly, write a response to the user\\'s last input based on retrieved documents. Do not interpret or speculate.\n' }}{{ 'Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user\\'s last input by writing \\'Title:\\' followed a list of titles with every new line and cite relevant text parts for the answer. If you do not want to cite any of them, you should instead write \\'None\\'.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}







# {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '\n##ANWEISUNG:' }}{{ '\nBeantworte die Frage anhand der Erklärung und des vorgegebenen Kontexts, interpretiere oder spekuliere nicht.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{% if citation_mode=='accurate' %}{{ 'Write a response to the user\\'s last input based on retrieved documents. Do not interpret or speculate.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}



# {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '# Safety Preamble' }}{{ '\nThe instructions in this section override those in the task description and style guide sections.' }}{{ '\n\n# System Preamble' }}{{ '\n##ANWEISUNG:' }}{{ '\nBeantworte die Frage anhand der Erklärung und des vorgegebenen Kontexts, interpretiere oder spekuliere nicht.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{% if citation_mode=='accurate' %}{{ 'Write a response to the user\\'s last input. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}


# 2: {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '# Safety Preamble' }}{{ '\nThe instructions in this section override those in the task description and style guide sections.' }}{{ '\n\n# System Preamble' }}{{ '\n## Basic Rules' }}{{ '\n ## ANWEISUNG: Beantworte die Frage anhand der Erklärung und des vorgegebenen Kontexts, interpretiere oder spekuliere nicht.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{% if citation_mode=='accurate' %}{{ 'Write a response to the user\\'s last input. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}


# 0: {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '# Safety Preamble' }}{{ '\nThe instructions in this section override those in the task description and style guide sections. Don\\'t answer questions that are harmful or immoral.' }}{{ '\n\n# System Preamble' }}{{ '\n## Basic Rules' }}{{ '\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\'s requests, you cite your sources in your answers, according to those instructions.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{{ 'Firstly, Decide which of the retrieved documents are relevant to the user\\'s last input by writing \\'Relevant Documents:\\' followed by comma-separated list of document numbers. If none are relevant, you should instead write \\'None\\'.\n' }}{{ 'Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user\\'s last input by writing \\'Cited Documents:\\' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write \\'None\\'.\n' }}{% if citation_mode=='accurate' %}{{ 'Thirdly, Write \\'Answer:\\' followed by a response to the user\\'s last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}
# tokenizer.push_to_hub(model_name)  # Upload your new template to the Hub!


# tokenizer.default_chat_template = {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '# Safety Preamble' }}{{ '\nThe instructions in this section override those in the task description and style guide sections. Don\\'t answer questions that are harmful or immoral.' }}{{ '\n\n# System Preamble' }}{{ '\n## Basic Rules' }}{{ '\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\'s requests, you cite your sources in your answers, according to those instructions.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{{ 'Firstly, Decide which of the retrieved documents are relevant to the user\\'s last input by writing \\'Relevant Documents:\\' followed by comma-separated list of document numbers. If none are relevant, you should instead write \\'None\\'.\n' }}{{ 'Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user\\'s last input by writing \\'Cited Documents:\\' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write \\'None\\'.\n' }}{% if citation_mode=='accurate' %}{{ 'Thirdly, Write \\'Answer:\\' followed by a response to the user\\'s last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}
# sys.exit()

# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"
# #################################################################
# # bitsandbytes parameters
# #################################################################

# # Activate 4-bit precision base model loading
# use_4bit = True

# # Compute dtype for 4-bit base models
# bnb_4bit_compute_dtype = "float16"

# # Quantization type (fp4 or nf4)
# bnb_4bit_quant_type = "nf4"

# # Activate nested quantization for 4-bit base models (double quantization)
# use_nested_quant = False

# #################################################################
# # Set up quantization config
# #################################################################
# compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# bnb_config = BitsAndBytesConfig(
#    load_in_4bit=use_4bit,
#    bnb_4bit_quant_type=bnb_4bit_quant_type,
#    bnb_4bit_compute_dtype=compute_dtype,
#    bnb_4bit_use_double_quant=use_nested_quant,
# )

# # Check GPU compatibility with bfloat16
# if compute_dtype == torch.float16 and use_4bit:
#    major, _ = torch.cuda.get_device_capability()
#    if major >= 8:
#        print("=" * 80)
#        print("Your GPU supports bfloat16: accelerate training with bf16=True")
#        print("=" * 80)
       

# #################################################################
# # Load pre-trained config
# #################################################################
# model = AutoModelForCausalLM.from_pretrained(
#    model_name,
#    quantization_config=bnb_config,
#    attn_implementation="flash_attention_2",
#    trust_remote_code=True,
# )


import os


import pandas as pd

import random
# sample_df = pd.read_pickle('traintest_dataset_rbs_psych.pkl')

from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
import nest_asyncio
nest_asyncio.apply()
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
# query = """
# Welche Bearbeitungsschritte der wischaftlicher Sozialhilfe müssen gemacht werden beim Todesfall eines Klienten?
# """
query = """
## Instruktion: Es geht hier nur um die Klienten in der wirtschaftlichen Sozialhilfe. Bitte beantworten Sie die Frage nur in Bezug auf die wirtschaftliche Sozialhilfe.
Müssen Zahlungen bei Todesfall in der wirtschaftlichen Sozialhilfe eingestellt werden vom Sozialamt?
"""

# define conversation input:
conversation = [
    {"role": "user", "content": """## Instruktion: Es geht hier nur um die Klienten in der wirtschaftlichen Sozialhilfe. Bitte beantworten Sie die Frage nur in Bezug auf die wirtschaftliche Sozialhilfe.
     ## Frage: Müssen Zahlungen bei Todesfall in der wirtschaftlichen Sozialhilfe eingestellt werden vom Sozialamt?
"""

# """
     }]

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, 
                                             quantization_config=bnb_config,
                                              attn_implementation="flash_attention_2",
                                             device_map=("auto")
                                              )


import spacy
from langchain.text_splitter import SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
# text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# text_splitter = SpacyTextSplitter(pipeline="de_core_news_lg", chunk_size=1000, chunk_overlap=100)
import re 
import gc

from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

class Document2:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = ''

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"


    

# with open('rb.txt', 'w') as file:
#     file.write(text)


# loader = TextLoader("rb.txt")
txt_files_folder = './txt_files'
txt_files = [os.path.join(txt_files_folder, file) for file in os.listdir(txt_files_folder) if file.endswith('.txt')]

documents = []
for file in txt_files:
    loader = TextLoader(file)
    docs_transformed = loader.load()
    chunked_documents = text_splitter.split_documents(docs_transformed)
    documents.extend(chunked_documents)
# loader = TextLoader(docs)
# docs_transformed = loader.load()    
# chunked_documents = text_splitter.split_documents(docs_transformed)
    
    
db = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large'))
retriever = db.as_retriever(
search_type="similarity",
search_kwargs={'k': 20, },    
    )
docs = retriever.get_relevant_documents(query)
# pretty_print_docs(docs)
# sys.exit()
# list_chunks = [doc.page_content for doc in docs]
#     # scores = reranker.compute_score([[query, chunk] for chunk in list_chunks])
#     # # print(scores)
#     # try:
#     #     list_chunks2 = [chunk for _, chunk in sorted(zip(scores, list_chunks), key=lambda x: x[0], reverse=True)]
#     # except TypeError:
#     #     list_chunks2 = []   
#     # # print(list_chunks2)
#     # dict_chunks2 = [Document2(text) for text in list_chunks2]
# documents= [doc.page_content for doc in docs]
#     # pretty_print_docs(dict_chunks2)
# documents2 = []
# for i, doc in enumerate(documents, start=1):
#         documents2.append({
#                             # "title": i, 
#                            "Kontext": doc})
#     # # Print the list
#     # print(documents2)
#     # for i, chunk in enumerate(extracted_texts, start=1):

# torch.cuda.empty_cache()
# gc.collect()    
    
    # documents2 = documents2[:2]
    # print(documents2)
    # return documents
chunked_documents = []
for doc in docs:
    chunked_documents.append({"title": doc.metadata, "text": doc.page_content})
    # chunked_documents = [
    # { "title": "Tall penguins", "text": "Emperor penguins are the tallest growing up to 122 cm in height." }, 
    # { "title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica."}
    # ]
grounded_generation_prompt = tokenizer.apply_grounded_generation_template(
conversation,
documents=chunked_documents,
citation_mode="accurate", #"accurate", # or 
tokenize=False,
add_generation_prompt=True,
return_tensors="pt",
)
# print(grounded_generation_prompt)
input_ids = tokenizer.encode(grounded_generation_prompt, add_special_tokens=True, return_tensors="pt")
input_ids = input_ids.to('cuda')
gen_tokens = model.generate(
input_ids, 
max_new_tokens=2024, 
do_sample=True, 
temperature=0.1,
top_k=50, top_p=0.90, 
repetition_penalty=1.1,
)
    # gen_text = tokenizer.decode(gen_tokens[0])
new_tokens = gen_tokens[0, input_ids.shape[1]:]
gen_text = tokenizer.decode(new_tokens)
    # db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large'))
    # retriever = db.as_retriever(
    # search_type="similarity",
    # search_kwargs={'fetch_k': 20, 'k': 10}
    # )


    # rag_chain = (
    # {"context": retriever, "question": RunnablePassthrough()}
    # | llm_chain
    # )
    # result= rag_chain.invoke(query)
    # print(result['text'])
print(gen_text)
with open('output.txt', 'w') as file:
    file.write(gen_text)
    # return gen_text, list_chunks, docs
sys.exit()
def extract_context(text):    
    try:
        extracted_texts = [doc.split("page_content='")[1].split("', ")[0] for doc in text.split("Document(")[1:]]
        # Print the extracted texts (you can save them or process further as needed)
        # print(extracted_texts)
        extracted_context = ""
        for i, chunk in enumerate(extracted_texts, start=1):
            extracted_context += f"TEXTABSCHNITT {i}:\n {chunk}\n"
        # print(extracted_context)
        return extracted_context
    except IndexError:
        # print("IndexError: list index out of range")
        return None
    
# context = {}
answer = []
list_chunks1 = []
list_chunks2 = []
for index, row in rbs_psych_eltern_sample.iterrows():
    text = row['text']
    # antwort, list1, list2 = get_answer(text)
    answer.append(antwort)
    # context[index] = kontext
    list_chunks1.append(list1)
    list_chunks2.append(list2)
    

rbs_psych_eltern_sample['antwort'] = answer
# rbs_psych_eltern_sample['context'] = context
rbs_psych_eltern_sample['list_chunks1'] = list_chunks1
rbs_psych_eltern_sample['list_chunks2'] = list_chunks2

# rbs_psych_eltern_sample.to_pickle('rbs_psych_eltern_sampleTemp2.pkl')
rbs_psych_eltern_sample.to_excel('rbs_psych_eltern_sampleTemp2.xlsx')
rbs_psych_eltern_sample0= pd.read_excel('rbs_psych_eltern_sampleTemp2.xlsx', sheet_name='Sheet1')
rbs_psych_eltern_sample0['extracted_context'] = rbs_psych_eltern_sample0['list_chunks2'].apply(lambda x: extract_context(x))
rbs_psych_eltern_sample0.to_excel('rbs_psych_eltern_sampleTemp2.xlsx')
rbs_psych_eltern_sample=rbs_psych_eltern_sample0.copy()

# rbs_psych_eltern_sample.to_pickle('commandR_rbs_psych_eltern_sampleTemp.pkl')
# rbs_psych_eltern_sample.to_excel('./psych_klient_classification/commandR_rbs_psych_eltern_sampleTemp.xlsx')
rbs_psych_eltern_sample.to_excel('rbs_ecan_klient_sample_antwort_llamaIndexRerank1-1000CommandR.xlsx')
sys.exit()

text_generation_pipeline = transformers.pipeline(
   model=model,
   tokenizer=tokenizer,
   task="text-generation",
   temperature=0.2,
   repetition_penalty=1.1,
#    return_full_text=True,
   max_new_tokens=300,
   do_sample=True,
   top_k=50, top_p=0.90, 
   pad_token_id=tokenizer.eos_token_id, return_full_text=False,
)

prompt_template = """
<s>[INST] 
Instruction: Answer the question based on provided context, do not interpret or speculate. Here is context to help:

{context}

### QUESTION:
{question}

[/INST]
"""

mixtral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
   input_variables=["context", "question"],
   template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=mixtral_llm, prompt=prompt)


# llm_chain.invoke({"context":"",
#                  "question": "What were the two main things the author worked on before college?"})

from langchain_core.runnables import RunnablePassthrough
query = "Liste klare, explizite Hinweise für psychische Störungen, psychische Erkrankungen, Persönlichkeitsstörungen, psychologische oder psychiatrische Behandlungen beim Kind, auch in der Vergangenheit? Damit ist gemeint: psychologische oder psychiatrische Therapie/Betreuung, welche besucht, verweigert wird oder empfohlen wird, Einweisung in eine psychiatrische Klinik, wenn psychische, psychiatrische Diagnosen diagnostiziert sind wie: Depressionen, Angststörungen, Wahn, Bipolar, Borderline Persönlichkeitsstörungen."



import logging
logging.getLogger().setLevel(logging.ERROR)


# answer = []
# for index, row in sample_20.iterrows():
#     text = row['text']
#     antwort = get_answer(text)
#     answer.append(antwort)
    

# sample_20['antwort'] = answer

# # sample_20.to_pickle('sample_df.pkl')
# sample_20.to_excel('sample_df_llammaIndex.xlsx')


   #  chunked_documents = text_splitter.create_documents(docs_transformed)
    
   


def extract_context(text):    
    try:
        extracted_texts = [doc.split("page_content='")[1].split("',")[0] for doc in text.split("Document(")[1:]]
        # Print the extracted texts (you can save them or process further as needed)
        # print(extracted_texts)
        extracted_context = '\n'.join(extracted_texts)
        extracted_context = '\nChunk: '.join(line for line in extracted_context.splitlines() if line.strip())
      #   print(extracted_context)
        return extracted_context
    except IndexError:
        # print("IndexError: list index out of range")
        return None
    
import logging
logging.getLogger().setLevel(logging.ERROR)

context = {}
answer = []
for index, row in rbs_psych_eltern_sample.iterrows():
    text = row['text']
    # antwort, kontext = get_answer(text)
    answer.append(antwort)
    context[index] = kontext
    

rbs_psych_eltern_sample['antwort'] = answer
rbs_psych_eltern_sample['context'] = context


rbs_psych_eltern_sample.to_pickle('rbs_psych_eltern_sampleTemp.pkl')
rbs_psych_eltern_sample.to_excel('rbs_psych_eltern_sampleTemp.xlsx')
rbs_psych_eltern_sample0= pd.read_excel('rbs_psych_eltern_sampleTemp.xlsx', sheet_name='Sheet1')
rbs_psych_eltern_sample0['extracted_context'] = rbs_psych_eltern_sample0['context'].apply(lambda x: extract_context(x))
rbs_psych_eltern_sample0.to_excel('rbs_psych_eltern_sampleTemp.xlsx')
rbs_psych_eltern_sample=rbs_psych_eltern_sample0.copy()


# Save rbs_alk_eltern_sample10 as an Excel file
rbs_psych_eltern_sample.to_excel('rbs_psych_klient_sample_antwort_recursFaiss1-500.xlsx', index=False)


sys.exit()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
model_name ="deepset/gelectra-large"
# Replace this with your own checkpoint
tokenizer = AutoTokenizer.from_pretrained("psych_prompt_class_gelectra400", do_lower_case = True,
                                         use_fast=True,
                                         eos_token='###', pad_token='[PAD]',)
                                          
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.add_special_tokens({'eos_token': '###'})
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained("psych_prompt_class_gelectra400").to("cuda")

predict_class = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer)

# device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# text=str(rbs_psych_eltern_sample10["antwort"][0:1])
# inputs = tokenizer(text, return_tensors="pt")
# inputs=inputs.to(device)
# print(inputs)


# with torch.no_grad():
#     logits = model(**inputs)


# label= torch.argmax(logits[0]).item()
# print(label)
rbs_psych_eltern_sample['label'] = None

rbs_psych_eltern_sample.reset_index(inplace=True)

model.eval()
for i in range(len(rbs_psych_eltern_sample)): 
         
        text=str(rbs_psych_eltern_sample.loc[i, "antwort"])
        # print(text)
        inputs = tokenizer(text, return_tensors="pt")
        inputs=inputs.to(device)
        # print(inputs)
        with torch.no_grad():
            logits = model(**inputs)
        predicted_class_id = torch.argmax(logits[0]).item()
        # print(predicted_class_id)
        rbs_psych_eltern_sample.loc[i, "label"]=predicted_class_id
        
rbs_psych_eltern_sample_codiert=rbs_psych_eltern_sample.copy()





# Save DataFrame as pkl file
rbs_psych_eltern_sample_codiert.to_pickle('rbs_psych_eltern_sample_codiert.pkl')


# Save DataFrame as Excel file
rbs_psych_eltern_sample_codiert.to_excel('rbs_psych_eltern_sample_codiert.xlsx', index=False)

