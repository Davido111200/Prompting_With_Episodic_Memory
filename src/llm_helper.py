import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    tik1 = time.time()
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='auto')
    tok1 = time.time()
    print("Init time: ", tok1 - tik1)

    tik2 = time.time()
    text = """You are given a prompt with instruction and in-context examples and a test example, each is separated by 2 newlines. \
            Your job is to paraphrase 3 different instructions and do not include the examples in it. Start each instruction with 'Instruction'. \
                There are two classes that should not be changed for the instructions, which are 'great' and 'terrible'. Do not use other words for classification. \
                    Each of the the generated instructions should be a completed instruction. Do not give redundant sentences in your response. After having generated \
                        all the instructions, do not generate any more sentences. Do not answer the last example.\n\nIn this task, you are given sentences from movie reviews. The task is to classify a sentence as 'great' if the sentiment of the sentence is positive or as 'terrible' if the sentiment of the sentence is negative.\n\nReview: it 's not the ultimate depression-era gangster movie .\nSentiment: terrible\n\nReview: visually rather stunning\nSentiment: great\n\nReview: lame\nSentiment:"""
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=500)
    print("Response:")
    output_ids = outputs[0][len(inputs[0]):]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    resp = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("RESP: ", resp)
    tok2 = time.time()
    print("Generation time: ", tok2 - tik2)

# if __name__=="__main__":
#     main()