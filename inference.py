from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

# Load the fine-tuned model
model = TFGPT2LMHeadModel.from_pretrained('./fine_tuned_distilgpt2_tf 2')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_distilgpt2_tf 2')
model.config.pad_token_id = tokenizer.pad_token_id

# Generate a sample quote
def generate_quote(prompt, temp):
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    attention_mask = tf.ones_like(input_ids)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=50,
        num_return_sequences=1,
        temperature=temp,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "Once a wise individual said,"
for temp in [0.5, 0.7, 1.0]:
    print(f"Temperature {temp}:")
    print(generate_quote(prompt, temp))
    print()