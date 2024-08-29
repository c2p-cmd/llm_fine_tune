from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model
model = TFGPT2LMHeadModel.from_pretrained('./fine_tuned_distilgpt2_tf')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_distilgpt2_tf')
model.config.pad_token_id = tokenizer.pad_token_id

# Generate a sample quote
input_text = "Once a wise individual said,"
input_ids = tokenizer.encode(input_text, return_tensors="tf")
print(input_ids)
output = model.generate(input_ids, max_length=60, num_return_sequences=1, no_repeat_ngram_size=2)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
