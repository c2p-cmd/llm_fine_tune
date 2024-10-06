import os
import datasets
import nltk
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import torch

# Set up device
device = torch.device('mps' if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# Load datasets
def load_datasets():
    train_dataset = datasets.load_dataset('c2p-cmd/Good-Quotes-Authors', name='train')
    test_dataset = datasets.load_dataset('c2p-cmd/Good-Quotes-Authors', name='test')
    return train_dataset, test_dataset

# Plot sentence length distribution
def plot_sentence_length_distribution(train_dataset, test_dataset):
    def sentence_length(s: str) -> int:
        return len(s)

    train_sentence_len_frequency = list(map(sentence_length, train_dataset['train']['text']))
    test_sentence_len_frequency = list(map(sentence_length, test_dataset['train']['content']))

    train_freq_dist = nltk.FreqDist(train_sentence_len_frequency)
    test_freq_dist = nltk.FreqDist(test_sentence_len_frequency)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(list(train_freq_dist.keys()), list(train_freq_dist.values()))
    plt.xlabel('Lengths')
    plt.ylabel('Frequency')
    plt.title("Sentence Length Distribution in Train Set")

    plt.subplot(1, 2, 2)
    plt.bar(list(test_freq_dist.keys()), list(test_freq_dist.values()))
    plt.xlabel('Lengths')
    plt.ylabel('Frequency')
    plt.title("Sentence Length Distribution in Test Set")

    plt.tight_layout()
    plt.savefig('sentence_length_distribution.png')
    plt.close()

# Prepare data
def prepare_data(dataset, tokenizer, max_len, col_name, name, max_length):
    with open(name, 'wt') as file:
        texts = []
        for item in dataset['train'][col_name]:
            text = item
            if len(text) <= max_len:
                formatted_text = f"{tokenizer.bos_token}{text.strip()}{tokenizer.eos_token}"
                texts.append(formatted_text)
        file.writelines(texts)

    return TextDataset(
        tokenizer,
        file_path=name,
        block_size=max_length
    )

# Training function
def train_model(train_dataset, test_dataset, model_name, output_dir, num_train_epochs, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    MAX_LEN = 150
    BLOCK_SIZE = 256  # maximum sequence length for GPT-2

    # Prepare and tokenize data
    train_dataset = prepare_data(train_dataset, tokenizer, MAX_LEN, 'text', 'train_data.txt', BLOCK_SIZE)
    test_dataset = prepare_data(test_dataset, tokenizer, MAX_LEN, 'content', 'test_data.txt', BLOCK_SIZE)

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=1000,
        warmup_steps=500,
        save_total_limit=2,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# Inference function
def generate_text(model_path, seed_text, max_length=150, num_return_sequences=3):
    device = torch.device('mps' if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

    gen_kwargs = {
        'max_length': max_length,
        'num_return_sequences': num_return_sequences,
        'no_repeat_ngram_size': 2,
        'do_sample': True,
        'top_k': 50,
        'top_p': 0.95,
        'temperature': 0.7,
    }

    generated_texts = generator(seed_text, **gen_kwargs)

    return [text['generated_text'] for text in generated_texts]

# Main execution
if __name__ == "__main__":
    # Load datasets
    train_dataset, test_dataset = load_datasets()

    # Plot sentence length distribution
    plot_sentence_length_distribution(train_dataset, test_dataset)

    # Train the model
    model_name = 'gpt2'
    output_dir = './gpt2-quotor-v3'
    num_train_epochs = 5
    batch_size = 16

    train_model(train_dataset, test_dataset, model_name, output_dir, num_train_epochs, batch_size)

    # Generate text
    seed_text = "Life is "
    generated_quotes = generate_text(output_dir, seed_text)

    print("Generated Quotes:")
    for i, quote in enumerate(generated_quotes, 1):
        print(f"{i}. {quote}\n")