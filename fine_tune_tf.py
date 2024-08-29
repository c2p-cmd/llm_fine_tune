import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, create_optimizer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Set up GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load the pre-trained model and tokenizer
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Prepare the dataset
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return Dataset.from_dict({"text": texts})

# Load and prepare the training data
train_dataset = load_dataset("quotes_train.txt")
val_dataset = load_dataset("quotes_test.txt")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=False, max_length=30)

# Tokenize the datasets
train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)

sample_text = tokenizer.decode(val_tokenized[10]['input_ids'])
print("Sample tokenized and decoded text:", sample_text)

if input('continue? [y]') != 'y':
    exit(0)

# Set up the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")

# Prepare the TensorFlow datasets
train_tf_dataset = model.prepare_tf_dataset(
    train_tokenized,
    collate_fn=data_collator,
    shuffle=True,
    batch_size=8
)

val_tf_dataset = model.prepare_tf_dataset(
    val_tokenized,
    collate_fn=data_collator,
    shuffle=False,
    batch_size=8
)

# Set up training arguments
num_train_steps = len(train_tokenized) // 4 * 3  # 3 epochs
optimizer, schedule = create_optimizer(
    init_lr=1e-4,  # Reduced learning rate
    num_warmup_steps=100,
    num_train_steps=num_train_steps
)

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

class PrintLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} - Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")

# Start training
history = model.fit(
    train_tf_dataset,
    validation_data=val_tf_dataset,
    epochs=3,  # Reduced number of epochs
    callbacks=[early_stopping, PrintLossCallback()]
)

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_distilgpt2_tf")
tokenizer.save_pretrained("./fine_tuned_distilgpt2_tf")

def generate_quote(prompt, temp):
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    output = model.generate(
        input_ids,
        max_length=30,
        num_return_sequences=1,
        temperature=temp,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "Once a wise individual said,"
for temp in [0.5, 0.7, 1.0]:
    print(f"Temperature {temp}:")
    print(generate_quote(prompt, temp))
    print()

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()