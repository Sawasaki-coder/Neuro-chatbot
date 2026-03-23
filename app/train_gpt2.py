import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TrainerCallback
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset (Question and Answer format)
def load_data(file_path):
    questions = []
    answers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                questions.append(parts[0])
                answers.append(parts[1])
    return questions, answers

# Path to the dataset file
data_path = r'C:\Users\nn436\Documents\test_code\neuro-sama\dataset.txt'

# Step 2: Load data from dataset.txt
questions, answers = load_data(data_path)

# Step 3: Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

# Step 4: Create a custom Dataset class for the question-answer data
class QADataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=128):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        # Concatenate question and answer into one input string
        input_text = f"Question: {question} Answer: {answer}"

        # Tokenize the input text
        encodings = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Set labels equal to the input_ids (this is typical for language modeling tasks)
        encodings['labels'] = encodings['input_ids']

        # Return the tokenized data as a dictionary
        return {key: val.squeeze() for key, val in encodings.items()}

# Create the dataset
dataset = QADataset(questions, answers, tokenizer)

# Step 5: Initialize GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Step 6: Custom Callback to Save Loss during Training
class SaveLossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.losses.append(logs['loss'])

# Initialize the callback
save_loss_callback = SaveLossCallback()

# Step 7: Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',             # Output directory for model and logs
    num_train_epochs=3,                 # Number of training epochs
    per_device_train_batch_size=2,      # Batch size per device during training
    logging_dir='./logs',               # Directory to store logs
    logging_steps=10,                   # Log every 10 steps
    save_steps=500,                     # Save checkpoint every 500 steps
    save_total_limit=2,                 # Limit the number of model saves
    warmup_steps=100,                   # Number of warmup steps for learning rate scheduler
    save_strategy='steps',              # Save model by steps
)

# Step 8: Initialize the Trainer
trainer = Trainer(
    model=model,                        # The GPT-2 model
    args=training_args,                 # Training arguments
    train_dataset=dataset,              # The dataset to train on
    callbacks=[save_loss_callback]      # Add the callback to save the loss
)

# Step 9: Train the model
trainer.train()

# Step 10: Save the model and tokenizer after training
model.save_pretrained('./neuro_sama_model')
tokenizer.save_pretrained('./neuro_sama_model')

# Step 11: Save the training loss to a CSV file
with open("training_losses.csv", "w") as f:
    for loss in save_loss_callback.losses:
        f.write(f"{loss}\n")

print("Training completed and losses saved.")

# Step 12: Plot the training loss using matplotlib
loss_data = pd.read_csv('training_losses.csv', header=None)
loss_values = loss_data[0].tolist()

plt.plot(loss_values)
plt.title("Training Loss Over Time")
plt.xlabel("Steps (logged every 10 steps)")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
