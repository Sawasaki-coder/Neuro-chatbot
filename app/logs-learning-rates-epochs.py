import tensorflow as tf
import matplotlib.pyplot as plt

# Path to your event log file
events_file_path = r"C:\Users\nn436\Documents\test_code\neuro-sama\logs"

# Lists to store the extracted values
losses = []
learning_rates = []
epochs = []

# Read the event log file and extract relevant data
for summary in tf.compat.v1.train.summary_iterator(events_file_path):
    for value in summary.summary.value:
        # Print tags to identify the correct ones
        print(f"Tag: {value.tag}, Step: {summary.step}, Value: {value.simple_value}")

        if value.tag == 'loss':  # Adjust the tag name based on printed output
            losses.append(value.simple_value)
            epochs.append(summary.step)  # Use step as epoch
        elif value.tag == 'learning_rate':  # Adjust the tag name based on printed output
            learning_rates.append(value.simple_value)

# Check if data was extracted
if not epochs or not losses:
    print("No data extracted from the event log file.")
    exit()

# Create a DataFrame for easier management
import pandas as pd

df = pd.DataFrame({
    'Epoch': epochs,
    'Loss': losses[:len(epochs)],  # Adjust length to match epochs
    'Learning Rate': learning_rates[:len(epochs)]  # Adjust length to match epochs
})

# Print the DataFrame to verify extraction
print(df)

# Plot Loss vs Epoch
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Loss'], label='Loss', color='blue', marker='o')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# Plot Learning Rate vs Epoch
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Learning Rate'], label='Learning Rate', color='green', marker='x')
plt.title('Learning Rate vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.legend()
plt.show()

# Save Loss plot as PNG
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Loss'], label='Loss', color='blue', marker='o')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig('loss_vs_epoch.png')

# Save Learning Rate plot as PNG
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Learning Rate'], label='Learning Rate', color='green', marker='x')
plt.title('Learning Rate vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.legend()
plt.savefig('learning_rate_vs_epoch.png')

print("Plots have been generated and saved as PNG files.")
