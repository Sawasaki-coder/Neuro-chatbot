import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the loss data from the CSV file
loss_data = pd.read_csv('training_losses.csv', header=None)
loss_values = loss_data[0].tolist()

# Step 2: Plot the training loss
plt.plot(loss_values)
plt.title("Training Loss Over Time")
plt.xlabel("Steps (logged every 10 steps)")
plt.ylabel("Loss")
plt.grid(True)

# Step 3: Save the plot as an image (e.g., PNG format)
plt.savefig("training_loss_chart.png")

# Display the plot (optional, depending on your environment)
plt.show()
