import discord
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the trained GPT-2 model and tokenizer
model_path = r"folder model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Use GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up the Discord client with message content intent
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
client = discord.Client(intents=intents)

# Define the bot token (replace with your actual token)
DISCORD_TOKEN = 'Discord bot token"

# Function to generate a response from the model
def generate_response(input_text):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate the response using the trained model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated response and return it
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Event handler when the bot is ready
@client.event
async def on_ready():
    print(f'Bot logged in as {client.user}')

# Event handler for when a message is received
@client.event
async def on_message(message):
    try:
        # Avoid responding to itself
        if message.author == client.user:
            return

        # Check if the message starts with "!chat"
        if message.content.startswith('!chat'):
            # Get the user input after the "!chat" command
            user_input = message.content[len('!chat '):]

            # Generate the bot response
            bot_response = generate_response(user_input)

            # Send the response back to the Discord channel
            await message.channel.send(bot_response)

        # Respond to direct messages (DMs)
        elif isinstance(message.channel, discord.DMChannel):
            # Generate the bot response for DMs
            bot_response = generate_response(message.content)

            # Send the response back in the DM
            await message.channel.send(bot_response)

    except Exception as e:
        print(f"Error: {e}")
        await message.channel.send("An error occurred while processing your request.")

# Run the bot
client.run(DISCORD_TOKEN)
