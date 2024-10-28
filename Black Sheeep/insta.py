import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# Load the Instagram dataset
df = pd.read_csv(r'D:/Black Sheeep/instagram_data.csv')


# Convert Date and Time to a datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Convert the Time column to datetime with hours, minutes, and seconds
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time

# Combine Date and Time into a Datetime column
df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')

# Handle missing 'Likes', 'Comments' columns by checking if they exist in the dataframe
def prepare_input(row):
    likes = row['Likes'] if 'Likes' in row else 'N/A'
    comments = row['Comments'] if 'Comments' in row else 'N/A'
    input_text = (
        f"On {row['Date']} at {row['Time']}, the post with caption '{row['Caption']}' "
        f"received {row['Views']} views, {likes} likes, and {comments} comments."
    )
    return input_text

# Apply the function to create the 'input_text' column
df['input_text'] = df.apply(prepare_input, axis=1)

# Now you can inspect the dataframe
print(df[['input_text']].head())




# Load the tokenizer and model
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")


# Tokenize the input text for model training
tokenized_inputs = tokenizer(list(df['input_text']), padding=True, truncation=True, return_tensors="pt")

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['input_text']])

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    learning_rate=5e-5,
    warmup_steps=500,
)

# Initialize the Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Start the fine-tuning process
trainer.train()

# Function to generate SEO insights
def generate_insight(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test with a new data point
new_post_data = "On 2024-09-01, at 14:00, a post was made with caption: 'Exciting news!' and hashtags: news, announcement. " \
                "It received 1000 views, 150 likes, and 10 comments."

# Get insight from the fine-tuned model
insight = generate_insight(new_post_data)
print(insight)
