import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# This model did not work very well on 2 outputs generated by GPT4. It is a few years old so...

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load the checkpoint
model_path = '/Users/dustinhayes/Desktop/gpt-2-output-dataset/detector-base.pt'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Check if there's a 'config' in the checkpoint
if 'config' in checkpoint:
    config = checkpoint['config']
    model = RobertaForSequenceClassification(config)
else:
    # Fallback to standard model initialization
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Extract the model state dict and load it
model_state_dict = checkpoint.get('model_state_dict', checkpoint)  # Supports both types of checkpoints
model.load_state_dict(model_state_dict, strict=False) 
model.eval()  # Set the model to evaluation mode

inputs = tokenizer(
    "Python and Java are two widely-used programming languages, each with its own unique characteristics. Python is known for its clean and concise syntax, which emphasizes readability and reduces the need for excessive code. It employs indentation to define code blocks, making it quite approachable for beginners. In contrast, Java's syntax is more verbose, requiring semicolons and curly braces for code structure, which can make it appear less concise compared to Python. Another fundamental difference lies in their type systems. Python is dynamically typed, meaning that variable types are determined at runtime, offering flexibility but also potentially leading to runtime errors. Java, on the other hand, is statically typed, requiring explicit type declarations for variables. This static typing allows for catching type-related errors at compile-time, which can enhance code reliability.",
    padding=True, 
    truncation=True, 
    max_length=512, 
    return_tensors="pt"
)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits)

print(probabilities)