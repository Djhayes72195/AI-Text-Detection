from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader
import torch

from dataset import EncodedDataset, Corpus

num_epochs = 1

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load the checkpoint
model_path = '/Users/dustinhayes/Desktop/DEEP LEARNING FINAL PROJECT/gpt-2-output-dataset/detector-base.pt'
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

corpus = Corpus()


# Prepare data loaders
# Encoded Dataset init
# def __init__(self, texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizer,
#                max_sequence_length: int = None):
train_dataset = EncodedDataset(texts=corpus.train_texts,
                               labels=corpus.train_labels,
                               tokenizer=tokenizer,
                               max_sequence_length=512) # Initialize with your train data
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

# Define loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    print('got to epoch loop')
    for batch in train_loader:
        print('It seems to be running')
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()