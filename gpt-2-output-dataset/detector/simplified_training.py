from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader
import torch
import pickle
from sklearn.metrics import accuracy_score

from dataset import EncodedDataset, Corpus


class Trainer:
    def __init__(self, data_path, num_epochs, starting_checkpoint, train_model_location,
                 optimizer, loss_fn, lr, save_model=True, transfer_learn=True):
        self.num_epochs = num_epochs
        # Load the tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        if transfer_learn:
            # Load the checkpoint
            model_path = starting_checkpoint
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

            # Check if there's a 'config' in the checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
                self.model = RobertaForSequenceClassification(config)
            else:
                # Fallback to standard model initialization
                self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

            # Extract the model state dict and load it
            model_state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(model_state_dict, strict=False)
        else:
            # Initialize with base RoBERTa weights
            self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

        self.trained_model_location = train_model_location

        # Initialize optimizer
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

        # Initialize loss function
        if loss_fn == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.save_model = save_model


    def prepare_data(self, data_path):
        corpus = Corpus(data_dir=data_path)
        train_data = EncodedDataset(texts=corpus.train_texts,
                                       labels=corpus.train_labels,
                                       tokenizer=self.tokenizer,
                                       max_sequence_length=512) # Initialize with your train data
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
        validation_data = EncodedDataset(texts=corpus.valid_texts,
                                            labels=corpus.valid_labels,
                                            tokenizer=self.tokenizer,
                                            max_sequence_length=512)
        validation_loader = DataLoader(validation_data, batch_size=8, shuffle=True)
        test_data = EncodedDataset(texts=corpus.test_texts,
                                   labels=corpus.test_labels,
                                   tokenizer=self.tokenizer,
                                   max_sequence_length=512) # Initialize with your train data
        test_loader = DataLoader(test_data, batch_size=8, shuffle=True)

        return train_loader, validation_loader, test_loader

    def train_model(self, train_loader, validation_loader,
                    test_loader):
        
        train_predictions = []
        train_true_labels = []
        validation_predictions = []
        validation_true_labels = []

        for epoch in range(self.num_epochs):
            train_predictions = []
            train_true_labels = []
            validation_predictions = []
            validation_true_labels = []
            self.model.train()
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            train_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                ###### Evaluation
                logits = outputs.logits
                # Convert logits to predictions
                batch_predictions = torch.argmax(logits, dim=1)
                train_predictions.extend(batch_predictions.tolist())
                train_true_labels.extend(labels.tolist())
                #########################

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            

            # Calculate average train loss
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = accuracy_score(train_true_labels, train_predictions)

            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in validation_loader:
                    input_ids, attention_mask, labels = batch
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    val_loss += loss.item()

                    #### Evaluation
                    logits = outputs.logits
                    # Convert logits to predictions
                    batch_predictions = torch.argmax(logits, dim=1)
                    validation_predictions.extend(batch_predictions.tolist())
                    validation_true_labels.extend(labels.tolist())
            
            


            # Calculate average validation loss
            val_accuracy = accuracy_score(validation_true_labels, validation_predictions)
            avg_val_loss = val_loss / len(validation_loader)
        
            print(f"""Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, 
                  Training Accuracy: {train_accuracy:.4f},
                  Validation Loss: {avg_val_loss:.4f},
                  Validation Accuracy: {val_accuracy:.4f}""")

        # Save model
        if self.save_model:
            torch.save(self.model.state_dict(), self.trained_model_location)



if __name__ == '__main__':

    #### Modify for run
    starting_checkpoint = '/Users/dustinhayes/Desktop/DEEP LEARNING FINAL PROJECT/gpt-2-output-dataset/detector-base.pt'
    data_path = 'gpt-2-output-dataset/detector/TestData'
    # def __init__(self, data_path, num_epochs, starting_checkpoint, train_model_location,
                #  optimizer, loss_fn): 
    num_epochs = 2
    train_model_location = 'gpt-2-output-dataset/detector/TrainedModels/test_training.pth'
    optimizer = 'Adam'
    loss_fn = 'cross_entropy'
    learning_rate = 2e-5
    trainer = Trainer(data_path=data_path,num_epochs=num_epochs,
                      starting_checkpoint=starting_checkpoint,
                      train_model_location=train_model_location,
                      optimizer=optimizer,
                      loss_fn=loss_fn, lr=learning_rate,
                      save_model=False,
                      transfer_learn=False)
    train_loader, validation_loader, test_loader = trainer.prepare_data(data_path=data_path)
    trainer.train_model(train_loader, validation_loader, test_loader)