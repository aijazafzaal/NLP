from sklearn.model_selection import train_test_split
import torch as pt
from transformers import TrainingArguments, Trainer


# Data processing utilities

def get_texts_and_labels(data, subtask):
    # Split data into train_data and eval_data
    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Pack train data in lists
    train_texts = train_data["tweet"].tolist()
    train_labels = train_data[subtask].tolist()
    # Pack eval data in lists
    eval_texts = eval_data["tweet"].tolist()
    eval_labels = eval_data[subtask].tolist()

    return train_texts, train_labels, eval_texts, eval_labels

def get_tokens_and_encodings(train_texts, train_labels, eval_texts, eval_labels, tokenizer):
    # Get text tokens
    train_texts_tokens = tokenizer(train_texts, truncation=True, padding=True, return_tensors = "pt")
    eval_texts_tokens = tokenizer(eval_texts, truncation=True, padding=True, return_tensors = "pt")
    # Get label tokens
    train_labels_encodings = pt.tensor(train_labels)
    eval_labels_encodings = pt.tensor(eval_labels)

    return train_texts_tokens, eval_texts_tokens, train_labels_encodings, eval_labels_encodings

class OffensiveDataset(pt.utils.data.Dataset):
    def __init__(self, encodings, labels):
        """
        Parameters:
        -encodings: A dictionary containing 'input_ids' and 'attention_mask'
        -labels: A tensor containing the labels
        """
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        """
        Gets the number of samples in the dataset
        """
        return len(self.labels)
    
    def __getitem__(self, index):
        """
        Gets the sample by its index:
        -the 'input_ids' and 'attention_mask'
        -the label of the sample
        """
        input_data = {
            "input_ids": self.encodings['input_ids'][index],
            'attention_mask': self.encodings['attention_mask'][index],
            'labels': self.labels[index]}
        
        return input_data

# Training utilities
def get_training_parameters(batch_size, epochs, lr, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,  # Where to save model checkpoints
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        per_device_train_batch_size=batch_size,  # Number of samples per batch during training
        per_device_eval_batch_size=batch_size,  # Number of samples per batch during evaluation
        num_train_epochs=epochs,  # Number of times the model will go through the entire dataset
        learning_rate=lr,  # Learning rate (small values work best for transformers)
        weight_decay=0.01,  # Regularization to prevent overfitting
    )
    return training_args

def get_trainer(model, training_args, train_batches, eval_batches):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_batches,
        eval_dataset=eval_batches
    )
    return trainer



