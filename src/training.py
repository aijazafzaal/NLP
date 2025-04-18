from matplotlib import pyplot as plt
import torch
import argparse
import os
import pandas as pd
import seaborn as sns
from utils import get_texts_and_labels, get_tokens_and_encodings, OffensiveDataset, get_training_parameters, get_trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocessing import remove_hashtags_urls_at, lowercasing, expand_emojis, remove_stopwords, stemming
from sklearn.metrics import classification_report, confusion_matrix

def training():

    parser = argparse.ArgumentParser(
        description="Fine-tune models"
        )
    
    parser.add_argument(
    "--task",
    type=str,
    choices=[
        "A-eng", "B-eng", "C-eng", "A-danish"
    ],
    required=True,
    help="Choose task to fine-tune model on: A-eng, B-eng, C-eng, or A-danish"
    )

    parser.add_argument(
    "--model",
    type=str,
    choices=[
        "bert-base-cased", "albert-base-v2", "bert-base-multilingual-cased"
    ],
    required=True,
    help="Specify the model name (e.g., albert-base-v2)"
    )

    parser.add_argument(
    "--hyperparameters",
    type=str,
    choices=[
        "batch-1", "batch-16", "epochs-6", "lr-1e-5", "lr-3e-5"
    ],
    default=None,
    help="Default is size = 8, epochs = 3, lr = 2e-5. Or change one at a time"
    )

    parser.add_argument(
    "--full-dataset",
    action="store_true",
    help="Use the full dataset instead of a 1000-tweet sample"
    )

    parser.add_argument(
    "--preprocessing-data",
    type=str,
    nargs="+",  # allows multiple values
    choices=[
        "remove-hashtags-urls-at", "lowercasing", "expand-emojis", "remove-stopwords", "stemming"
    ],
    default=None,
    help="Specify one or several data preprocessing techniques (e.g., lowercasing or lowercasing and stemming)"
    )

    args = parser.parse_args()

    # TRAIN MODEL
    print("*****TRAINING MODEL*****")
    print("--Task: ", args.task)

    # Construct path to a training file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.task == "A-eng" or args.task == "B-eng" or args.task == "C-eng":
        path = os.path.join(script_dir, "..", "data", "data_eng", "data_training", "olid-training-v1.0.tsv")
    else: # if the dataset is Danish
        path = os.path.join(script_dir, "..", "data", "data_danish", "data_training", "offenseval-da-training-v1.tsv")

    # Load data
    data = pd.read_csv(path, sep="\t")

    model_to_try = args.model
    print("--Model: ", args.model)

    # Create data sample
    if args.full_dataset:
        print("--Training on full dataset")
    else:
        print("--Training on a sample of 1000 tweets")
        data = data.sample(n=1000, random_state=42)

    # Preprocess data
    if args.preprocessing_data:
        print("--Using data preprocessing techniques: ", args.preprocessing_data)
        if args.preprocessing_data == "remove_hashtags_urls_at":
            data["tweet"] = data["tweet"].astype(str).apply(remove_hashtags_urls_at)
        elif args.preprocessing_data == "lowercasing":
            data["tweet"] = data["tweet"].astype(str).apply(lowercasing)
        elif args.preprocessing_data == "expand_emojis":
            data["tweet"] = data["tweet"].astype(str).apply(expand_emojis)
        elif args.preprocessing_data == "remove_stopwords":
            data["tweet"] = data["tweet"].astype(str).apply(remove_stopwords)
        elif args.preprocessing_data == "stemming":
            data["tweet"] = data["tweet"].astype(str).apply(stemming)
    else:
        print("--Using no data preprocessing technique")

    # Convert labels to numerical form
    if args.task == "A-eng":
        label_mapping = {"NOT": 0, "OFF": 1}
        subtask = "subtask_a"
    # Convert labels to numerical form
    elif args.task == "B-eng":
        # Remove unnecessary raws from task B
        data = data.dropna(subset=["subtask_b"])
        label_mapping = {"UNT": 0, "TIN": 1}
        subtask = "subtask_b"
    elif args.task == "C-eng":
        # Remove unnecessary raws from task C
        data = data.dropna(subset=["subtask_c"])
        label_mapping = {"OTH": 0, "IND": 1, "GRP": 2}
        subtask = "subtask_c"
    else: # task A-danish
        # Remove unnecessary raws from task A
        data = data.dropna(subset=["subtask_a"])
        label_mapping = {"NOT": 0, "OFF": 1}
        subtask = "subtask_a"
    data[subtask] = data[subtask].map(label_mapping)
    num_labels=len(label_mapping)

    # Split the data into train and eval datasets and get texts and labels in a special form of a list
    train_texts, train_labels, eval_texts, eval_labels = get_texts_and_labels(data, subtask)

    # Convert texts into tokens and labels into tensor encodings
    tokenizer = AutoTokenizer.from_pretrained(model_to_try)
    train_texts_tokens, eval_texts_tokens, train_labels_encodings, eval_labels_encodings = get_tokens_and_encodings(train_texts, train_labels, eval_texts, eval_labels, tokenizer)
    tokenizer.save_pretrained("./offensive_detector")

    # Apply OffensiveDataset to create batches while training
    train_batches = OffensiveDataset(train_texts_tokens, train_labels_encodings)
    eval_batches = OffensiveDataset(eval_texts_tokens, eval_labels_encodings)

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_to_try, num_labels=num_labels)

    # Set up key training parameters
    output_dir="./results" #save logs
    if args.hyperparameters is None:
        print("--Using batch_size=8, epochs=3, lr=2e-5")
        training_args = get_training_parameters(batch_size=8, epochs=3, lr=2e-5, output_dir=output_dir)
    elif args.hyperparameters == "batch-1":
        print("--Using batch_size=1, epochs=3, lr=2e-5")
        training_args = get_training_parameters(batch_size=1, epochs=3, lr=2e-5, output_dir=output_dir)
    elif args.hyperparameters == "batch-16":
        print("--Using batch_size=16, epochs=3, lr=2e-5")
        training_args = get_training_parameters(batch_size=16, epochs=3, lr=2e-5, output_dir=output_dir)
    elif args.hyperparameters == "epochs-6":
        print("--Using batch_size=8, epochs=6, lr=2e-5")
        training_args = get_training_parameters(batch_size=8, epochs=6, lr=2e-5, output_dir=output_dir)
    elif args.hyperparameters == "lr-1e-5":
        print("--Using batch_size=8, epochs=3, lr=1e-5")
        training_args = get_training_parameters(batch_size=8, epochs=6, lr=1e-5, output_dir=output_dir)
    elif args.hyperparameters == "lr-3e-5":
        print("--Using batch_size=8, epochs=3, lr=3e-5")
        training_args = get_training_parameters(batch_size=8, epochs=6, lr=3e-5, output_dir=output_dir)
    
    #Set up Trainer
    trainer = get_trainer(model, training_args, train_batches, eval_batches)
    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained("./offensive_detector")

    # EVALUATING MODEL
    print("*****EVALUATING MODEL*****")
    print("--Evaluating on the full test dataset")

    # Construct paths to a test text file and golden label file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.task == "A-danish":
        path_data = os.path.join(script_dir, "..", "data", "data_danish", "data_test", "offenseval-da-test-v1.tsv")
        data = pd.read_csv(path_data, sep='\t') # Load data and golden_labels
    
    else:
        if args.task == "A-eng":
            testset = "testset-levela.tsv"
            labels = "labels-levela.csv"
        
        elif args.task == "B-eng":
            testset = "testset-levelb.tsv"
            labels = "labels-levelb.csv"
        
        else: # test "C-eng"
            testset = "testset-levelc.tsv"
            labels = "labels-levelc.csv"

        path_data = os.path.join(script_dir, "..", "data", "data_eng", "data_test", testset)
        path_gold_labels = os.path.join(script_dir, "..", "data", "data_eng", "data_test", labels)
        
        # Load data and golden_labels
        data = pd.read_csv(path_data, sep='\t')
        gold_labels = pd.read_csv(path_gold_labels, names=["id", "label"])

        # Merge texts and labels
        data = data.merge(gold_labels, on="id")

    # Convert labels into integers
    if args.task == "A-eng" or args.task == "A-danish":
        label_mapping = {"NOT": 0, "OFF": 1}
    elif args.task == "B-eng":
        label_mapping = {"UNT": 0, "TIN": 1}
    else: # test C-eng
        label_mapping = {"OTH": 0, "IND": 1, "GRP": 2}

    if args.task == "A-eng" or args.task == "B-eng" or args.task == "C-eng":
        data["label"] = data["label"].map(label_mapping)
    else: # test A-danish
        data["subtask_a"] = data["subtask_a"].map(label_mapping)

    # Preprocess data
    if args.preprocessing_data:
        print("--Using data preprocessing techniques: ", args.preprocessing_data)
        if args.preprocessing_data == "remove_hashtags_urls_at":
            data["tweet"] = data["tweet"].astype(str).apply(remove_hashtags_urls_at)
        elif args.preprocessing_data == "lowercasing":
            data["tweet"] = data["tweet"].astype(str).apply(lowercasing)
        elif args.preprocessing_data == "expand_emojis":
            data["tweet"] = data["tweet"].astype(str).apply(expand_emojis)
        elif args.preprocessing_data == "remove_stopwords":
            data["tweet"] = data["tweet"].astype(str).apply(remove_stopwords)
        elif args.preprocessing_data == "stemming":
            data["tweet"] = data["tweet"].astype(str).apply(stemming)
    else:
        print("--Using no data preprocessing technique")

    # Pack test data and labels in a list
    test_texts = data["tweet"].tolist()
    if args.task == "A-danish":
        gold_labels = data["subtask_a"].tolist()
    else:
        gold_labels = data["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained("./offensive_detector")
    tokens = tokenizer(test_texts, truncation=True, padding=True, return_tensors = "pt") # Get tokens

    # Load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained("./offensive_detector")

    # Make predictions
    with torch.no_grad():
        logits = model(**tokens).logits
        predictions = torch.argmax(logits, dim=1).tolist()

    print("\nClassification Report:\n")
    print(classification_report(gold_labels, predictions, target_names=label_mapping.keys(), digits=4))
    # Compute confusion matrix
    conf_matrix = confusion_matrix(gold_labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("Please find your trained model in offensive_detector directory")
    print("Training logs are in results directory")

if __name__ == '__main__':
    training()