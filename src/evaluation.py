import argparse
import pandas as pd
import os
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import clean_text, remove_stopwords

def evaluation():

    parser = argparse.ArgumentParser(
        description="Evaluate model"
        )

    parser.add_argument(
    "--model",
    type=str,
    choices=[
        "bert-base-cased-task-A-eng", "bert-base-cased-task-B-eng", "bert-base-cased-task-C-eng", "bert-base-multilingual-cased-task-A-danish"
    ],
    required=True,
    help="Please choose one of our best fine-tuned models you want to evaluate"
    )

    args = parser.parse_args()

    # EVALUATING MODEL
    print("*****EVALUATING MODEL*****")
    print("--Testing on full dataset: ", args.model)

    # Construct paths to a test text file and golden label file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.model == "bert-base-multilingual-cased-task-A-danish":
        path_data = os.path.join(script_dir, "..", "data", "data_danish", "data_test", "offenseval-da-test-v1.tsv")
        data = pd.read_csv(path_data, sep='\t') # Load data and golden_labels
    else:
        if args.model == "bert-base-cased-task-A-eng":
            testset = "testset-levela.tsv"
            labels = "labels-levela.csv"
        
        elif args.model == "bert-base-cased-task-B-eng":
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
    if args.model == "bert-base-cased-task-A-eng" or args.model == "bert-base-multilingual-cased-task-A-danish":
        label_mapping = {"NOT": 0, "OFF": 1}
    elif args.model == "bert-base-cased-task-B-eng":
        label_mapping = {"UNT": 0, "TIN": 1}
    else: # test C-eng
        label_mapping = {"OTH": 0, "IND": 1, "GRP": 2}

    if args.model == "bert-base-cased-task-A-eng" or args.model == "bert-base-cased-task-B-eng" or args.model == "bert-base-cased-task-C-eng":
        data["label"] = data["label"].map(label_mapping)
    else: # test A-danish
        data["subtask_a"] = data["subtask_a"].map(label_mapping)

    if args.model == "bert-base-cased-task-C-eng":
        data["tweet"] = data["tweet"].astype(str).apply(remove_stopwords)
    elif args.model == "bert-base-multilingual-cased-task-A-danish":
        data["tweet"] = data["tweet"].astype(str).apply(clean_text)

    # Pack test data and labels in a list
    test_texts = data["tweet"].tolist()
    if args.model == "bert-base-multilingual-cased-task-A-danish":
        gold_labels = data["subtask_a"].tolist()
    else:
        gold_labels = data["label"].tolist()

    # Load tokenizer
    if args.model == "bert-base-cased-task-A-eng":
        tokenizer = AutoTokenizer.from_pretrained("JanikBERT/ANLP_FINAL_ENG_A")
    elif args.model == "bert-base-cased-task-B-eng":
        tokenizer = AutoTokenizer.from_pretrained("JanikBERT/ANLP_FINAL_ENG_B")
    elif args.model == "bert-base-cased-task-C-eng":
        tokenizer = AutoTokenizer.from_pretrained("JanikBERT/ANLP_FINAL_ENG_C")
    elif args.model == "bert-base-multilingual-cased-task-A-danish":
        tokenizer = AutoTokenizer.from_pretrained("aijazafzaal/bertlemmatization")

    if args.model == "bert-base-multilingual-cased-task-A-danish":
        tokens = tokenizer(test_texts, truncation=True, max_length=512, padding=True, return_tensors = "pt")
    else:
        tokens = tokenizer(test_texts, truncation=True, padding=True, return_tensors = "pt")

    # Load fine-tuned model
    if args.model == "bert-base-cased-task-A-eng":
        model = AutoModelForSequenceClassification.from_pretrained("JanikBERT/ANLP_FINAL_ENG_A")
    elif args.model == "bert-base-cased-task-B-eng":
        model = AutoModelForSequenceClassification.from_pretrained("JanikBERT/ANLP_FINAL_ENG_B")
    elif args.model == "bert-base-cased-task-C-eng":
        model = AutoModelForSequenceClassification.from_pretrained("JanikBERT/ANLP_FINAL_ENG_C")
    elif args.model == "bert-base-multilingual-cased-task-A-danish":
        model = AutoModelForSequenceClassification.from_pretrained("aijazafzaal/bertlemmatization")

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

if __name__ == '__main__':
    evaluation()