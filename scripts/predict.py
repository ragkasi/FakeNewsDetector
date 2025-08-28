from transformers import pipeline
# Load the classifier
classifier = pipeline("text-classification", model="models/bert-liar-fake-news", tokenizer="bert-base-uncased")
# Define the predict function
def predict(text):
    result = classifier(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score
# Example usage
if __name__ == "__main__":
    text = input("Enter a statement to evaluate:\n")
    label, score = predict(text)
    print(f"Prediction: {label} (Confidence: {score:.2f})")