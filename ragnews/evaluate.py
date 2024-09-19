import sys
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from ragnews import rag  # Assuming ragnews.rag is the prediction function
import pandas as pd

class RAGClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, valid_labels):
        """
        Initialize the RAGClassifier with a set of valid labels.
        """
        self.valid_labels = valid_labels

    def fit(self, X, y=None):
        """
        This classifier does not require fitting, so we simply return self.
        """
        return self

    def predict(self, X):
        """
        Predict labels for the provided input using the rag function.
        X is expected to be a list of instances.
        """
        predictions = []
        for instance in X:
            # Use the rag function to predict the label
            predicted_label = rag(instance)
            # Ensure the predicted label is valid
            if predicted_label in self.valid_labels:
                predictions.append(predicted_label)
            else:
                # Handle invalid predictions (e.g., append a default label or None)
                predictions.append(None)
        return np.array(predictions)

    def score(self, X, y):
        """
        Compute the accuracy of the predictions compared to the true labels.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def extract_labels_and_data(file_path):
    """
    Extract possible labels and instances from the HairyTrumpet data file.
    Assume the file contains two columns: 'instance' and 'label'.
    """
    data = pd.read_csv(file_path)
    labels = data['label'].unique()  # Extract unique labels
    instances = data['instance']  # The input data
    true_labels = data['label']  # The true labels
    return instances, true_labels, labels


if __name__ == '__main__':
    # Check for command line arguments
    if len(sys.argv) != 2:
        print("Usage: python rag_classifier.py <path_to_hairy_trumpet_datafile>")
        sys.exit(1)
    
    # Get the path to the data file
    data_file_path = sys.argv[1]
    
    # Extract instances, true labels, and possible labels from the file
    instances, true_labels, valid_labels = extract_labels_and_data(data_file_path)
    
    # Create a RAGClassifier with the valid labels
    classifier = RAGClassifier(valid_labels)
    
    # Since this classifier does not need training, we skip fit()
    
    # Compute and print the accuracy
    accuracy = classifier.score(instances, true_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

