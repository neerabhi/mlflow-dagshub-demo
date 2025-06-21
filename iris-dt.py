import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model parameters
max_depth = 15
n_estimators = 10


mlflow.set_experiment('iris-dt')

with mlflow.start_run():
    # Train model
    dt = DecisionTreeClassifier(max_depth=max_depth,  random_state=42)
    dt.fit(X_train, y_train)

    # Predict
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and parameters
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    # mlflow.log_param('n_estimators', n_estimators)

    print('Accuracy:', accuracy)

# Optional: Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")   

    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(dt, 'decision Tree')