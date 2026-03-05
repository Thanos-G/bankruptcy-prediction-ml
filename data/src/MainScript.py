# Χειρισμός δεδομένων
import pandas as pd
import numpy as np
import os

# Προεπεξεργασία
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ταξινομητές
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

# Αξιολόγηση
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Εξαγωγή αποτελεσμάτων
import openpyxl

# Φόρτωση και προεπεξεργασία δεδομένων
df = pd.read_excel("data/Dataset2Use_Assignment.xlsx")
print(df.head())    # εμφάνιση 5 πρώτων γραμμών
df = df.drop(columns = ["ΕΤΟΣ"])    # διαγραφή στήλης "ΕΤΟΣ"
print(df.head())

df["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"] = df["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"].map({1: 0, 2: 1}) # κωδικοποίηση στόχου (0: Healthy, 1: Bankrupt)
X = df.iloc[:, :-1] # features, όλες οι στήλες πλην της τελευταίας
Y = df.iloc[:, -1]  # target, μόνο η τελευταια στήλη

# Κανονικοποίηση
scaler = StandardScaler()
scaler.fit(X)   
X_scaled = scaler.transform(X)

# Διαχωρισμός
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    Y,
    test_size = 0.25,
    random_state = 42,
    stratify = Y
)

# Έλεγχος ισορροπίας (Class balancing)
print(y_train.value_counts())


# Μετατροπή των X_train σε DataFrame
train_df = pd.DataFrame(X_train)
train_df["target"] = y_train.values

# Διαχωρισμός Healthy (0) και Bankrupt (1)
healthy_df = train_df[train_df["target"] == 0]
bankrupt_df = train_df[train_df["target"] == 1]


# Υπολογισμός στόχου
target_healthy_size = 3 * len(bankrupt_df)

# Downsample Healthy
healthy_downsampled = resample(
    healthy_df,
    replace=False,
    n_samples=target_healthy_size,
    random_state=42
)

# Δημιουργία balanced training set
train_balanced = pd.concat([healthy_downsampled, bankrupt_df]).sample(frac=1, random_state=42)

# Νέα balanced X_train και y_train
X_train_balanced = train_balanced.drop(columns=["target"]).values
y_train_balanced = train_balanced["target"].values

# Έλεγχος
print("\nHealthy AFTER:", sum(y_train_balanced == 0))
print("Bankrupt AFTER:", sum(y_train_balanced == 1))
print("Ratio AFTER:", sum(y_train_balanced == 0) / sum(y_train_balanced == 1))

# Linear Discriminant Analysis (LDA)
lda_model = LinearDiscriminantAnalysis()

lda_model.fit(X_train_balanced, y_train_balanced)

y_pred = lda_model.predict(X_test)

# Αξιολόγηση LDA
print("\n--- Αξιολόγηση LDA ---")
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("F1-score: ", f1_score(y_test, y_pred, zero_division=0))

# Logistic Regression (max_iter=1000)
log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_balanced, y_train_balanced)

y_pred = log_model.predict(X_test)

# Αξιολόγηση Logistic Regression
print("\n--- Αξιολόγηση Logistic Regression ---")
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("F1-score: ", f1_score(y_test, y_pred, zero_division=0))

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_balanced, y_train_balanced)

y_pred = knn_model.predict(X_test)

# Αξιολόγηση KNN (k=5)
print("\n--- Αξιολόγηση KNN (k=5) ---")
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("F1-score: ", f1_score(y_test, y_pred, zero_division=0))

knn_model = KNeighborsClassifier(n_neighbors=7)

knn_model.fit(X_train_balanced, y_train_balanced)

y_pred = knn_model.predict(X_test)

# Αξιολόγηση KNN (k=7)
print("\n--- Αξιολόγηση KNN (k=7) ---")
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("F1-score: ", f1_score(y_test, y_pred, zero_division=0))

knn_model = KNeighborsClassifier(n_neighbors=9)

knn_model.fit(X_train_balanced, y_train_balanced)

y_pred = knn_model.predict(X_test)

# Αξιολόγηση KNN (k=9)
print("\n--- Αξιολόγηση KNN (k=9) ---")
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("F1-score: ", f1_score(y_test, y_pred, zero_division=0))

# Naive Bayes (GaussianNB)
nb_model = GaussianNB()

nb_model.fit(X_train_balanced, y_train_balanced)

y_pred = nb_model.predict(X_test)

# Αξιολόγηση Naive Bayes
print("\n--- Αξιολόγηση Naive Bayes ---")
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("F1-score: ", f1_score(y_test, y_pred, zero_division=0))

# Support Vector Machines (SVM)
svm_model = SVC(kernel='linear')

svm_model.fit(X_train_balanced, y_train_balanced)

y_pred = svm_model.predict(X_test)

# Αξιολόγηση SVM    
print("\n--- Αξιολόγηση SVM ---")
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("F1-score: ", f1_score(y_test, y_pred, zero_division=0))

# Neural Networks (MLP)
mlp_model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=400)

mlp_model.fit(X_train_balanced, y_train_balanced)

y_pred = mlp_model.predict(X_test)

# Αξιολόγηση MLP
print("\n--- Αξιολόγηση MLP ---")
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred, zero_division=0))
print("Recall: ", recall_score(y_test, y_pred, zero_division=0))
print("F1-score: ", f1_score(y_test, y_pred, zero_division=0))

# Αξιολόγηση και αρχείο εξόδου
def evaluate_model(clf, X, y, set_name, clf_name):
    y_pred = clf.predict(X)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    accuracy = accuracy_score(y, y_pred)

    return {
        "Classifier": clf_name,
        "Train/Test Set": set_name,
        "Samples Count": len(y),
        "TP (True Positives)": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy
    }


models = [
    ("LDA", LinearDiscriminantAnalysis()),
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
    ("KNN (k=7)", KNeighborsClassifier(n_neighbors=7)),
    ("KNN (k=9)", KNeighborsClassifier(n_neighbors=9)),
    ("GaussianNB", GaussianNB()),
    ("SVM (RBF)", SVC(kernel="rbf", probability=True)),
    ("MLP (64,32 nodes)", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=400))
]

results = []


for name, model in models:
    model.fit(X_train_balanced, y_train_balanced)

    res_train = evaluate_model(model, X_train_balanced, y_train_balanced, "train", name)
    results.append(res_train)

    res_test = evaluate_model(model, X_test, y_test, "test", name)
    results.append(res_test)


output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "classification_results.xlsx")

df_results = pd.DataFrame(results)
df_results.to_excel(output_path, index=False)

print("Αρχείο δημιουργήθηκε στο:", output_path)
