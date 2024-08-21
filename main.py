import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Drop unnecessary columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Rename columns
df.rename(columns={'v1': 'class', 'v2': 'sms'}, inplace=True)

# Drop duplicates
df.drop_duplicates(keep='first', inplace=True)

# Add a column for the length of the SMS messages
df["length"] = df['sms'].apply(len)

# Plot histogram of SMS lengths by class
df.hist(column='length', by='class', bins=20, figsize=(10, 5))

# Preprocessing: Define a function to clean and preprocess text
nltk.download('stopwords')
nltk.download('punkt')

pt = PorterStemmer()

def clean(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Stem the words
    text = [pt.stem(i) for i in text]

    return " ".join(text)

# Apply cleaning function to the SMS messages
df['sms_cleaned'] = df['sms'].apply(clean)

# Vectorize the cleaned SMS messages
tvec = TfidfVectorizer(max_features=3000)
x = tvec.fit_transform(df['sms_cleaned']).toarray()

# Define the target variable
y = df['class'].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], output_dict=True)

# Convert to DataFrame for easier plotting
report_df = pd.DataFrame(report).transpose()

# Plot Precision, Recall, and F1-Score
report_df.iloc[:-1, :-1].plot(kind='bar', figsize=(10, 5))
plt.title('Classification Report Metrics')
plt.ylabel('Score')
plt.show()

# Binarize the output for ROC curve
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_pred_bin = lb.transform(y_pred)

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test_bin, y_pred_bin)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
