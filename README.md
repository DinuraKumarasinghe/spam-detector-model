

### Code Breakdown of spam detector 

1. **Import Libraries**
   - Libraries for data manipulation, natural language processing, machine learning, and visualization are imported.

2. **Load and Preprocess Data**
   - **`df = pd.read_csv('spam.csv', encoding='latin-1')`**: Loads the dataset.
   - **`df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)`**: Drops unnecessary columns.
   - **`df.rename(columns={'v1': 'class', 'v2': 'sms'}, inplace=True)`**: Renames columns for clarity.
   - **`df.drop_duplicates(keep='first', inplace=True)`**: Removes duplicate rows.
   - **`df["length"] = df['sms'].apply(len)`**: Adds a column for the length of SMS messages.
   - **`df.hist(column='length', by='class', bins=20, figsize=(10, 5))`**: Plots the histogram of SMS lengths by class.

3. **Text Preprocessing**
   - **`nltk.download('stopwords')` and `nltk.download('punkt')`**: Download NLTK resources for text processing.
   - **`pt = PorterStemmer()`**: Initializes the Porter Stemmer.
   - **`def clean(text): ...`**: Defines a function to preprocess text by converting to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
   - **`df['sms_cleaned'] = df['sms'].apply(clean)`**: Applies the cleaning function to the SMS messages.

4. **Feature Extraction and Model Training**
   - **`tvec = TfidfVectorizer(max_features=3000)`**: Initializes TF-IDF vectorizer.
   - **`x = tvec.fit_transform(df['sms_cleaned']).toarray()`**: Vectorizes the cleaned SMS messages.
   - **`y = df['class'].values`**: Defines the target variable.
   - **`x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)`**: Splits the data into training and testing sets.
   - **`model = MultinomialNB()`**: Initializes Naive Bayes classifier.
   - **`model.fit(x_train, y_train)`**: Trains the model.
   - **`y_pred = model.predict(x_test)`**: Makes predictions on the test set.

5. **Model Evaluation and Visualization**
   - **`print("Accuracy:", accuracy_score(y_test, y_pred))`**: Prints the accuracy of the model.
   - **`print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))`**: Prints the confusion matrix.
   - **`sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])`**: Plots the confusion matrix as a heatmap.
   - **`report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], output_dict=True)`**: Generates a classification report.
   - **`report_df = pd.DataFrame(report).transpose()` and `report_df.iloc[:-1, :-1].plot(kind='bar', figsize=(10, 5))`**: Creates and plots the classification report metrics.
   - **`lb = LabelBinarizer()`**: Binarizes the output for ROC curve calculation.
   - **`fpr, tpr, _ = roc_curve(y_test_bin, y_pred_bin)`**: Computes the ROC curve.
   - **`roc_auc = auc(fpr, tpr)`**: Calculates the AUC.
   - **`plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)`**: Plots the ROC curve.

This code performs data preprocessing, trains a Naive Bayes classifier, and evaluates the model using several visualizations to understand its performance comprehensively.


