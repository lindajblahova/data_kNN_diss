import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# Function to plot line plots
def plot_line_plots(data, diagnosis, title):
    colors = {'neg': 'blue', 'pos': 'none' if diagnosis == 'neg' else 'red'}

    for index, row in data[data['diagnosis'] == diagnosis].iterrows():
        plt.plot(row[1:], label=f'Row {index + 1}', color=colors[row['diagnosis']])

    plt.xlabel('Feature')
    plt.ylabel('Feature Value')
    every_30th_indices = np.arange(29, X.shape[1], 30)
    plt.xticks(every_30th_indices, feature_names[every_30th_indices], rotation=90)
    plt.title(f'Line Plot of \'{diagnosis}\' measurements' + (' (Scaled)' if 'Scaled' in title else ''))
    plt.show()


# Strings to replace
strings_to_replace = ['CaE', 'caE', 'iné', 'cysta', 'endom', 'col', 'U']

# Read the CSV file into a DataFrame
filename = 'DATAPhD.csv'
df = pd.read_csv(filename, sep=';')

# Convert ',' to '.' in numeric columns
df.iloc[:, 1:] = df.iloc[:, 1:].map(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

# Replace specified strings in the first column
df['diagnosis'] = df['diagnosis'].replace(strings_to_replace, 'pos', regex=True)

# Write the modified DataFrame back to the CSV file
filename = 'DATAPhD_bin.csv'
df.to_csv(filename, index=False)

# Load binary data
filename = "DATAPhD_bin.csv"
df = pd.read_csv(filename, sep=',')
class_labels = df['diagnosis'].unique()

# Drop rows with missing values
df = df.dropna()

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

feature_names = X.columns

# Plot line plots for 'neg' and 'pos'
for diagnosis in ['neg', 'pos']:
    plot_line_plots(df, diagnosis, '')

# Initialize the RobustScaler
scaler = RobustScaler()

# Initialize the RandomForestClassifier for feature selection
rf_clf = RandomForestClassifier(n_estimators=100)

# Initialize the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Initialize the variables
num_replications = 5
accuracies = []
conf_matrices = []
feature_importance_sum = np.zeros(X.shape[1])

# Replications
for replication in range(num_replications):

    # Split the dataset to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Standardize the dataset using scaling to interquartile range
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Combine the dataset back (only for plotting)
    train_df = pd.concat(
        [pd.DataFrame(y_train, columns=['diagnosis']), pd.DataFrame(X_train_scaled, columns=X.columns)], axis=1)
    test_df = pd.concat([pd.DataFrame(y_test, columns=['diagnosis']), pd.DataFrame(X_test_scaled, columns=X.columns)],
                        axis=1)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    if replication == 0:  # Plot only in the first iteration

        # Plot the data by diagnose pos/neg
        for diagnosis in ['neg', 'pos']:
            plot_line_plots(combined_df[combined_df['diagnosis'] == diagnosis], diagnosis, ' (Scaled)')

    # Use random forest to calculate the feature importance
    rf_clf.fit(X_train_scaled, y_train)
    feature_importance_sum += rf_clf.feature_importances_

    # Select 30 best features
    sfm = SelectFromModel(rf_clf, threshold=-np.inf, max_features=30)
    sfm.fit(X_train_scaled, y_train)

    # Transform the selected data
    X_train_selected = sfm.transform(X_train_scaled)
    X_test_selected = sfm.transform(X_test_scaled)

    # Fit and predict the y on test data
    knn.fit(X_train_selected, y_train)
    y_pred_selected = knn.predict(X_test_selected)

    # Save the accuracy
    accuracy_selected = accuracy_score(y_test, y_pred_selected)
    accuracies.append(accuracy_selected)

    # Save the confusion matrix
    conf_matrix_selected = confusion_matrix(y_test, y_pred_selected)
    conf_matrices.append(conf_matrix_selected)

# Calculate the feature importance over the replications for every feature
average_feature_importance = feature_importance_sum / num_replications

# Plot the feature importance (red ae the 30 most important ones)
plt.figure(figsize=(10, 6))
plt.title("Average Feature Importance")
plt.bar(range(X.shape[1]), average_feature_importance, align="center")
plt.xlabel("Feature")
plt.ylabel("Average Importance")
top_30_indices = np.argsort(average_feature_importance)[::-1][:30]
plt.bar(top_30_indices, average_feature_importance[top_30_indices], color="red", label="Top 30")
every_30th_indices = np.arange(29, X_train.shape[1], 30)
plt.xticks(every_30th_indices, feature_names[every_30th_indices], rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Calculate and plot average accuracy
average_accuracy = np.mean(accuracies)
print(f'Average Accuracy over {num_replications} replications: {average_accuracy * 100:.2f}%')
print(accuracies)

# Calculate and plot average confusion matrix as heatmap
average_conf_matrix = np.mean(conf_matrices, axis=0)
print('\nAverage Confusion Matrix:')
print(average_conf_matrix)
plt.figure(figsize=(6, 4))
annot_kws = {"size": 18}
sns.heatmap(average_conf_matrix, annot=True, fmt='g', cmap='flare', cbar=False, annot_kws=annot_kws,
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Average Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()