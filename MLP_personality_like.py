import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Reading
user_data_path = './data/MNS_data_full.csv'
article_data_path = './data/articles_filtered.csv'
user_data = pd.read_csv(user_data_path)
article_data = pd.read_csv(article_data_path)
print("User Data:")
print(user_data.head())
print("\nArticle Data:")
print(article_data.head())

# Merge
merged_data = pd.merge(user_data, 
                       article_data[['id', 'reading_time','difficulty']],  
                       left_on='article_id', 
                       right_on='id', 
                       how='inner')

merged_data = merged_data.drop(columns=['id'])
print(merged_data.shape)
print(merged_data.head())
print(f"Total rows and columns in merged data: {merged_data.shape}")
print("Columns in merged data:", merged_data.columns)

features = ['Extraversion', 'Agreeableness', 'Conscientiousness', 
            'Neuroticism', 'OpennessToExperience', 'likability',"difficulty", 'reading_time']
merged_data = merged_data[features].dropna()  
print(merged_data.shape)

scaler = MinMaxScaler()
merged_data[['Extraversion', 'Agreeableness', 'Conscientiousness', 
             'Neuroticism', 'OpennessToExperience',"difficulty", 'reading_time']] = scaler.fit_transform(
    merged_data[['Extraversion', 'Agreeableness', 'Conscientiousness', 
                 'Neuroticism', 'OpennessToExperience',"difficulty", 'reading_time']])
print(merged_data.head)

personality_features = merged_data[['Extraversion', 'Agreeableness', 
                                    'Conscientiousness', 'Neuroticism', 
                                    'OpennessToExperience',"difficulty","reading_time"]].values
personality_features = personality_features.astype(float)
y_binary = (merged_data['likability'] > 2).astype(int)  
print("Binary target variable distribution:")
print(y_binary.value_counts())
print(personality_features)

X = np.hstack([personality_features])
y_binary = y_binary.astype(int)
print(y_binary)

#Model
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
y_train = y_train.astype(int)  

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(zip(np.unique(y_train), class_weights))
print("Corrected Class Weights:", class_weights)
print("Unique values in y_train:", np.unique(y_train))
print("Class Weights Keys:", class_weights.keys())

model = Sequential([
    Input(shape=(X.shape[1],)),  
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer=Adam(learning_rate=0.000008), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=10,         
    restore_best_weights=True  
)

history = model.fit(
    X_train, 
    y_train, 
    validation_data=(X_test, y_test), 
    epochs=100, 
    batch_size=32, 
    class_weight=class_weights,  
    callbacks=[early_stopping]  
)

y_train_pred = model.predict(X_train).flatten()
y_test_pred = model.predict(X_test).flatten()

y_train_pred_labels = (y_train_pred > 0.5).astype(int)
y_test_pred_labels = (y_test_pred > 0.5).astype(int)

train_accuracy = accuracy_score(y_train, y_train_pred_labels)
test_accuracy = accuracy_score(y_test, y_test_pred_labels)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print("Classification Report on Test Set:")
print(classification_report(y_test, y_test_pred_labels))

#Visualize
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

#Evaluate
feature_names = ['Extraversion', 'Agreeableness', 'Conscientiousness', 
                 'Neuroticism', 'OpennessToExperience', "difficulty", 'reading_time']
base_loss, base_acc = model.evaluate(X_test, y_test, verbose=0)

print(f'Baseline Test Loss: {base_loss:.4f}, Baseline Test Accuracy: {base_acc:.4f}')

feature_importance = {}

for i, feature in enumerate(feature_names):
    X_test_shuffled = X_test.copy()
    np.random.shuffle(X_test_shuffled[:, i])  
    
    shuffled_loss, shuffled_acc = model.evaluate(X_test_shuffled, y_test, verbose=0)
    acc_decrease = base_acc - shuffled_acc
    loss_increase = shuffled_loss - base_loss
    
    feature_importance[feature] = acc_decrease
    print(f'Feature: {feature}, Accuracy Decrease: {acc_decrease:.4f}, Loss Increase: {loss_increase:.4f}')

sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance (based on accuracy decrease):")
for feature, importance in sorted_importance:
    print(f"{feature}: {importance:.4f}")