#With Mood Mapping.

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from urllib.parse import quote

class MusicRecommendationSystem:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, encoding='latin1')
        self.data_ingestion_and_preprocessing()
        
    def data_ingestion_and_preprocessing(self):
        # Check for missing values
        missing_values = self.data.isnull().sum()
        print("Missing values per column:\n", missing_values)
        
        # Fill or drop missing values as necessary
        self.data = self.data.dropna()  # Dropping rows with missing values for simplicity
        
        # Check for duplicate records
        duplicates = self.data.duplicated().sum()
        print("Number of duplicate records:", duplicates)
        
        # Drop duplicate records if any
        self.data = self.data.drop_duplicates()

        # Map moods to the dataset
        self.data['mood'] = self.data.apply(self.map_mood, axis=1)
        
        # Generate links dynamically
        base_url = "https://www.youtube.com/results?search_query="
        self.data['link'] = self.data['name'].apply(lambda x: base_url + quote(x))

        # Encode the target labels (assuming the target column is 'mood')
        self.target = self.data['mood']
        self.encoder = LabelEncoder()
        self.encoded_target = self.encoder.fit_transform(self.target)

        # Select features for training by excluding the specified columns
        columns_to_exclude = ['name', 'album', 'artist', 'release_date', 'mood', 'link']
        self.features = self.data.drop(columns=columns_to_exclude)

        # Standardize the features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)

        # Convert scaled features back to DataFrame for easier handling
        self.scaled_features_df = pd.DataFrame(self.scaled_features, columns=self.features.columns)
        
        # Check if models are already trained
        if not hasattr(self, 'knn_classifier'):
            self.train_knn_model()
        if not hasattr(self, 'nn_model'):
            self.train_nn_model()

    def map_mood(self, row):
        # Example mapping based on energy and valence
        if row['energy'] > 0.7 and row['valence'] > 0.7:
            return 'energetic'
        elif row['energy'] < 0.3 and row['valence'] < 0.3:
            return 'sad'
        elif row['energy'] < 0.3 and row['valence'] > 0.7:
            return 'calm'
        else:
            return 'happy'

    def train_knn_model(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_features_df, self.encoded_target, test_size=0.2, random_state=42)
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize the KNN classifier
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        
        # Train the model
        self.knn_classifier.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = self.knn_classifier.predict(X_test)
        
        # Evaluate the model
        self.accuracy = accuracy_score(y_test, y_pred)
        print("KNN Model Accuracy:", self.accuracy)

    def train_nn_model(self):
        # Select relevant features for recommendation
        features = ['popularity', 'length', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'valence', 'loudness', 'speechiness', 'tempo']
        X = self.scaler.fit_transform(self.data[features])
        
        # Train the k-NN model
        self.nn_model = NearestNeighbors(n_neighbors=len(self.data), algorithm='ball_tree')
        self.nn_model.fit(X)
        self.X_nn = X
    
    def recommend_songs_by_emotion(self, emotion, num_recommendations=5):
        # Check if the emotion exists in the dataset
        if emotion not in self.encoder.classes_:
            return f"Emotion '{emotion}' not found in the dataset."
        
        # Find the encoded value for the given emotion
        emotion_code = self.encoder.transform([emotion])[0]
        
        # Find the indices of the songs with the same emotion in the test set
        emotion_indices = [i for i, e in enumerate(self.y_test) if e == emotion_code]
        
        # If there are fewer recommendations available than requested, adjust the number
        num_recommendations = min(num_recommendations, len(emotion_indices))
        
        # Get the indices of the recommended songs
        recommended_indices = emotion_indices[:num_recommendations]
        
        # Get the recommended songs from the original dataset
        recommendations = self.data.iloc[recommended_indices][['name', 'artist', 'link']]
        return recommendations
    
    def recommend_songs_by_rating(self, song_name, rating):
        # Check if the song exists in the dataset
        if song_name not in self.data['name'].values:
            return f"Song '{song_name}' not found in the dataset."
        
        # Find the index of the selected song
        song_idx = self.data[self.data['name'] == song_name].index[0]
        
        # Get the features of the selected song
        song_features = self.X_nn[song_idx].reshape(1, -1)
        
        # Find similar songs using k-NN
        distances, indices = self.nn_model.kneighbors(song_features)
        
        # Determine the range of indices to select based on rating
        if rating in [4, 5]:
            recommended_indices = indices[0][1:6]  # Recommend 5 most similar songs
        elif rating == 3:
            recommended_indices = indices[0][10:15]  # Recommend somewhat similar songs
        else:  # rating in [1, 2]
            recommended_indices = indices[0][-6:-1]  # Recommend 5 most dissimilar songs
        
        recommended_songs = self.data.iloc[recommended_indices][['name', 'artist', 'link']]
        return recommended_songs
    
    def sort_by_favorite(self):
        # Sort songs by 'favorite' column if it exists
        if 'favorite' in self.data.columns:
            return self.data.sort_values(by='favorite', ascending=False)
        else:
            return self.data
    
    def apply_user_rating(self, song_name, rating):
        # Apply user rating to the song
        if 'user_rating' not in self.data.columns:
            self.data['user_rating'] = 0
        self.data.loc[self.data['name'] == song_name, 'user_rating'] = rating

# Create an instance of the class and test the methods
music_rec_system = MusicRecommendationSystem('data_moods.csv')

# Function to get user input and recommend songs
def dynamic_recommendation(music_rec_system):
    # Test recommendation by emotion
    emotion = input("Enter an emotion (e.g., Calm, Happy, Sad, etc.): ")
    recommended_songs_by_emotion = music_rec_system.recommend_songs_by_emotion(emotion)
    print("Recommended Songs for", emotion, ":\n", recommended_songs_by_emotion)
    
    # Test recommendation by rating
    song_name = input("Enter the name of the song you want to rate: ")
    rating = int(input("Enter your rating for the song (e.g., 1-5): "))
    recommended_songs_by_rating = music_rec_system.recommend_songs_by_rating(song_name, rating)
    print("Recommended Songs based on rating:\n", recommended_songs_by_rating)

# Assuming you have instantiated your music recommendation system as 'music_rec_system'
dynamic_recommendation(music_rec_system)
