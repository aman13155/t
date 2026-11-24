#importing required libraries

from flask import Flask, request, render_template, flash, redirect, url_for
import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import os
import gradio as gr
import requests
from textblob import TextBlob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.layers import TextVectorization 
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from flask import Flask, request, render_template
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model
import pickle
import csv
import cv2
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.secret_key = os.environ.get('FLASK_SECRET_KEY')

if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY environment variable is missing")

# Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
# Load Tesseract OCR

# Load the saved vocabulary for text vectorization
with open("vectorizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Rebuild the vectorizer and set its vocabulary
loaded_vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=1800, output_mode='int')
loaded_vectorizer.set_vocabulary(vocab)

# Load the trained model
loaded_model = load_model("youtube_comment_classification_model.h5")

# Define toxicity categories
categories = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if (
            username == os.environ.get("ADMIN_USERNAME")
            and password == os.environ.get("ADMIN_PASSWORD")
        ):
            flash("Welcome!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

    return render_template("login.html")

@app.route("/logout")
def logout():
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route('/home')
def home():
    return render_template('index.html')
    # User is not loggedin redirect to login page

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/category')
def category():
    return render_template('category.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/imageupload')
def imageupload():
    return render_template('imageupload.html')



def predict_toxicity(comment):
    """Predict toxicity of extracted text."""
    vectorized_comment = loaded_vectorizer([comment])
    vectorized_comment = tf.convert_to_tensor(vectorized_comment)
    vectorized_comment = tf.reshape(vectorized_comment, (1, 1800))
    
    prediction = loaded_model.predict(vectorized_comment)[0]
    result_dict = {categories[i]: round(float(prediction[i]), 4) for i in range(len(categories))}
    return result_dict


def get_comments(video_id1):
    api_service_name = "youtube"
    api_version = "v3"

    # Read API key from environment variable
    developer_key = os.environ.get("YOUTUBE_API_KEY")

    if not developer_key:
        raise RuntimeError("YOUTUBE_API_KEY environment variable is missing")

    youtube = build(
        api_service_name,
        api_version,
        developerKey=developer_key
    )

    video_id = video_id1
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token
        )

        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return comments

from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    try:
        query = urlparse(url)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        elif query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                return parse_qs(query.query)['v'][0]
        return url  # fallback, if already an ID
    except Exception:
        return url




@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        # 1. Get YouTube URL
        input_data = request.form.get("url", "").strip()

        if not input_data:
            flash("Please enter a YouTube video URL.")
            return redirect(url_for("predict"))

        # 2. Extract video ID
        video_id = extract_video_id(input_data)

        if not video_id:
            flash("Invalid YouTube video URL.")
            return redirect(url_for("predict"))

        # 3. Fetch comments using YouTube Data API
        try:
            comments = get_comments(video_id)

        except HttpError as e:
            print("YouTube API error:", e)
            flash("Unable to fetch YouTube comments. Check your API key, quota and video permissions.")
            return redirect(url_for("predict"))

        except Exception as e:
            print("Comment fetching error:", e)
            flash("An error occurred while fetching comments.")
            return redirect(url_for("predict"))

        if not comments:
            flash("No comments found for this video.")
            return redirect(url_for("predict"))

        df = pd.DataFrame(comments, columns=["comment"])

        # 4. Toxicity analysis configuration
        TOXIC_THRESHOLD = 0.50

        selected_cats = [
            "Severe Toxic",
            "Obscene",
            "Threat",
            "Insult",
            "Identity Hate"
        ]

        results = []
        total_count = 0

        cat_counts = {cat: 0 for cat in categories}
        selected_overall_count = 0

        # 5. Process each comment using trained TensorFlow model
        for comment in df["comment"]:

            if not isinstance(comment, str) or not comment.strip():
                continue

            total_count += 1

            vectorized_comment = loaded_vectorizer([comment])
            vectorized_comment = tf.convert_to_tensor(vectorized_comment)
            vectorized_comment = tf.reshape(
                vectorized_comment, (1, 1800)
            )

            prediction = loaded_model.predict(
                vectorized_comment, verbose=0
            )[0]

            scores = {
                categories[i]: round(float(prediction[i]), 2)
                for i in range(len(categories))
            }

            # Category-wise toxicity counts
            for cat, score in scores.items():
                if score >= TOXIC_THRESHOLD:
                    cat_counts[cat] += 1

            # Overall toxicity count
            if any(
                scores[cat] >= TOXIC_THRESHOLD
                for cat in selected_cats
            ):
                selected_overall_count += 1

            # Sentiment analysis
            polarity = TextBlob(comment).sentiment.polarity

            sentiment = (
                "Positive" if polarity > 0
                else "Negative" if polarity < 0
                else "Neutral"
            )

            results.append({
                "comment": comment,
                "toxicity": scores,
                "sentiment": sentiment
            })

        # 6. Calculate percentages
        if total_count > 0:

            cat_percentages = {
                cat: round(
                    (cat_counts[cat] / total_count) * 100, 2
                )
                for cat in categories
            }

            overall_percentage = round(
                (selected_overall_count / total_count) * 100, 2
            )

        else:
            cat_percentages = {
                cat: 0.0 for cat in categories
            }

            overall_percentage = 0.0

        # 7. Display results
        return render_template(
            "contact.html",
            results=results,
            res=1,
            youtube_url=input_data,
            total_count=total_count,
            toxic_threshold=TOXIC_THRESHOLD,
            selected_cats=selected_cats,
            cat_counts=cat_counts,
            cat_percentages=cat_percentages,
            overall_count=selected_overall_count,
            overall_percentage=overall_percentage
        )

    return render_template("contact.html", res=0)





@app.route('/upload/csv', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return "No file uploaded"

        # Read the uploaded CSV file
        data = pd.read_csv(file)

        # Remove extra spaces from column names
        data.columns = data.columns.str.strip()

        # Display available columns
        print("CSV columns:", data.columns.tolist())

        # Find comment column (case-insensitive)
        comment_column = next(
            (col for col in data.columns
             if col.lower() == 'comment'),
            None
        )

        if comment_column is None:
            return "CSV file must contain a 'comment' column."

        comments = data[comment_column].dropna().astype(str).tolist()

        # Classify each comment using TextBlob
        sentiments = []

        for comment in comments:
            blob = TextBlob(comment)
            polarity = blob.sentiment.polarity

            if polarity > 0:
                sentiment = 'Positive'
            elif polarity < 0:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'

            sentiments.append((comment, sentiment))

        return render_template('upload.html', comments=sentiments)

    return render_template('upload.html')





@app.route("/upload1", methods=["GET", "POST"])
def upload1():
    if request.method == "POST":
        file = request.files.get('file')

        if not file or file.filename == '':
            return render_template(
                "SocialMedia.html",
                error="No file uploaded",
                comments=[]
            )

        if not file.filename.lower().endswith('.csv'):
            return render_template(
                "SocialMedia.html",
                error="File must be a .csv",
                comments=[]
            )

        # Read CSV file
        df = pd.read_csv(file)

        # Clean column names
        df.columns = df.columns.astype(str).str.strip()

        print("CSV columns:", df.columns.tolist())

        # Find comment column (case-insensitive)
        comment_column = next(
            (col for col in df.columns if col.lower() == "comment"),
            None
        )

        if comment_column is None:
            return render_template(
                "SocialMedia.html",
                error=f"CSV must contain a comment column. Found: {df.columns.tolist()}",
                comments=[]
            )

        toxic_comments = []

        for comment in df[comment_column]:
            if not isinstance(comment, str) or comment.strip() == "":
                continue

            # Vectorize comment
            vectorized_comment = loaded_vectorizer([comment])
            vectorized_comment = tf.convert_to_tensor(vectorized_comment)
            vectorized_comment = tf.reshape(vectorized_comment, (1, 1800))

            # Predict toxicity
            prediction = loaded_model.predict(
                vectorized_comment,
                verbose=0
            )[0]

            # Check toxicity threshold
            if any(p > 0.5 for p in prediction):
                scored = {
                    categories[i]: round(float(prediction[i]), 2)
                    for i in range(len(categories))
                }

                toxic_comments.append((comment, scored))

        return render_template(
            "SocialMedia.html",
            comments=toxic_comments,
            res=1
        )

    return render_template("SocialMedia.html", comments=[], res=0)



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)