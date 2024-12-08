import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import pymongo
import cv2
import numpy as np
from image_utils import (
    calculate_color_histogram, 
    find_dominant_colors, 
    calculate_gabor_descriptors, 
    calculate_hu_moments, 
    calculate_texture_energy, 
    calculate_circularity
)
from similarity import compute_similarity_score

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB Connection
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['image_database']
descriptors_collection = db['image_descriptors2']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        # If file is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Compute descriptors for uploaded image
            query_descriptors = {
                "histogram": calculate_color_histogram(filepath),
                "dominant_colors": find_dominant_colors(filepath),
                "gabor_descriptors": calculate_gabor_descriptors(filepath),
                "hu_moments": calculate_hu_moments(filepath),
                "texture_energy": calculate_texture_energy(filepath),
                "circularity": calculate_circularity(filepath)
            }
            
            # Find similar images
            similar_images = find_similar_images(query_descriptors)
            
            # After finding matches
            matches = find_similar_images(query_descriptors)
            print("Matched Images:", matches)  # Debugging
            return render_template('results.html', 
                                   query_image=filename, 
                                   similar_images=similar_images)
    
    return render_template('index.html')

def find_similar_images(query_descriptors, top_k=10):
    # Fetch all descriptors from MongoDB
    all_descriptors = list(descriptors_collection.find())
    similarities = []

    for doc in all_descriptors:
        similarity_score = compute_similarity_score(
            query_descriptors,
            {
                "histogram": doc['histogram'],
                "dominant_colors": doc['dominant_colors'],
                "gabor_descriptors": doc['gabor_descriptors'],
                "hu_moments": doc['hu_moments'],
                "texture_energy": doc['texture_energy'],
                "circularity": doc['circularity'],
            }
        )
        similarities.append({
            'category': doc['category'],
            'image_name': doc['image_name'],
            'similarity_score': similarity_score
        })

    # Sort by similarity score and take the top K results
    similarities = sorted(similarities, key=lambda x: x['similarity_score'])[:top_k]

    # Resolve local file paths for each similar image
    base_dataset_folder = os.path.join('static', 'dataset')
    for sim in similarities:
        sim['image_path'] = f"/static/dataset/{sim['category']}/{sim['image_name']}"
        
    return similarities


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)