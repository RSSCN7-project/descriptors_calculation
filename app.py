import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import pymongo
import cv2
import numpy as np
import json
from image_utils import (
    calculate_color_histogram, 
    find_dominant_colors, 
    calculate_gabor_descriptors, 
    calculate_hu_moments, 
    calculate_texture_energy, 
    calculate_circularity
)
from similarity import compute_similarity_score
from relevance_feedback import RelevanceFeedbackManager

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB Connection
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['image_database']
descriptors_collection = db['image_descriptors2']

# Relevance Feedback Manager
feedback_manager = RelevanceFeedbackManager()

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    # Parse incoming feedback data
    feedback_data = request.json
    
    # Enrich feedback items with descriptors from the database
    for item in feedback_data['feedback_items']:
        # Find the matching document in the database
        doc = descriptors_collection.find_one({
            'image_name': item['image_name'], 
            'category': item['category']
        })
        
        if doc:
            # Populate descriptors from the database document
            item['descriptors'] = {
                "histogram": doc.get('histogram', []),
                "dominant_colors": doc.get('dominant_colors', []),
                "gabor_descriptors": doc.get('gabor_descriptors', []),
                "hu_moments": doc.get('hu_moments', []),
                "texture_energy": doc.get('texture_energy', []),
                "circularity": doc.get('circularity', [])
            }
    
    # Process feedback and get updated weights
    try:
        new_weights = feedback_manager.update_weights(
            query_descriptors=feedback_data['query_descriptors'], 
            feedback_data=feedback_data['feedback_items']
        )
        
        # Find similar images with the new weights
        matches = find_similar_images(
            query_descriptors=feedback_data['query_descriptors'], 
            weights=new_weights
        )
        
        # Persist the new weights
        feedback_manager.save_feedback_history()
        
        return jsonify({
            "status": "success", 
            "new_weights": new_weights,
            "similar_images": matches
        })
    
    except Exception as e:
        print(f"Error processing feedback: {e}")
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/refresh_results', methods=['POST'])
def refresh_results():
    # Get data from request
    data = request.json
    query_descriptors = data.get('query_descriptors')
    weights = data.get('weights')

    if not query_descriptors:
        return jsonify({"status": "error", "message": "No query descriptors provided"}), 400

    # Find similar images with the provided or default weights
    matches = find_similar_images(
        query_descriptors=query_descriptors, 
        weights=weights
    )
    
    return jsonify({
        "status": "success",
        "similar_images": matches
    })

def find_similar_images(query_descriptors, top_k=10, weights=None):
    # If no specific weights provided, use default
    if weights is None:
        weights = feedback_manager.current_weights
    
    # Fetch all descriptors from MongoDB
    all_descriptors = list(descriptors_collection.find())
    similarities = []

    for doc in all_descriptors:
        try:
            similarity_score = compute_similarity_score(
                query_descriptors,
                {
                    "histogram": doc.get('histogram', []),
                    "dominant_colors": doc.get('dominant_colors', []),
                    "gabor_descriptors": doc.get('gabor_descriptors', []),
                    "hu_moments": doc.get('hu_moments', []),
                    "texture_energy": doc.get('texture_energy', []),
                    "circularity": doc.get('circularity', []),
                },
                weights  # Pass custom weights
            )
            similarities.append({
                'category': doc['category'],
                'image_name': doc['image_name'],
                'similarity_score': similarity_score
            })
        except Exception as e:
            print(f"Error processing descriptor for {doc.get('image_name')}: {e}")

    # Sort by similarity score (lower score means more similar)
    similarities = sorted(similarities, key=lambda x: x['similarity_score'])[:top_k]

    # Resolve local file paths for each similar image
    for sim in similarities:
        sim['image_path'] = f"/static/dataset/{sim['category']}/{sim['image_name']}"
        
    return similarities

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
            
            # Function to convert ndarray to list
            def convert_ndarray_to_list(data):
                if isinstance(data, np.ndarray):
                    return data.tolist()
                return data

            # Compute descriptors for uploaded image
            query_descriptors = {
                "histogram": convert_ndarray_to_list(calculate_color_histogram(filepath)),
                "dominant_colors": convert_ndarray_to_list(find_dominant_colors(filepath)),
                "gabor_descriptors": convert_ndarray_to_list(calculate_gabor_descriptors(filepath)),
                "hu_moments": convert_ndarray_to_list(calculate_hu_moments(filepath)),
                "texture_energy": convert_ndarray_to_list(calculate_texture_energy(filepath)),
                "circularity": convert_ndarray_to_list(calculate_circularity(filepath))
            }
            
            # Find similar images
            matches = find_similar_images(query_descriptors)
            print("Matched Images:", matches)  # Debugging
            return render_template('results.html', 
                       query_image=filename, 
                       similar_images=matches,
                       query_descriptors=json.dumps(query_descriptors))  # Pass as JSON string

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)