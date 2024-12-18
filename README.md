﻿### README: Image Similarity Search Project

---

#### **Project Structure**
1. **`app.py`**  
   - The main Flask application that serves as the backend.
   - Handles image uploads, computes query descriptors, and retrieves similar images from the dataset.

2. **`image_utils.py`**  
   - Contains functions to calculate visual descriptors for the query image, including:  
     - Color histograms  
     - Dominant colors  
     - Gabor descriptors  
     - Hu moments  
     - Texture energy  
     - Circularity

3. **`similarity.py`**  
   - Provides functionality to compute similarity scores between the query image's descriptors and dataset images.  
   - Lower similarity scores indicate closer matches.

4. **`script.py`**  
   - A one-time script to load image descriptors from a JSON file into a MongoDB collection.
   - **Note**: Run this script before using the application.  
     - Download the JSON file from Google Drive:  
       [Download JSON file](https://drive.google.com/file/d/16t4xucfzeiq3R8e2fl0_Y2fP118smp7g/view?usp=sharing)

5. **Frontend**  
   - `templates/index.html`: Allows users to upload an image.  
   - `templates/results.html`: Displays the query image and top similar images from the dataset.

6. **Dataset**  
   - Stored in `static/dataset/` with subfolders:  
     - `aGrass`, `bField`, `cIndustry`, ... , `gParking`

---

#### **Setup and Usage**


1. **Set Up MongoDB**  
   - Ensure MongoDB is installed and running on your system.  
   - The application connects to MongoDB at `mongodb://localhost:27017/`. Modify `app.py` if your configuration is different.

2. **Populate MongoDB**  
   - Run the `script` script to load image descriptors from the JSON file into MongoDB:  
     ```bash
     python script
     ```

3. **Run the Flask Application**  
   Start the Flask server:
   ```bash
   python app.py
   ```
   The application will be accessible at `http://127.0.0.1:5000`.

5. **Using the Application**  
   - Upload an image on the homepage (`index.html`).  
   - View the query image and top similar images on the results page (`results.html`).

---

#### **Folder Structure**
```
project/
│
├── app.py               # Main Flask app
├── image_utils.py       # Descriptor calculation functions
├── similarity.py        # Similarity computation 
├── relevance_feedback.py # relevance feedback computation                     
├── static/
│   ├── dataset/
│   │   ├── aGrass/
│   │   ├── bField/
│   │   ├── cIndustry/
│   │   └── ...          # Other categories
│   └── ...              # Other static assets
├── templates/
│   ├── index.html       # Frontend: upload page
│   ├── results.html     # Frontend: results page
```


---
