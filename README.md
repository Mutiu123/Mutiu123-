# Fashion Recommendation System: A Hybrid CNN-KNN Approach

## Project Description
Fashion recommendation systems enhance the shopping experience by suggesting stylish items to users. In this project, we've developed a hybrid recommendation system that combines Convolutional Neural Networks (CNNs) and K-Nearest Neighbors (KNN) algorithms. Our goal is to recommend fashionable clothing, accessories, and footwear based on user preferences.

## Problem Statement
The challenge lies in creating a system that understands fashion images and provides personalized recommendations. Users expect accurate suggestions aligned with their unique styles. Our task is to build an efficient and accurate recommendation system.

## Methodology
1. **Data Collection**:
   - Gather a diverse dataset of fashion items (clothing, accessories, footwear).
   - Ensure uniform image formats (e.g., JPEG, PNG) and resolutions.

2. **Preprocessing**:
   - Standardize and enhance images:
     - Resize images.
     - Normalize pixel values.
     - Reduce noise.
   - Prepare data for feature extraction.

3. **Feature Extraction**:
   - Utilize a pre-trained CNN model (e.g., VGG16, ResNet, InceptionV3).
   - Transfer learning: Extract high-level features from fashion images.

4. **Similarity Measurement**:
   - Define a metric (e.g., cosine similarity, Euclidean distance) to quantify feature similarity between images.

5. **Ranking and Recommendation**:
   - Rank dataset images based on similarity to input image features.
   - Recommend the top N items with the highest similarity scores.

6. **System Implementation**:
   - Integrate preprocessing, feature extraction, similarity computation, and recommendation generation.
   - Deliver a user-friendly experience.

## Contributions
1. **Hybrid Approach**:
   - Our system uniquely combines CNNs and KNN for fashion recommendations.
   - Leveraging both image features (CNN) and user-item interactions (KNN) improves accuracy.

2. **Performance Metrics**:
   - Evaluate using accuracy, precision, recall, and F1-score.
   - Ensure robust performance.

3. **User Feedback and Iterations**:
   - Incorporate user input to enhance recommendations.
   - Iterate based on real-world usage.

4. **Scalability and Real-world Deployment**:
   - Address scalability challenges for large user bases.
   - Plan for deployment in online fashion stores or apps.

5. **Ethical Considerations**:
   - Acknowledge biases (gender, cultural) in recommendations.
   - Mitigate bias and handle privacy concerns.

6. **Future Work**:
   - Explore context-aware recommendations.
   - Handle cold start problem.
   - Integrate other recommendation algorithms (e.g., matrix factorization).

---

**Code Breakdown and Explanations**:

1. **Data Preparation and Feature Extraction**:
   - **Loading Images and Creating File Paths (`datanames`)**:
     - You start by listing the image files in the `'data'` directory.
     - The `os.listdir('data')` function retrieves all filenames in that directory.
     - Each filename is appended to the `datanames` list, creating a list of file paths.
   - **Pre-trained ResNet50 Model**:
     - You load the pre-trained ResNet50 model with weights pre-trained on ImageNet data.
     - By setting `model.trainable = False`, you freeze the weights to prevent further training.
   - **Adding GlobalMaxPool2D Layer**:
     - You create a new model by adding a `GlobalMaxPool2D` layer after the ResNet50 base.
     - This layer reduces the spatial dimensions of the extracted features, resulting in a fixed-size feature vector for each image.
   - **Image Loading and Feature Extraction**:
     - The code loads an image (`'1634.jpg'`) and preprocesses it.
     - The image is converted to an array, expanded to 4 dimensions, and preprocessed using `preprocess_input`.
     - The modified ResNet50 model predicts the features for this image.
     - The resulting feature vector is normalized using the L2 norm.

2. **Feature Extraction Function**:
   - The `extract_features_from_images(image_path, model)` function encapsulates the feature extraction process.
   - Given an image path and the modified ResNet50 model, it:
     - Loads the image.
     - Preprocesses the image.
     - Predicts the features using the model.
     - Normalizes the results.
   - The function returns the normalized feature vector.

3. **Extracting Features for All Images**:
   - You iterate through all image file paths in `datanames`.
   - For each image, you call the `extract_features_from_images` function to extract features.
   - The resulting features are stored in the `image_features` list.

4. **Saving Extracted Features and File Paths**:

## **System Demo:**

![The System Demo](https://github.com/Mutiu123/Mutiu123-/blob/main/demo/demo.png)

![The System Demo](https://github.com/Mutiu123/Mutiu123-/blob/main/demo/demo1.png)

![The System Demo](https://github.com/Mutiu123/Mutiu123-/blob/main/demo/demo2.png)


## **To run the model**
1. **Clone the Repository**:
   - First, clone the repository containing your movie recommendation system code to your local machine. You can do this using Git or by downloading the ZIP file from the repository.

2. **Install Dependencies**:
   - Open your terminal or command prompt and navigate to the project directory.
   - Install the necessary Python libraries mentioned in the `requirements.txt` file using the following command:
     ```
     pip install -r requirements.txt
     ```

3. **Run the Streamlit App**:
   - In the same terminal or command prompt, execute the following command to run the Streamlit app:
     ```
     streamlit run app.py
     ```
   - This will start the local development server, and you'll see a message indicating that the app is running.
   - Open your web browser and visit `http://localhost:8501` (or the URL provided in the terminal) to access the interactive web app.

4. **Select a Movie**:
   - On the Streamlit app, you'll find a search bar where you can either select a movie from the dropdown list or type the movie name.
   - Choose a movie for which you want to receive recommendations.

5. **View Recommendations**:
   - Click the "Show Recommendation" button.
   - The app will display recommended movies based on the selected input movie. You'll see movie posters and titles for the top recommendations.

6. **Explore Further**:
   - Feel free to explore other movies by selecting different titles or typing new ones.
   - The app dynamically updates recommendations based on your input.

