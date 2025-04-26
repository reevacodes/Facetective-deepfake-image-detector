🕵️‍♂️ Facetective - Deepfake Image Detector
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
Detect fake vs real images with the power of deep learning!

Facetective is a lightweight deepfake image detector that classifies images as either real or fake using a custom-trained deep learning model.
Built with ❤️ for curious minds, students, and researchers to fight misinformation and digital fraud.

🚀 Features
> Classifies images as Real or Fake.
> Trained on custom datasets of real and AI-generated faces.
> Lightweight .h5 model ready for deployment.
> Clean and modular codebase (train_model.py, app.py).
> Streamlit (or Flask-ready) app for easy UI integration.
> Large files handled efficiently with Git LFS.

🗂️ Project Structure
arduino
Copy code
Facetective-deepfake-image-detector/
├── Data/
│   ├── fake/
│   └── real/
├── data_split/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   └── deepfake_image_model.h5
├── src/
│   └── train_model.py
├── app.py
├── best_model.h5
├── requirements.txt
├── .gitignore
├── .gitattributes

🔧 Setup Instructions

> Clone the repository:
git clone https://github.com/reevacodes/Facetective-deepfake-image-detector.git
cd Facetective-deepfake-image-detector

> (Recommended) Create a virtual environment:
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/Mac

> Install dependencies:
pip install -r requirements.txt

> Run the app:
python app.py
(or if using Streamlit UI: streamlit run app.py)

🧠 Model Details
> Model: Convolutional Neural Network (CNN)
> Format: .h5 (Keras/TensorFlow SavedModel)
> Training data: Real human faces vs Fake AI-generated faces (deepfakes)
> Loss function: Binary Crossentropy

✨ Future Improvements
> Add webcam real-time detection.
> Improve accuracy with larger datasets.
> Build a mobile version (Android/iOS).
> Deploy app online using Streamlit Sharing, Heroku, or Vercel.

🛡️ License
This project is licensed under the MIT License.
 
🤝 Connect with Me
GitHub: reevacodes
LinkedIn:(https://www.linkedin.com/in/reeva-gupta-72877b258/)
