# 🌱 Crop Recommendation System

🔍 Predict the best crop to grow based on real-time soil data — using machine learning and an intuitive web interface.

## 🧠 Live Demo
 [Take a look](https://crop-prediction-webapp.onrender.com/)
---

## 💡 Features

- ⚡ **Instant prediction of crops**: brinjal, cauliflower, cabbage, cucurbits, etc.
- 🤖 Trained on soil nutrient data using a Random Forest Classifier
- 🎯 Clean Tailwind-styled UI for easy input
- 🧪 Easily extensible with new crops or models

---

## 🛠️ Tech Stack

- **Frontend**: HTML, Tailwind CSS, Jinja2
- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, pandas, LabelEncoder
- **Model Storage**: `pickle` (`model.pkl`, `label_encoder.pkl`)

---

## 📁 Project Structure


crop-predictor/
├── app.py                # Flask app
├── train_model.py        # ML training script
├── crop_data.csv         # Dataset
├── model.pkl             # Trained ML model
├── label_encoder.pkl     # Label encoder for crop labels
├── templates/
│   └── index.html        # Tailwind form UI
├── static/               # Optional: CSS/images
├── requirements.txt
└── README.md

---

## 🚀 Getting Started

Follow these steps to run the project locally:

### 1. Clone the Repository

```
git clone https://github.com/ashika0124/Crop_prediction_webapp.git
cd crop-predictor
```
### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows

```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
If requirements.txt is missing, run:
```
pip install flask pandas scikit-learn
```
### 4. Train the Model

```bash
python train_model.py
```
### 5. Run the App

```bash
python app.py
```
Visit in browser: http://127.0.0.1:5000


📊 Sample crop_data.csv
csv
```
Nitrogen,Phosphorus,Potassium,pH,Humidity,Salinity,Crop
85,40,50,6.5,70,0.4,brinjal
90,35,45,6.4,72,0.5,brinjal
60,25,30,6.2,65,0.6,cauliflower
```
✅ Make sure all values are numeric (not ranges like 50–70).

📄 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Built with ❤️ by Ashika
🔗 https://www.linkedin.com/in/ashika0124/ 
📧 ashika8482@gmail.com
