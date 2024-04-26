# End-to-End Machine Learning Project with AWS Cloud Server Deployment

## Project Description

This project is an end-to-end machine learning application deployed on an AWS cloud server. The application is designed to predict student performance based on various input features such as gender, race/ethnicity, parental level of education, lunch type, test preparation course, reading score, and writing score.

The machine learning model used for prediction is trained using the scikit-learn library and is deployed using Flask, a web framework for Python. The project includes a preprocessing pipeline for feature scaling and transformation.

## Application Overview

The application consists of a Flask web server that provides two main routes:

1. **Home Page (/)**: Renders an HTML template (`index.html`) to display the home page of the application.
   
2. **Prediction Endpoint (/predictdata)**: Accepts POST requests containing user input data and returns the predicted outcome. This route also renders an HTML template (`home.html`) to display the prediction results.

The application uses a custom data class (`CustomData`) to collect and preprocess user input data. The input data is then passed through a prediction pipeline (`PredictPipeline`) to generate predictions.

## Code Description

The main Python script (`app.py`) contains the Flask application setup and route definitions. Here's an overview of the key components:

- `app.py`: Contains the Flask application setup, route definitions, and logic for handling user requests.
  
- `predict_pipeline.py`: Defines the `CustomData` class for data collection and preprocessing, as well as the `PredictPipeline` class for making predictions using the trained machine learning model.

## Deployment

The application is deployed on an AWS cloud server to make it accessible over the internet. The deployment process involves configuring the server environment, installing required dependencies, and running the Flask application.

To deploy the application:

1. Set up an AWS EC2 instance.
2. Install necessary dependencies (Python, Flask, scikit-learn, etc.).
3. Upload the project files to the server.
4. Run the Flask application (`app.py`) using a production-ready web server like Gunicorn.
5. Configure security settings and network settings to ensure secure access to the application.

## Running the Application

To run the application locally:

1. Install Python and required dependencies (`Flask`, `scikit-learn`, etc.).
2. Navigate to the project directory.
3. Run the Flask application using the command: `python app.py`.
4. Access the application in your web browser at `http://localhost:5000`.

## Disclaimer

This project is for demonstration purposes only and may not be suitable for production use without further testing, validation, and security hardening. Use caution when deploying machine learning applications in real-world scenarios.
