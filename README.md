# Car Dheko - Used Car Price Prediction

## Project Overview
Car Dheko aims to enhance customer experience and streamline the pricing process using machine learning. This project develops an interactive Streamlit-based web application that predicts used car prices based on various features. The model is trained on historical data and deployed as a user-friendly tool for customers and sales representatives.

## Skills Takeaway
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning Model Development
- Price Prediction Techniques
- Model Evaluation and Optimization
- Model Deployment
- Streamlit Application Development
- Documentation and Reporting

## Domain
- Automotive Industry
- Data Science
- Machine Learning

## Problem Statement
Car Dheko requires a machine learning solution to predict the price of used cars based on historical data. The aim is to build a robust prediction model and integrate it into an interactive Streamlit application for real-time use by customers and sales representatives.

## Project Scope
The project utilizes historical data from CarDekho, covering multiple cities and car features. The machine learning model will process this structured dataset, predict car prices, and be deployed as a Streamlit-based web application.

## Approach
### Data Processing
1. **Import and Concatenation**
   - Combine multiple city-wise datasets into a structured format.
   - Add a 'City' column to identify the respective city for each record.
2. **Handling Missing Values**
   - Fill numerical missing values with mean, median, or mode.
   - Handle categorical missing values with mode or a new category.
3. **Standardizing Data Formats**
   - Convert data types appropriately (e.g., removing units like 'kms').
4. **Encoding Categorical Variables**
   - One-hot encoding for nominal variables.
   - Label encoding for ordinal variables.
5. **Normalizing Numerical Features**
   - Apply Min-Max Scaling or Standard Scaling where necessary.
6. **Removing Outliers**
   - Use IQR (Interquartile Range) or Z-score analysis.

### Exploratory Data Analysis (EDA)
1. **Descriptive Statistics**
   - Compute mean, median, standard deviation, etc.
2. **Data Visualization**
   - Use scatter plots, histograms, box plots, and correlation heatmaps.
3. **Feature Selection**
   - Use correlation analysis and feature importance from models.

### Model Development
1. **Train-Test Split**
   - Split dataset into training (70-80%) and testing (20-30%).
2. **Model Selection**
   - Consider algorithms like Linear Regression, Decision Trees, Random Forests, and Gradient Boosting Machines.
3. **Model Training**
   - Implement cross-validation for robust performance.
4. **Hyperparameter Tuning**
   - Use Grid Search or Random Search for optimization.

### Model Evaluation
- **Performance Metrics:**
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared (R²)
- **Model Comparison:**
  - Compare different models to select the best performer.

### Optimization
1. **Feature Engineering**
   - Create or modify features based on domain knowledge.
2. **Regularization**
   - Apply Lasso (L1) or Ridge (L2) regularization.

### Deployment
1. **Streamlit Application**
   - Deploy the final model using Streamlit.
   - Enable real-time price prediction based on user inputs.
2. **User Interface Design**
   - Ensure user-friendliness with clear instructions and error handling.

## Results
- A functional and accurate price prediction model.
- Detailed analysis and visualizations.
- Streamlit application for real-time predictions.
- Comprehensive documentation.

## Project Evaluation Metrics
1. **Model Performance**
   - MAE, MSE, R-squared.
2. **Data Quality**
   - Completeness and accuracy of preprocessed data.
3. **Application Usability**
   - User satisfaction and feedback.
4. **Documentation**
   - Clarity and completeness.

## Technical Tags
- Data Preprocessing, Machine Learning, Price Prediction, Regression, Python, Pandas, Scikit-Learn, EDA, Streamlit, Model Deployment

## Dataset Details
- Contains multiple Excel files, each representing a city.
- Includes car details, specifications, and features.
- Data source: CarDekho.

## Project Deliverables
- Source code for data processing and model training.
- Documentation detailing methodology, models, and results.
- Data visualizations and analysis reports.
- Final predictive model with user guide.
- Deployed Streamlit application.

## Project Guidelines
1. Follow PEP 8 coding standards.
2. Use Git for version control.
3. Write clear documentation and comments.
4. Ensure modular and reusable code.
5. Validate and test models thoroughly.
6. Design a user-friendly Streamlit application.

---

### Contributors
Developed by: [Jeba Jini]

For queries and suggestions, feel free to reach out!

