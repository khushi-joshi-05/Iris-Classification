
# Iris Flower Classification

This project demonstrates the classification of Iris flowers into three species (Setosa, Versicolor, and Virginica) using various machine learning models. The project utilizes the Iris dataset, a well-known resource for machine learning classification tasks.

## Dataset

The Iris dataset consists of 150 samples of Iris flowers, each with four features: sepal length, sepal width, petal length, and petal width. The dataset is commonly used for classification tasks.

## Project Workflow

The project follows these steps:

1. **Data Loading and Preprocessing:** Load the Iris dataset, handle missing values if any, and perform min-max normalization for feature scaling.
2. **Exploratory Data Analysis:** Visualize data using scatter plots, box plots, heatmaps, and histograms to understand data distribution and feature relationships.
3. **Dimensionality Reduction (PCA):** Apply Principal Component Analysis (PCA) to reduce the data’s dimensionality for visualization and potentially improved model performance.
4. **Model Training and Evaluation:** Train various machine learning models, including Random Forest, Decision Tree, SVM, k-NN, and a Stacked Model. Evaluate model performance using accuracy, precision, recall, and F1-score.
5. **Hyperparameter Tuning:** Use GridSearchCV to find the best hyperparameter settings for each model.
6. **Model Comparison and Selection:** Compare model performance and select the best-performing model based on evaluation metrics.
7. **Cross-Validation:** Apply cross-validation to assess the generalization performance of the selected model for more robust accuracy on unseen data.

## Results

The following table summarizes the performance of the classification models trained on the Iris dataset, along with their best hyperparameters:

| Model           | Accuracy | Precision | Recall | F1-Score | Best Hyperparameters                                         |
|-----------------|----------|-----------|--------|----------|----------------------------------------------------------------|
| Random Forest   | 1.000    | 1.000     | 1.000  | 1.000    | `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 50}` |
| Decision Tree   | 1.000    | 1.000     | 1.000  | 1.000    | `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}` |
| SVM             | 0.967    | 0.969     | 0.967  | 0.966    | `{'C': 10, 'kernel': 'linear'}`                                  |
| k-NN            | 1.000    | 1.000     | 1.000  | 1.000    | `{'n_neighbors': 9, 'weights': 'distance'}`                      |
| Stacked Model   | 1.000    | 1.000     | 1.000  | 1.000    | `{'final_estimator__C': 0.1}`                                    |

## Usage

To use the pre-trained Random Forest model:

1. **Install necessary libraries:**
   ```bash
   pip install scikit-learn joblib pandas numpy
   ```

2. **Load the model:**
   ```python
   import joblib
   model = joblib.load('iris_rf_model.pkl')
   ```

3. **Prepare input data:**
   - Create a NumPy array or pandas DataFrame containing the features (sepal length, sepal width, petal length, petal width) of the Iris flower you want to classify.
   - Normalize the input data with min-max scaling as was done in training.

4. **Make predictions:**
   ```python
   predictions = model.predict(input_data)
   ```
   `predictions` will contain the predicted species labels (0, 1, or 2) corresponding to the input data. You can map these labels back to the species names (Setosa, Versicolor, Virginica) if needed.



###  Report on Iris Dataset Classification Analysis

---

#### 1. **Data Loading and Preprocessing**

- **Import**: Load the Iris dataset using `pd.read_csv('IRIS.csv')`.
- **Handling Missing Data**: Check for missing values with `df.isnull().sum()` and handle (though there were none) using `df.dropna()`.
- **Min-Max Normalization**: Normalize features to a range of 0 to 1 with a custom `min_max_normalization` function.
- **Descriptive Statistics**: Examine statistics, such as mean and standard deviation, using `df.describe()`.

#### 2. **Data Visualization**

- **Scatter Plot**: Visualize relationships between features color-coded by species.
- **Box Plot**: Show the distribution of sepal length by species to reveal median, quartiles, and potential outliers.
- **Heatmap**: Display correlation matrix to identify relationships between numerical features.
- **Histograms**: Show each feature’s distribution across species.

#### 3. **Dimensionality Reduction (PCA)**

- **PCA Application**: Apply Principal Component Analysis (PCA) to reduce the data to 2 components.
- **Standardization**: Standardize data before PCA to improve separation.
- **Visualization**: Plot reduced data in 2D space to visualize species clustering.

#### 4. **Model Training, Evaluation, and Hyperparameter Tuning**

- **Dataset Split**: Split dataset into training and testing sets (80/20).
- **Models Trained**: Random Forest, Decision Tree, SVM, k-NN, and Stacked Model.
- **Metrics Used**: Accuracy, precision, recall, and F1-score.
- **Hyperparameter Tuning**: Use GridSearchCV to select optimal hyperparameters.

#### 5. **Results and Analysis**

- **Performance Comparison**: Analyze the accuracy of each model.
- **Model Selection**: Select the best model (Random Forest or Decision Tree) based on accuracy.
- **Cross-Validation**: Use cross-validation to ensure robustness and generalization.

#### 6. **Important Considerations**

- **Overfitting**: Cross-validation is employed to address potential overfitting.
- **Generalizability**: Notes on performance limitations on real-world data.
- **Future Directions**: Suggestions include further feature engineering or testing other model architectures.

---

### Summary Table of Results

| Model           | Accuracy | Precision | Recall | F1-Score |
|-----------------|----------|-----------|--------|----------|
| Random Forest   | 1.000    | 1.000     | 1.000  | 1.000    |
| Decision Tree   | 1.000    | 1.000     | 1.000  | 1.000    |
| SVM             | 0.967    | 0.969     | 0.967  | 0.966    |
| k-NN            | 1.000    | 1.000     | 1.000  | 1.000    |
| Stacked Model   | 1.000    | 1.000     | 1.000  | 1.000    |

---

### Conclusion

This report showcases the successful application of multiple classification models on the Iris dataset, with high precision, recall, and F1-scores. The Random Forest model proved to be especially effective, indicating its suitability for similar classification tasks. This pipeline, which includes data preprocessing, model training, evaluation, hyperparameter tuning, and validation, presents a reliable approach to classification for structured datasets like Iris.
```

