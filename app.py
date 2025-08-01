import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib 

loaded_model = joblib.load('random_forest_model.pkl')


# Streamlit app layout
st.title("Random Forest Regressor Trainer")
st.write("Upload your dataset (CSV format) to train and evaluate a Random Forest model.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("Preview of dataset:")
    st.dataframe(df.head())

    # Let user pick target column
    target_column = st.selectbox("Select the target column (what you want to predict):", df.columns)

    # Features and target split
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Show feature columns
    st.write("Feature columns:", list(X.columns))

    # Train-test split
    test_size = st.slider("Test set size (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20]
    }

    # Train model button
    if st.button("Train Random Forest Regressor"):
        st.write("Training model...")

        # Initialize model and perform grid search
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=KFold(n_splits=3, shuffle=True, random_state=42),
                                   scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predict
        y_pred = best_model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Output
        st.subheader("Model Performance")
        st.write(f"**Best Parameters:** {grid_search.best_params_}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Optional: Plot true vs predicted
        st.subheader("Prediction vs Actual")
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.line_chart(results_df.reset_index(drop=True))
else:
    st.info("Please upload a CSV file to begin.")
