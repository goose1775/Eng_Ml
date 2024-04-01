models_map_class = {
    "Logistic Regression": "lr",
    "K-Nearest Neighbors": "knn",
    "Naive Bayes": "nb",
    "Decision Tree": "dt",
    "SVM - Linear Kernel": "svm",  
    "SVM - Radial Kernel": "rbfsvm",
    "Gaussian Process": "gpc",
    "Multi-Layer Perceptron": "mlp",
    "Ridge Classifier": "ridge",
    "Random Forest": "rf",
    "Quadratic Discriminant Analysis": "qda",
    "AdaBoost Classifier": "ada",
    "Gradient Boosting": "gbc",
    "Linear Discriminant Analysis": "lda",
    "Extra Trees": "et",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "CatBoost": "catboost",
    "Dummy Classifier": "dummy"
}
 
plots_map_class = {
    "Pipeline Schematic": "pipeline",
    "Area Under the Curve (AUC)": "auc",
    "Discrimination Threshold": "threshold",
    "Precision Recall Curve": "pr",
    "Confusion Matrix": "confusion_matrix",
    "Class Prediction Error": "error",
    "Classification Report": "class_report",
    "Decision Boundary": "boundary",
    "Recursive Feature Selection": "rfe",
    "Learning Curve": "learning",
    "Manifold Learning": "manifold",
    "Calibration Curve": "calibration",
    "Validation Curve": "vc",
    "Dimensionality Reduction": "dimension",  # Minor correction for clarity
    "Feature Importance": "feature",
    "Feature Importance (All)": "feature_all",
    "Model Hyperparameters": "parameter",
    "Lift Curve": "lift",
    "Gain Chart": "gain",
    "Decision Tree": "tree",
    "KS Statistic Plot": "ks"
}

metrics_list_class = {    
    "Acurácia": "accuracy_score",
    "Recall": "recall_score",
    "Precisão": "precision_score",
    "F1-score": "f1_score",
}

metrics_list_eval = {    
    "Acurácia": "accuracy_score",
    "Recall": "recall_score",
    "Precisão": "precision_score",
    "F1-score": "f1_score",
    "Log Loss Avaliação": "log_loss",
    "Área Sob a Curva (ROC) Validação":"roc_auc",
    "Precisão/Recall Avaliação": "precision_recall_auc",
    "Tamanho Dataset Avaliação:": "example_count",    
}

optimize_class = {
    "Acurácia": "Accuracy",
    "Precisão": "Precision",
    "Recall": "Recall",
    "F1": "F1",
    "AUC": "AUC",
    "Kappa": "Kappa",
    "MCC": "MCC"
}

normalize_methods = {
    "Z-Score": "zscore",
    "MinMax": "minmax",
    "MaxAbs": "maxabs",
    "Robust": "robust"
}

outliers_methods = {
    "Isolation Forest": "iforest",
    "Elliptic Envelope": "ee",
    "Local Outlier Factor": "lof",
}

fold_strategys ={
    "Stratified K-fold": "stratifiedkfold",
    "Group Fold": "groupkfold",
    "Time Series": "timeseries",        
}
