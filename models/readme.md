

## Folder Contents

* **newfile.ipynb** – Main notebook containing full preprocessing, model training, ensemble stacking, SHAP analysis, and drift detection.

* **1D_CNN_model.keras** – Trained CNN student model for sequence-based solar forecasting.

* **LSTM_model.keras** – Trained LSTM student model capturing long-term temporal patterns.

* **TCN_model.keras** – Trained Temporal Convolutional Network student model.

* **Transformer_model.keras** – Transformer model tested during experiments but not used in final ensemble.

* **Teacher_XGBoost.joblib** – Final XGBoost teacher model used as the top-level meta-learner.

* **Teacher_NN.keras** – Neural network–based teacher model (alternate meta-model).

* **scaler_X.save** – Feature scaler used to normalize input weather variables.

* **scaler_y.save** – Target scaler used to normalize solar generation values.

* **teacher_feature_scaler.save / teacher_feature_scaler.joblib** – Scaler for the teacher model’s meta-features (student outputs + raw features).

* **teacher_meta_info.joblib** – Metadata needed for inference, such as feature ordering and shapes.

* **X_ref_compressed.npz** – Reference feature distribution used for feature drift detection (MMD).

* **y_ref.npy** – Reference target distribution used for KS-based drift detection.

* **y_pred_xgb_test.npy** – Saved test predictions from the XGBoost teacher model.

* **y_pred_nn_test.npy** – Saved test predictions from the teacher neural network model.

* **eda_report.html** - providing data cleaning steps, visualizations, and preliminary insights for the project.

* **readme.md** – Documentation file describing the project.

---

