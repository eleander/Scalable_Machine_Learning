import modal

LOCAL=False
HOURS=24
IMAGE_FOLDER="monitor"

if LOCAL == False:
   stub = modal.Stub("heart_batch_inference")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks", "dataframe-image", "joblib", "seaborn", "scikit-learn==1.3.2", "shap"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(hours=HOURS), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    from datetime import datetime, timedelta
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    import seaborn as sns
    import shap
    import os
    import matplotlib.pyplot as plt

    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("heart_model_v1", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/heart_model.pkl")
    preprocessing_pipeline = joblib.load(model_dir + "/preprocessing_pipeline.pkl")
    
    fg = fs.get_feature_group(name="heart", version=1)
    df = fg.read()

    # Filter so we get only last data added
    now = datetime.now()
    diff = now - timedelta(hours=HOURS)

    df['clean_timestamp'] = pd.to_datetime(df.timestamp).dt.tz_localize(None)
    df = df[df['clean_timestamp'] >= diff]    

    # remove clean_timestamp 
    df = df.drop(columns=['clean_timestamp'])
    
    # Hacky fix due to Hopsworks Magic
    df["timestamp"] = df['timestamp'] - pd.to_timedelta(0 * df.index, unit='s')

    y_true = df['heart_disease']
    X = preprocessing_pipeline.transform(df)
    y_pred = model.predict(X)

    # Store predictions and truth
    print("Storing predictions in Monitor_df")
    print(df)
    monitor_df = pd.DataFrame({"pred": y_pred, "true": y_true, "timestamp": df['timestamp']})
    monitor_fg = fs.get_or_create_feature_group(
        name="heart_predictions",
        version=1,
        primary_key=monitor_df.columns,
        description="Heart Monitoring Predictions",
        event_time="timestamp",
    )
    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    print("Finished insertion")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_cols = ['True: ' + str(col) for col in ["0", "1"]]
    pred_rows = ['Pred: ' + str(col) for col in ["0", "1"]]
    df_cm = pd.DataFrame(conf_matrix, true_cols, pred_rows)
    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig(f"{IMAGE_FOLDER}/confusion_matrix_heart.png")
    print("Added confusion matrix")

    # Historical data
    hist_df = monitor_fg.read()
    concat_df = pd.concat([hist_df, monitor_df])
    dfi.export(concat_df.tail(5), f"{IMAGE_FOLDER}/df_recent_heart.png", table_conversion = 'matplotlib')
    print("Added historical data")
    
    # Explainability
    shap.initjs()

    concat_explain_df = concat_df.drop(columns=['timestamp'])
    
    explainer = shap.TreeExplainer(model)    
    shap_values = explainer.shap_values(concat_explain_df, approximate=True, check_additivity=False)
    print("Successfully trained shap")
    shap.summary_plot(shap_values[1], concat_explain_df, show=False)
    print("Created summary_plot")
    plt.savefig(f"{IMAGE_FOLDER}/shap_heart.png")
    print("Added explainability")

    # Upload images
    print("Began uploading images....")
    dataset_api = project.get_dataset_api()
    dataset_api.upload(f"{IMAGE_FOLDER}/confusion_matrix_heart.png", "Resources/images", overwrite=True)
    dataset_api.upload(f"{IMAGE_FOLDER}/df_recent_heart.png", "Resources/images", overwrite=True)
    dataset_api.upload(f"{IMAGE_FOLDER}/shap_heart.png", "Resources/images", overwrite=True)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()