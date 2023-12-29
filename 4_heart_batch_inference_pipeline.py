import modal

LOCAL=True
HOURS=24
IMAGE_FOLDER="monitor"

if LOCAL == False:
   stub = modal.Stub("heart_batch_inference")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "seaborn", "scikit-learn==1.1.1", "shap"])
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
    import datetime
    import shap
    import os
    import matplotlib.pyplot as plt

    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("heart_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/heart_model.pkl")
    preprocessing_pipeline = joblib.load(model_dir + "/preprocessing_pipeline.pkl")
    
    fg = fs.get_feature_group(name="heart", version=1)
    df = fg.read()

    # Filter so we get only last data added
    now = datetime.now()
    df = df[(df['timestamp'] >= (now - timedelta(hours=HOURS)))]
    
    y_true = df['heart_disease']
    X = preprocessing_pipeline.transform(df)
    y_pred = model.predict(X)

    # Store predictions and truth
    monitor_df = pd.DataFrame({"pred": y_pred, "true": y_true, "timestamp": df['timestamp']})
    monitor_fg = fs.get_or_create_feature_group(
        name="heart_predictions",
        version=1,
        primary_key=monitor_df.columns,
        description="Heart Monitoring Predictions",
        event_time="timestamp",
    )
    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    true_cols = ['True: ' + str(col) for col in ["0", "1"]]
    pred_rows = ['Pred: ' + str(col) for col in ["0", "1"]]
    df_cm = pd.DataFrame(conf_matrix, true_cols, pred_rows)
    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig(f"{IMAGE_FOLDER}/confusion_matrix_heart.png")

    # Historical data
    hist_df = monitor_fg.read()
    concat_df = pd.concat([hist_df, monitor_df])
    dfi.export(concat_df.tail(5), f"{IMAGE_FOLDER}/df_recent.png")
    
    # Explainability
    shap.initjs()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(concat_df, approximate=True)
    shap.summary_plot(shap_values[1], concat_df, show=False)
    plt.savefig(f"{IMAGE_FOLDER}/shap.png")

    # Historical metrics
    # Create temporary column for grouping by groups of HOURS from now
    concat_df["groups"] = (now - concat_df['timestamp']).apply(lambda x: x.total_seconds()) // (HOURS * 3600)

    def groupby_fn(x):
        min_timestamp = x["timestamp"].min()
        accuracy = accuracy_score(x["true"], x["pred"])
        f1_weighted = f1_score(x["true"], x["pred"], average="weighted")
        return pd.Series({"timestamp": min_timestamp, "accuracy": accuracy, "f1_weighted": f1_weighted})

    group_df = concat_df.groupby(by="groups", group_keys=False).apply(groupby_fn).set_index("timestamp", drop=True)
    ax = group_df.plot()
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Timestamp")
    fig = ax.get_figure()
    fig.savefig(f"{IMAGE_FOLDER}/metrics.png")

    # Upload images
    dataset_api = project.get_dataset_api()
    dataset_api.upload(f"{IMAGE_FOLDER}/confusion_matrix_heart.png", "Resources/images", overwrite=True)
    dataset_api.upload(f"{IMAGE_FOLDER}/df_recent_heart.png", "Resources/images", overwrite=True)
    dataset_api.upload(f"{IMAGE_FOLDER}/shap_heart.png", "Resources/images", overwrite=True)
    dataset_api.upload(f"{IMAGE_FOLDER}/metrics_heart.png", "Resources/images", overwrite=True)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
