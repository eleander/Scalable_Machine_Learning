import modal

LOCAL=True
HOURS = 24

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
    import datetime
    from datetime import datetime
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import datetime


    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("heart_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/heart_model.pkl")
    preprocessing_pipeline = joblib.load(model_dir + "/preprocessing_pipeline.pkl")
    
    fg = fs.get_feature_group(name="heart", version=1)
    df = fg.read()

    # sort by timestamp
    df = df.sort_values(by=['timestamp'], ascending=False)

    # get last HOURS of data based on timestamp
    start_date = (datetime.datetime.now() - datetime.timedelta(hours=HOURS))
    end_date = (datetime.datetime.now())

    df['timestamp'] = pd.to_datetime(df.timestamp).dt.tz_localize(None)
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    timestamp = df['timestamp']
    df = df.drop(['timestamp'], axis=1)
    
    y_true = df['heartdisease']
    df_processed = preprocessing_pipeline.transform(df)
    y_pred = model.predict(df_processed)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    true_cols = ['True: ' + str(col) for col in ["0", "1"]]
    pred_cols = ['Pred: ' + str(col) for col in ["0", "1"]]
    df_cm = pd.DataFrame(conf_matrix, true_cols, pred_cols)
    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig("./confusion_matrix_heart.png") 

    # C

    dataset_api = project.get_dataset_api()
    dataset_api.upload("./confusion_matrix_heart.png", "Resources/images", overwrite=True)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
