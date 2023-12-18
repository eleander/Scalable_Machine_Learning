import os
import modal
from PIL import Image, ImageDraw
import numpy as np

LOCAL=True

if LOCAL == False:
   stub = modal.Stub("heart_batch_inference")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe-image", "Pillow", "shap"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from datetime import datetime
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import datetime


    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("heart_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/heart_model.pkl")
    
    feature_view = fs.get_feature_view(name="heart", version=1, event_time="timestamp")


    start_date = (datetime.datetime.now() - datetime.timedelta(hours=24))
    end_date = (datetime.datetime.now())

    batch_data = feature_view.get_batch_data(
        start_time=start_date,
        end_time=end_date
    )
    print(batch_data)
    y_pred = model.predict(batch_data)

    heart_fg = fs.get_feature_group(name="heart", version=1)
    df = heart_fg.read()
    y_true = df.iloc[:-y_pred.shape[0]]['heartdisease']
    print(y_true)

    
    dataset_api = project.get_dataset_api()    
    
    monitor_fg = fs.get_or_create_feature_group(name="heart_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Heart Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    
    data = {
        'prediction': [y for y in y_pred],
        'label': [y for y in y_true],
        'datetime': [now] * len(y_pred),
    }
    # data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}
    monitor_df = pd.DataFrame.from_dict(data)
    print(monitor_df)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # # Add our prediction to the history, as the history_df won't have it - 
    # # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])
    print(history_df)
    raise Exception("STOP")


    # df_recent = history_df.tail(4)
    # dfi.export(df_recent, './df_recent_heart.png', table_conversion = 'matplotlib')
    # dataset_api.upload("./df_recent_heart.png", "Resources/images", overwrite=True)
    
    # predictions = history_df[['prediction']]
    # labels = history_df[['label']]

    # # Only create the confusion matrix when our heart_predictions feature group has examples of all hearts
    # columns = sorted(list(set(np.unique(labels)) | set(np.unique(predictions))))
    # heart_count = len(columns)
    # print("Number of different heart quality predictions or truth up to date: " + str(heart_count))
    # # We modified the code so that the confusion matrix is generated dynamically depending on the selected labels and predictions
    # if heart_count < 2:
    #     # Create an empty image to avoid deployment errors in the monitor app
    #     empty_image = Image.new('RGB', (100, 100), color = (73, 109, 137))
    #     empty_image.save("./confusion_matrix_heart.png")
    # else:
    #     results = confusion_matrix(labels, predictions)
    #     true_cols = [f'True {col}' for col in columns]
    #     pred_cols = [f'Pred {col}' for col in columns]

    #     df_cm = pd.DataFrame(results, true_cols, pred_cols)

    #     cm = sns.heatmap(df_cm, annot=True)

    #     fig = cm.get_figure()
    #     fig.savefig("./confusion_matrix_heart.png")

    # dataset_api.upload("./confusion_matrix_heart.png", "Resources/images", overwrite=True)



if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
