from datetime import datetime
import gradio as gr
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("heart_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/heart_model.pkl")
ordinal = joblib.load(model_dir + "/ordinal_encoder.pkl")
scaler = joblib.load(model_dir + "/heart_scaler.pkl")
print("Model downloaded")

def heart(heartdisease, smoking, alcoholdrinking, stroke, diffwalking, sex, agecategory, race, diabetic, physicalactivity, genhealth, asthma, kidneydisease, skincancer, bmi, mentalhealth, physicalhealth, sleeptime):
    # ValueError: The feature names should match those that were passed during fit.
    # Feature names must be in the same order as they were in fit.
    df = pd.DataFrame({
        'Smoking': [smoking],
        'AlcoholDrinking': [alcoholdrinking],
        'Stroke': [stroke],
        'DiffWalking': [diffwalking],
        'Sex': [sex],
        'AgeCategory': [agecategory],
        'Race': [race],
        'Diabetic': [diabetic],
        'PhysicalActivity': [physicalactivity],
        'GenHealth': [genhealth],
        'Asthma': [asthma],
        'KidneyDisease': [kidneydisease],
        'SkinCancer': [skincancer],
        'HeartDisease': ["No"]
    })

    def predict(df):
        df = ordinal.transform(df)
        df = scaler.transform(df)
        df.drop(["HeartDisease"], axis=1)
        prediction = model.predict(df)
        return prediction[0]
    
    if not heartdisease == "Please predict":
        df['heartdisease'] = heartdisease
        pred = predict(df)

        df['timestamp'] =  datetime.now() - pd.to_timedelta(df.index, unit='s')
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        heart_fg = fs.get_feature_group(name="heart",version=1)
        heart_fg.insert(df)

    else:
        return "We predict that" + str(predict(df)) + "has heart disease"


    return "could not complete request"

demo = gr.Interface(
    fn=heart,
    title="Wine Predictive Analytics",
    description="Experiment with different wine configurations.",
    allow_flagging="never",
    # default values are the mean values of the training dataset
inputs=[
    gr.Dropdown(["No", "Yes", "Please predict"], label="Heart Disease"),
    gr.Dropdown(["Yes", "No"], label="Smoking"),
    gr.Dropdown(["No", "Yes"], label="Alcohol Drinking"),
    gr.Dropdown(["No", "Yes"], label="Stroke"),
    gr.Dropdown(["No", "Yes"], label="Diff Walking"),
    gr.Dropdown(["Female", "Male"], label="Sex"),
    gr.Dropdown(['55-59', '80 or older', '65-69', '75-79', '40-44', '70-74', '60-64', '50-54', '45-49', '18-24', '35-39', '30-34', '25-29'], label="Age Category"),
    gr.Dropdown(['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'], label="Race"),
    gr.Dropdown(['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'], label="Diabetic"),
    gr.Dropdown(['Yes', 'No'], label="Physical Activity"),
    gr.Dropdown(['Very good', 'Fair', 'Good', 'Poor', 'Excellent'], label="General Health"),
    gr.Dropdown(['Yes', 'No'], label="Asthma"),
    gr.Dropdown(['No', 'Yes'], label="Kidney Disease"),
    gr.Dropdown(['Yes', 'No'], label="Skin Cancer"),
    # numerical: Index(['BMI', 'MentalHealth', 'PhysicalHealth', 'SleepTime'], dtype='object')
    gr.Number(label="BMI"),
    gr.Number(label="Mental Health"),
    gr.Number(label="Physical Health"),
    gr.Number(label="Sleep Time")
],
    outputs="text")

demo.launch(debug=True)