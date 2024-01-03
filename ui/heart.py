from datetime import datetime
import gradio as gr
import hopsworks
import joblib
import pandas as pd
import numpy as np

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("heart_model_v1", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/heart_model.pkl")
preprocessing_pipeline = joblib.load(model_dir + "/preprocessing_pipeline.pkl")
print("Model downloaded")

generator = mr.get_model("heart_generator", version=1)
generator_dir = generator.download()
inverse_pipeline = joblib.load(generator_dir + "/inverse_pipeline.pkl")
print("Inverse pipeline downloaded")

def impute(df):
    df = preprocessing_pipeline.transform(df)
    df = inverse_pipeline.transform(df)
    return df

def predict(df):
    df = preprocessing_pipeline.transform(df)
    prediction = model.predict(df)
    return prediction[0]

def heart(heartdisease, smoking, alcoholdrinking, stroke, diffwalking, sex, agecategory, race, diabetic, physicalactivity, genhealth, asthma, kidneydisease, skincancer, mentalhealth, physicalhealth, sleeptime, bmi):
    df = pd.DataFrame({
        'smoking': [smoking],
        'alcohol_drinking': [alcoholdrinking],
        'stroke': [stroke],
        'diff_walking': [diffwalking],
        'sex': [sex],
        'age_category': [agecategory],
        'race': [race],
        'diabetic': [diabetic],
        'physical_activity': [physicalactivity],
        'gen_health': [genhealth],
        'asthma': [asthma],
        'kidney_disease': [kidneydisease],
        'skin_cancer': [skincancer],
        'b_m_i': [bmi],
        'mental_health': [mentalhealth],
        'physical_health': [physicalhealth],
        'sleep_time': [sleeptime],
    })

    # Replace Unknowns with NaNs
    # Feature pipeline has an imputer
    df = df.replace('Unknown', np.nan)
        
    if heartdisease != "Unknown":
        df = impute(df)

        df['heart_disease'] = heartdisease
        df['timestamp'] = pd.to_datetime(datetime.now())
        df["timestamp"] = df['timestamp'] - pd.to_timedelta(0 * df.index, unit='s')


        heart_fg = fs.get_feature_group(name="heart", version=1)
        heart_fg.insert(df)

        # If insert fails, insert the imputed value instead of nan

    pred = predict(df)

    if not pred:
        return "We predict that you do NOT have heart disease. (But this is not medical advice!)"
    else:
        return "We predict that you MIGHT have heart disease. (But this is not medical advice!)"

demo = gr.Interface(
    fn=heart,
    title="Heart Disease Predictive Analytics",
    description="Experiment with different heart configurations.",
    allow_flagging="never",
inputs=[
    gr.Dropdown(['Unknown', 'No', 'Yes'], label="Heart Disease (TARGET)"),
    gr.Dropdown(['Unknown', 'No', 'Yes'], label="Smoking"),
    gr.Dropdown(['Unknown', 'No', 'Yes'], label="Alcohol Drinking"),
    gr.Dropdown(['Unknown', 'No', 'Yes'], label="Stroke"),
    gr.Dropdown(['Unknown', 'No', 'Yes'], label="Diff Walking"),
    gr.Dropdown(['Unknown', 'Female', 'Male'], label="Sex"),
    gr.Dropdown(['Unknown', '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'], label="Age Category"),
    gr.Dropdown(['Unknown', 'American Indian/Alaskan Native', 'Asian', 'Black', 'Hispanic', 'Other', 'White'], label="Race"),
    gr.Dropdown(['Unknown', 'No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)'], label="Diabetic"),
    gr.Dropdown(['Unknown', 'No', 'Yes'], label="Physical Activity"),
    gr.Dropdown(['Unknown', 'Poor', 'Fair', 'Good', 'Very good', 'Excellent'], label="General Health"),
    gr.Dropdown(['Unknown', 'No', 'Yes'], label="Asthma"),
    gr.Dropdown(['Unknown', 'No', 'Yes'], label="Kidney Disease"),
    gr.Dropdown(['Unknown', 'No', 'Yes'], label="Skin Cancer"),
    gr.Number(label="Mental Health", minimum=0, maximum=30),
    gr.Number(label="Physical Health", minimum=0, maximum=30),
    gr.Number(label="Sleep Time", minimum=1, maximum=24),
    gr.Number(label="BMI"),
],
    outputs="text")

demo.launch(debug=True)
