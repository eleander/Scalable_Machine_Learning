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
preprocessing_pipeline = joblib.load(model_dir + "/preprocessing_pipeline.pkl")
print("Model downloaded")

    # gr.Dropdown(["No", "Yes", "Unknown"], label="Heart Disease"),
    # gr.Dropdown(["Yes", "No"], label="Smoking"),
    # gr.Dropdown(["No", "Yes"], label="Alcohol Drinking"),
    # gr.Dropdown(["No", "Yes"], label="Stroke"),
    # gr.Dropdown(["No", "Yes"], label="Diff Walking"),
    # gr.Dropdown(["Female", "Male"], label="Sex"),
    # gr.Dropdown(['55-59', '80 or older', '65-69', '75-79', '40-44', '70-74', '60-64', '50-54', '45-49', '18-24', '35-39', '30-34', '25-29'], label="Age Category"),
    # gr.Dropdown(['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'], label="Race"),
    # gr.Dropdown(['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'], label="Diabetic"),
    # gr.Dropdown(['Yes', 'No'], label="Physical Activity"),
    # gr.Dropdown(['Very good', 'Fair', 'Good', 'Poor', 'Excellent'], label="General Health"),
    # gr.Dropdown(['Yes', 'No'], label="Asthma"),
    # gr.Dropdown(['No', 'Yes'], label="Kidney Disease"),
    # gr.Dropdown(['Yes', 'No'], label="Skin Cancer"),
    # gr.Number(label="BMI"),
    # gr.Number(label="Mental Health"),
    # gr.Number(label="Physical Health"),
    # gr.Number(label="Sleep Time")


def heart(heartdisease, smoking, alcoholdrinking, stroke, diffwalking, sex, agecategory, race, diabetic, physicalactivity, genhealth, asthma, kidneydisease, skincancer, bmi, mentalhealth, physicalhealth, sleeptime):
    # camel_case
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
        'general_health': [genhealth],
        'asthma': [asthma],
        'kidney_disease': [kidneydisease],
        'skin_cancer': [skincancer],
        'bmi': [bmi],
        'mental_health': [mentalhealth],
        'physical_health': [physicalhealth],
        'sleep_time': [sleeptime],
    })

    def predict(df):
        df = preprocessing_pipeline.transform(df)
        prediction = model.predict(df)
        return prediction[0]
    
    pred = predict(df) # 0.0 or 1.0
    
    if heartdisease != "Unknown":
        df['heart_disease'] = heartdisease
        df['timestamp'] =  datetime.now() - pd.to_timedelta(df.index, unit='s')
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        heart_fg = fs.get_feature_group(name="heart",version=1)
        heart_fg.insert(df)

    if not pred:
        return "We predict that you do NOT have heart disease. (But this is not medical advice!)"
    else:
        return "We predict that you MIGHT have heart disease. (But this is not medical advice!)"

demo = gr.Interface(
    fn=heart,
    title="Heart Disease Predictive Analytics",
    description="Experiment with different heart configurations.",
    allow_flagging="never",
    # default values are the mean values of the training dataset
inputs=[
    gr.Dropdown(["No", "Yes", "Unknown"], label="Heart Disease"),
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
    gr.Number(label="BMI"),
    gr.Number(label="Mental Health"),
    gr.Number(label="Physical Health"),
    gr.Number(label="Sleep Time")
],
    outputs="text")

demo.launch(debug=True)