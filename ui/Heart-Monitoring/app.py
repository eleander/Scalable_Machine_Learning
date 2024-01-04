import gradio as gr
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/confusion_matrix_heart.png", overwrite=True)
dataset_api.download("Resources/images/shap_heart.png", overwrite=True)
dataset_api.download("Resources/images/df_recent_heart.png", overwrite=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Label("Feature Importance based on SHAP")
            gr.Image("shap_heart.png", elem_id="shap")       
    with gr.Row():
        with gr.Column():
            gr.Label("Recent Prediction History")
            gr.Image("df_recent_heart.png", elem_id="recent-predictions")
        with gr.Column():
            gr.Label("Confusion Maxtrix with Historical Prediction Performance")
            gr.Image("confusion_matrix_heart.png", elem_id="confusion-matrix")  

demo.launch(debug=False)  