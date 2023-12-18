import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("heart_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks", "ydata-synthetic==1.1.0"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
   def f():
       g()


def generate_random_heart(project):
    from ydata_synthetic.synthesizers.regular import RegularSynthesizer

    mr = project.get_model_registry()
    model = mr.get_model("heart_generator", version=1)
    model_dir = model.download()
    model = RegularSynthesizer.load(model_dir + '/generator.pkl')
    sample = model.sample(1)
    print(sample)
    return sample

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    heart_sample = generate_random_heart(project)

    heart_fg = fs.get_feature_group(name="heart",version=1)
    heart_fg.insert(heart_sample)

if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        stub.deploy("heart_daily")
        with stub.run():
            f()