from runway_radar.api.db import REPO_ROOT
import json
import numpy as np

# this function is called once, when the server starts
# works with data from clip_labels.json
def load_artifact():
    read_path = REPO_ROOT / "src" / "runway_radar" / "api" / "artifacts" / "clip_labels.json"
    with open(read_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    temp_object = {}

    for axis_name, axis_content in data["axes"].items():
        names = []
        embeddings = []
        baselines = []
        for label_name, label_data in axis_content.items():
            names.append(label_name)
            embeddings.append(label_data["embedding"])
            baselines.append(label_data["baseline"])
        temp_object[axis_name] = {
            "names": names,
            "matrix": np.array(embeddings),
            "baselines": np.array(baselines),
        }

    logit_scale = data["logit_scale"]
    return_object = {"logit_scale": logit_scale, "axes": temp_object}

    return return_object

# this function will be used in the future /classify endpoint
# data from the embedding of the uploaded photo by the user + the prepared artifact
def classify(image_embedding, artifact):
    classify_return = {}
    image_embedding = image_embedding / np.linalg.norm(image_embedding)

    for axis_name, axis_content in artifact["axes"].items():
        sims = axis_content["matrix"] @ image_embedding
        adjusted = sims - axis_content["baselines"]
        scores = adjusted * artifact["logit_scale"]

        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()

        best = int(np.argmax(probs))
        classify_return[axis_name] = (axis_content["names"][best], round(float(probs[best]), 4))

    return classify_return
