import os
import json
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.environ.get("AIP_MODEL_DIR", "/tmp"), "model.joblib")

model = joblib.load(MODEL_PATH)


def predict(instances):
    X = np.array(instances, dtype=float)
    preds = model.predict(X)
    return preds.tolist()


# Vertex default prediction entrypoint
def run():
    import sys
    data = json.load(sys.stdin)
    instances = data.get("instances", [])
    outputs = predict(instances)
    print(json.dumps({"predictions": outputs}))


if __name__ == "__main__":
    run()
