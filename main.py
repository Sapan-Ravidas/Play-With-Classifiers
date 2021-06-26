import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from util import BuildModel

st.set_option('deprecation.showPyplotGlobalUse', False)

def add_parameter_ui(classifier):
    parameters = dict()
    if classifier == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        parameters["K"] = K
    elif classifier == "SVM":
        C = st.sidebar.slider("C", 0.01, 1.00)
        parameters["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        parameters["max_depth"] = max_depth
        parameters["n_estimators"] = n_estimators
    return parameters    


# Heading
st.title("Machine Learning Models")
st.write("""
         ## Explore Classifiers
         """)

dataset = st.sidebar.selectbox("Select data-sets", ("Iris-dataset", "Breast-cancer-dataset", "Wine-dataset"))
classifiers = st.sidebar.selectbox("Select classifiers", ("KNN", "SVM", "Random-Forest"))
parameters = add_parameter_ui(classifiers)

model = BuildModel(classifiers, parameters)

model.get_dataset(dataset)
st.write("shape of dataset: ", model.X.shape)
st.write("Number of classes: ", len(np.unique(model.Y)))

model.train_model()
st.write(f"classifier: {classifiers}")
st.write(f"accuracy: {model.accuracy}")

# PLOTs
pca = PCA(2)
X_projected = pca.fit_transform(model.X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c = model.Y, alpha = 0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("principal Component 2")
plt.colorbar()

st.pyplot()

