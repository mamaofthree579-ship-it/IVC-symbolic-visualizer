import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

st.title("🗂️ Symbol Dialect Mapper")

df = st.file_uploader("Upload symbol-feature CSV", type=["csv"])

if df:
    df = pd.read_csv(df)
    n = st.slider("Clusters", 2, 12, 4)

    k = KMeans(n_clusters=n)
    df["dialect"] = k.fit_predict(df.iloc[:,1:])

    st.dataframe(df)

    st.success("Dialect clustering complete.")
