import pandas as pd
import streamlit as st
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sklearn
import numpy as np

df = pd.read_csv("diamonds.csv")

st.subheader("Price Estimation")


typess = {1: "Ideal", 2: "Premium", 3: "Very Good", 4:"Good",5:"Fair"}
colorss = {1:"D",2:"E",3:"F",4:"G",5:"H",6:"I",7:"J"}
clarss = {8:'IF', 7:'VVS1', 6:'VVS2', 5:'VS1', 4:'VS2', 3:'SI1', 2:'SI2', 1:'I1'}

def format_func(option):
    return typess[option]


carat = st.number_input("Carat:")
types = st.selectbox("Cut:",options=list(typess.keys()), format_func=lambda x:typess[x])
color = st.selectbox("Color:",options=list(colorss.keys()), format_func=lambda x:colorss[x])
clar = st.selectbox("Clarity:",options=list(clarss.keys()), format_func=lambda x:clarss[x])
depth = st.number_input("Depth:")
table = st.number_input("Table")
xz = st.number_input("x:")
yz = st.number_input("y:")
zz = st.number_input("z:")


def calculate():
    reg = LinearRegression()
    x = df[["carat", "types", "Colr", "clar", "depth", "table", "x", "y", "z"]]
    y = df[["price"]]
    model = reg.fit(x, y)
    tahmin = model.predict([[carat, types, color, clar, depth, table, xz, yz, zz]])
    tahmin2 = np.float(tahmin)
    tahmin2 = "%.2f" % tahmin2
    dogruluk = model.score(x,y)
    dogruluk = "%.2f" % dogruluk

    st.warning(f"Recommended Price:   {tahmin2}")
    st.warning(f"Probability:    {dogruluk}")

if st.button("Calculate"):
    calculate()




