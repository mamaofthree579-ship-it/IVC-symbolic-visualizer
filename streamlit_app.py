import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Indus Waveform Lite", layout="centered")

st.title("🛕 Indus Waveform Simulator — Mobile Edition")
st.write("""
This version is designed specifically to run **on Streamlit Cloud**, 
so you can open it on your **phone** with no installation needed.
""")

st.header("📤 Upload Indus Symbol")
uploaded = st.file_uploader("Upload PNG or JPG symbol", type=["png","jpg","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    arr = np.array(img)

    st.image(img, caption="Uploaded Symbol", width=250)

    st.subheader("📡 FFT Frequency Response")

    # Basic FFT
    F = np.abs(np.fft.fftshift(np.fft.fft2(arr)))

    fig, ax = plt.subplots()
    ax.imshow(np.log(F + 1), cmap="inferno")
    ax.set_title("Frequency Spectrum (log-scaled)")
    ax.axis("off")

    st.pyplot(fig)

st.header("🔊 Generate Test Waveform")
freq = st.slider("Frequency (Hz)", 1, 100, 20)
amp = st.slider("Amplitude", 1, 10, 3)

if st.button("Generate Waveform"):
    t = np.linspace(0, 1, 1000)
    y = amp * np.sin(2*np.pi*freq*t)

    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.set_title(f"Sine Wave — {freq} Hz")
    st.pyplot(fig)

st.info("This lightweight edition works fully on Streamlit Cloud and mobile browsers.")
