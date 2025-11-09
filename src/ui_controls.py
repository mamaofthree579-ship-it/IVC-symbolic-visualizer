import streamlit as st

def play_pause_controls(key_prefix):
    play = st.button("▶️ Play", key=f"{key_prefix}_play")
    pause = st.button("⏸️ Pause", key=f"{key_prefix}_pause")
    reset = st.button("🔄 Reset", key=f"{key_prefix}_reset")
    return play, pause, reset
