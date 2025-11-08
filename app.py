import streamlit as st
from src.vector_data import sample_water_symbol
from src.vector_plot import draw_vector_field
from modules.visualization import plot_vector_field, plot_symbolic_lattice, show_frequency_chart
from modules.analytics import (
    calculate_symbol_frequencies,
    resonance_matrix,
    find_resonant_clusters,
    generate_resonance_spectrum
)

st.set_page_config(page_title='IVC Symbolic Visualizer', layout='centered')

st.title('IVC Symbolic Visualizer — Vector Flow (sample)')
st.markdown('A neutralized sample visualization inspired by water glyph geometry. For research and educational use.')

mode = st.selectbox('Flow mode', ['default', 'calm', 'intense'])
arrow_scale = st.slider('Arrow scale (visual size)', min_value=10, max_value=60, value=25)
show_colorbar = st.checkbox('Show colorbar (magnitude)', value=True)

data = sample_water_symbol(mode=mode)
fig = draw_vector_field(data, show_colorbar=show_colorbar, arrow_scale=arrow_scale, figsize=(6,6))

st.pyplot(fig)

st.markdown('**Metadata**')
st.json(data['meta'])

st.markdown('---')
st.markdown('Notes: This demo uses simplified, neutralized geometry for visualization. Respect cultural boundaries when working with real ceremonial symbols.')
