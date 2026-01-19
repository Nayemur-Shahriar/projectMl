
import gradio as gr
import pandas as pd
import pickle
import numpy as np

#load
with open("mobile_price_rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

#logic
def predict_price_range(
    battery_power, blue, clock_speed, dual_sim, fc, four_g,
    int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
    px_width, ram, sc_h, sc_w, talk_time, three_g,
    touch_screen, wifi
):
    # match dataFram
    input_df = pd.DataFrame([[
        battery_power, blue, clock_speed, dual_sim, fc, four_g,
        int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
        px_width, ram, sc_h, sc_w, talk_time, three_g,
        touch_screen, wifi
    ]], columns=[
        'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
        'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
        'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
        'touch_screen', 'wifi'
    ])

    # Predict
    pred = model.predict(input_df)[0]

    #formatted result
    if pred == 0:
        return "Low Cost"
    elif pred == 1:
        return " Medium Cost"
    elif pred == 2:
        return "High Cost"
    else:
        return "Very High Cost"

#app Interface
inputs = [
    gr.Number(label="Battery Power", value=1500),
    gr.Radio([0, 1], label="Blue (0/1)", value=1),
    gr.Number(label="Clock Speed", value=2.0),
    gr.Radio([0, 1], label="Dual SIM (0/1)", value=1),
    gr.Number(label="Front Camera (fc)", value=5),
    gr.Radio([0, 1], label="4G (0/1)", value=1),

    gr.Number(label="Internal Memory (GB)", value=32),
    gr.Number(label="Mobile Depth (m_dep)", value=0.5),
    gr.Number(label="Mobile Weight", value=150),
    gr.Slider(1, 8, step=1, label="Number of Cores (n_cores)", value=4),
    gr.Number(label="Primary Camera (pc)", value=12),
    gr.Number(label="Pixel Height (px_height)", value=800),

    gr.Number(label="Pixel Width (px_width)", value=1200),
    gr.Number(label="RAM (MB)", value=2000),
    gr.Number(label="Screen Height (sc_h)", value=12),
    gr.Number(label="Screen Width (sc_w)", value=6),
    gr.Number(label="Talk Time", value=10),
    gr.Radio([0, 1], label="3G (0/1)", value=1),

    gr.Radio([0, 1], label="Touch Screen (0/1)", value=1),
    gr.Radio([0, 1], label="WiFi (0/1)", value=1),
]

app = gr.Interface(
    fn=predict_price_range,
    inputs=inputs,
    outputs="text",
    title="Mobile Price Classification"
)

app.launch(share=True)
