## with CSS & livestreaming (WORKS !)
# documentation for ASR Demo with Transformers : https://www.gradio.app/guides/real-time-speech-recognition
# documentation for livestreaming: https://www.gradio.app/guides/reactive-interfaces

import gradio as gr
from transformers import pipeline
import numpy as np
from inference_gradio import main
import json
import ast
from knetworking import DataBridgeServer_TCP
from collections import deque

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

theme = gr.themes.Default(primary_hue= gr.themes.colors.emerald, secondary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate).set(
    button_primary_background_fill="*primary_200",
    button_primary_background_fill_hover="*primary_200",
)

print('Waiting for Turtlbot connection...')
server = DataBridgeServer_TCP()

css = """
.color_btn textarea  {background-color: #228B22; !important}
"""

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def nlnt (vid_check, prompt, video, history="None", progress=gr.Progress()):
    progress(0, desc="Starting...")

    if vid_check == True:
        return level3_model(prompt, video)
    else:

        server.send_data('START')

        # Level 2 Model
        x_dict = ast.literal_eval(main(prompt))
        
        history = deque([])

        state_number = 0
        while x_dict["instruction complete"] == "#ongoing":       # JSON bug here where it doesn't recognize dictionary 
            
            #while "#ongoing" in x:
            # send x to Turtlebot using TCP => Turtlebot churva
            # send json string to Turtlebot
            # send x_dict["movement message"]
            # receive y from turtlebot using TCP
            # x_dict["orientation"] = actual_orientation
            # x_dict["distance to next point"] = actual_distance
            # x_dict["execution length"] = actual_execution length
            # send json string back and append to history
            # {
            #   "state number": "0x2", 
            #   "orientation": 1.29, 
            #   "distance to next point": 0.001, 
            #   "execution length": 1.402, 
            #   "movement message": (0.0, 0.2), 
            #   "instruction complete": "#complete"
            # }
            
            # x_dict is previous state;

            lin_x, ang_z = x_dict['movement_message']
            dt = x_dict['execution length']
            mess = str([lin_x, ang_z, dt, 0])

            server.send_data(mess.encode())
            data = ast.literal_eval(server.receive_data().decode())

            x_dict['state number'] = hex(state_number)
            x_dict['orientation'] = data['orientation']
            x_dict['distance to next point'] = data['distance_traveled']

            history.append(str(x_dict))
            if len(history) > 5:
               history.popleft()

            x = main(prompt, history)
            x_dict = ast.literal_eval(x)
            state_number += 1
            
        return x

def level3_model (prompt, video):
    return "level 3: " + prompt

def show_vid (vid_check):
    if vid_check:
      return gr.update(visible=True)
    else:
      return gr.update(visible=False)

with gr.Blocks(theme=theme, css=css, title = "NLNT Demo") as demo:
    gr.Markdown(
    """
    # Natural Language Ninja Turtle
    A Natural Language to ROS2 Translator for the Turtlebot V3 Burger
    """)
    with gr.Row():
        vid_check = gr.Checkbox(label = "connect live video")
    with gr.Row(equal_height=True):
        audio = gr.Audio(sources=["microphone"])
        prompt = gr.Textbox(label = "Instruction", placeholder = "move 1.5 meters forward", interactive = True)
    with gr.Row():
        clr_audio = gr.ClearButton(value = "clear audio", components = [audio])
        transcribe_btn = gr.Button(value = "Transcribe", elem_classes = "color_btn")
        clr_text = gr.ClearButton(value = "clear text", components = [audio, prompt])
        transcription = transcribe_btn.click(fn=transcribe, inputs=audio, outputs=prompt)
        #transcription = gr.Interface(transcribe, audio, prompt)
    with gr.Row():
      ttbt_btn = gr.Button(value = "Run Instruction", elem_classes = "color_btn")
    with gr.Row():
        video = gr.Image(sources=["webcam"], streaming=True, visible=False)
        ckbx = vid_check.select(fn = show_vid, inputs = vid_check, outputs = video)
        #vid_check.change(show_vid, vid_check, video)
    with gr.Row():
      with gr.Column():
        status = gr.Textbox(label = "Status", placeholder = "Please enter your prompt.")
        #history = None
        run_nlnt = ttbt_btn.click(fn=nlnt, inputs=[vid_check, prompt, video], outputs=status)


demo.launch()
