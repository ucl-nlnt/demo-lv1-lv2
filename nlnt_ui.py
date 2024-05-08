## with CSS & livestreaming (WORKS !)
# documentation for ASR Demo with Transformers : https://www.gradio.app/guides/real-time-speech-recognition
# documentation for livestreaming: https://www.gradio.app/guides/reactive-interfaces

import gradio as gr
from transformers import pipeline
import numpy as np
from inference_gradio import main
import ast
from knetworking import DataBridgeServer_TCP
from collections import deque
import subprocess
import os
import time
import json

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

theme = gr.themes.Monochrome(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont('Jersey 25 Charted '), gr.themes.GoogleFont('exo-2'), 'system-ui', 'sans-serif'],
).set(
    button_small_radius='*radius_sm',
    button_primary_background_fill='*secondary_800',
    button_primary_background_fill_dark='*neutral_400',
    button_primary_border_color_dark='*secondary_800',
    button_primary_text_color_dark='*neutral_800',
    button_primary_text_color_hover='*neutral_50',
    button_primary_text_color_hover_dark='*neutral_50',
    button_secondary_background_fill_dark='*neutral_400',
    button_secondary_background_fill_hover='*neutral_800',
    button_secondary_background_fill_hover_dark='*button_primary_background_fill_hover',
    button_secondary_text_color='*neutral_50',
    button_secondary_text_color_dark='*neutral_800',
    button_secondary_text_color_hover='*button_secondary_border_color_hover',
    button_secondary_text_color_hover_dark='*neutral_50',

    shadow_drop='none',
    shadow_drop_lg='none',
    shadow_inset='none',
    shadow_spread='none',
    shadow_spread_dark='none'
)

# <link href="https://fonts.googleapis.com/css2?family=Jersey+25+Charted&display=swap" rel="stylesheet">
# font-family: "Jersey 25 Charted", sans-serif;

css = """
h1 {
    text-align: center;
    font-size: 3vw;
    display:block;
}
p {
    text-align: center;
    font-size: 1.2vw;
    display:block;
}
"""

# print('Waiting for Turtlbot connection...')
# ttb_script_path = os.path.join(os.getcwd(),"demo_ttb.py")
# launch_demo_ttb = subprocess.Popen(f'python3 {ttb_script_path}', stdout=subprocess.DEVNULL, shell=True)
server = DataBridgeServer_TCP()
print(server)
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
        return level2_model(prompt, history, progress=gr.Progress())

def level2_model(prompt, history="None", progress=gr.Progress()):
    
    progress(0, desc="Starting...")
    server.send_data('START')
    
    history = deque([])
    state_number = 0
        
    x_dict = {"instruction complete" : "#ongoing"}
    i = 0

    while x_dict["instruction complete"] == "#ongoing":
        if history != deque([]):
            x = main(prompt, [i for i in history])                  # send prompt to lambda server
        else:
            x = main(prompt, "None")                                # send prompt to lambda server

        x_dict = ast.literal_eval(x)

        print('Predicted:', x_dict)
        lin_x, ang_z = x_dict['movement message']
        dt = x_dict['execution length']
        code = 1 if x_dict['instruction complete'] == '#complete' else 0
        expected_states = x_dict['expected number of states']               # TODO: double chack with actual output

        if i != expected_states:
            i += 1
            status = i/expected_states
        else:
            status = 0.99

        progress(status, desc=f'Ongoing... Next Action: ({str(lin_x)}, {str(ang_z)}, {str(dt)})')

        mess = str([lin_x, ang_z, dt, code])

        server.send_data(mess.encode())                                     # send the predicted action to turtlebot
        data = ast.literal_eval(server.receive_data().decode())             # receive actual action from turtlebot

        x_dict['state number'] = hex(state_number)
        x_dict['orientation'] = data['orientation']

        if data['blocked']:

            print('============= [WARN] =============')
            print('block received')
            print('============= [WARN] =============')
            server.send_data(str([0.0, 0.0, 0.0, 1]))
            x_dict['instruction complete'] = '#complete' # finish command
            break

        history.append(str(x_dict))
        if len(history) > 5:
            history.popleft()
            
        print(history[-1])
        print('\n')

        state_number += 1
            
    progress(1, desc="Movement done!")
    return "Instruction accomplished. Waiting for next instruction."

def level3_model (prompt, video):
    return "level 3: " + prompt

def show_vid (vid_check):
    if vid_check:
      return gr.update(visible=True)
    else:
      return gr.update(visible=False)
    
def total_distance():
    counter = 0

    while True:
        yield str(counter)
        counter += 1
    
    #return time.ctime()

def total_rotation():
   
    return time.ctime()

data_buffer = []

def super_json_listener():

        t = time.time() + 1.0
        x = 0
        gathering_data = True

        while True:
            
            if not gathering_data: time.sleep(0.007); continue

            if time.time() > t:  # used to calculate framerate for debug purposes
                t = time.time() + 1.0
                x = 0
            x += 1

            data = json.loads(server.receive_data().decode())
            if data_buffer == None: print("WARNING: data buffer is still None type."); continue

            
            data_buffer.append(data)

import threading
data_listener_thread = threading.Thread(target=super_json_listener)
data_listener_thread.start()

with gr.Blocks(theme=theme, css=css, title = "NLNT Demo",js="metadata.js") as demo:
    with gr.Row():
        gr.Markdown(
        """
        # Natural Language Ninja Turtle
        A Natural Language to ROS2 Translator for the Turtlebot V3 Burger
        """)
    with gr.Row():
        vid_check = gr.Checkbox(label = "Connect Live Video")
    with gr.Row(equal_height=True):
        audio = gr.Audio(sources=["microphone"])
        prompt = gr.Textbox(label = "Instruction", placeholder = "move 1.5 meters forward", interactive = True)
    with gr.Row():
        clr_audio = gr.ClearButton(value = "Clear Audio", components = [audio])
        transcribe_btn = gr.Button(value = "Transcribe")
        clr_text = gr.ClearButton(value = "Clear Text", components = [audio, prompt])
        transcription = transcribe_btn.click(fn=transcribe, inputs=audio, outputs=prompt)
        #transcription = gr.Interface(transcribe, audio, prompt)
    with gr.Row():
        ttbt_btn = gr.Button(value = "Run Instruction")
    with gr.Row():
        video = gr.Image(sources=["webcam"], streaming=True, visible=False)
        ckbx = vid_check.select(fn = show_vid, inputs = vid_check, outputs = video)
        #vid_check.change(show_vid, vid_check, video)
    with gr.Row():
        with gr.Column():
            status = gr.Textbox(label = "Status", placeholder = "Please enter your prompt.")
            #history = None
            run_nlnt = ttbt_btn.click(fn=nlnt, inputs=[vid_check, prompt, video], outputs=status)
    #with gr.Row():
    #    total_dist = gr.Textbox(value = next(total_distance()), label = "Total Distance Traveled", interactive=False)
    #    total_rot = gr.Textbox(value = total_rotation(), label = "Total Degrees Rotated", interactive=False, every=0.5)
    with gr.Row():
        gr.HTML("""
        <div style='height: 30px; width: 100%;'>
            <div style='display:flex;justify-content:space-around;'>
                <div>
                    <span style='font-weight:bold;'>Total Distance Traveled:</span> <span id="total-dist">0</span>
                </div>
                <div>
                    <span style='font-weight:bold;'>Total Degrees Rotated:</span> <span id="total-degs">0</span>
                </div>
            </div>
        </div>
        """)



### add webserver to host data from turtlebot
from fastapi import FastAPI
from random import randrange

webserver = FastAPI()
print("launching webserver")

@webserver.get("/hello")
def read_root():
    return {"Hello": "World"}


@webserver.get("/metadata")
def read_metadata():
    # TODO, format the data from server = DataBridgeServer_TCP() to get the different metadata
    total_distance_traveled = randrange(20, 50)
    total_degrees_rotated = randrange(-10,10)
    return {"total_distance_traveled": total_distance_traveled, "total_degrees_rotated": total_degrees_rotated}

app = gr.mount_gradio_app(webserver, demo, path="/")