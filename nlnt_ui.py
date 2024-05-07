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
    button_secondary_text_color_hover_dark='*neutral_50'
)

css = """
.color_btn textarea  {background-color: #228B22; !important}
"""

#print('Waiting for Turtlbot connection...')
#ttb_script_path = os.path.join(os.getcwd(),"demo_ttb.py")
#launch_demo_ttb = subprocess.Popen(f'python3 {ttb_script_path}', stdout=subprocess.DEVNULL, shell=True)
server = DataBridgeServer_TCP()

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
    
    # new output format
    
    x = main(prompt, "None")                          # initial prompt ; returns whether prompt is possible or not
    
    # if impossible return: "Action impossible. Please try again."
    # else go to while loop
    
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
        #x_dict['distance to next point'] = data['distance_traveled']

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

# js=
with gr.Blocks(theme=theme, css=css, title = "NLNT Demo",js="metadata.js") as demo:
    with gr.Row():
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
    with gr.Row():
        gr.HTML("""
        <div style='height: 30px; width: 100%;'>
            <div style='display:flex;justify-content:space-around;'>
                <div>
                    <span style='font-weight:bold;'>Total Distance Traveled:</span> <span id="total-dist"></span>
                </div>
                <div>
                    <span style='font-weight:bold;'>Total Degrees Rotated:</span> <span id="total-degs"></span>
                </div>
            </div>
        </div>
        """)


demo.launch()