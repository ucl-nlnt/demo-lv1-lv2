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

#theme = gr.themes.Default(primary_hue= gr.themes.colors.emerald, secondary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate)

theme = gr.themes.Monochrome(
    secondary_hue="emerald",
    neutral_hue="zinc",
    radius_size="sm",
    font=[gr.themes.GoogleFont('Work sans'), 'Work sans', 'system-ui', 'sans-serif'],
).set(
    color_accent_soft='*secondary_950',
    prose_header_text_weight='800',
    shadow_drop='none',
    shadow_drop_lg='none',
    shadow_inset='none',
    shadow_spread='none',
    shadow_spread_dark='none',
    block_label_border_color='*neutral_100',
    block_label_border_color_dark='*neutral_50',
    section_header_text_size='*text_lg',
    button_primary_background_fill_hover='*secondary_950',
    button_primary_border_color_hover='*secondary_950'
)

#print('Waiting for Turtlbot connection...')
ttb_script_path = os.path.join(os.getcwd(),"demo_ttb.py")
launch_demo_ttb = subprocess.Popen(f'python3 {ttb_script_path}', stdout=subprocess.DEVNULL, shell=True)
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
        
    history = deque([])
    state_number = 0
        
    x_dict = {"instruction complete" : "#ongoing"}
    i = 0

    while x_dict["instruction complete"] == "#ongoing":
        
        if i != 11:
            i += 1 
  
        if history != deque([]):
            x = main(prompt, [i for i in history])
        else:
            x = main(prompt, "None")

        x_dict = ast.literal_eval(x)

        print('Predicted:', x_dict)
        lin_x, ang_z = x_dict['movement message']
        dt = x_dict['execution length']
        code = 1 if x_dict['instruction complete'] == '#complete' else 0

        progress(i/12, desc=f'Ongoing... Next Action: ({str(lin_x)}, {str(ang_z)}, {str(dt)})')

        mess = str([lin_x, ang_z, dt, code])

        server.send_data(mess.encode())
        data = ast.literal_eval(server.receive_data().decode())

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

with gr.Blocks(theme=theme, title = "NLNT Demo") as demo:
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


demo.launch()