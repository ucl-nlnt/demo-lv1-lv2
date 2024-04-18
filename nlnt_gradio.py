# Gazebo & SSH Quickstart Connection to Turtlebot

## with CSS & livestreaming (WORKS !)
# documentation for livestreaming: https://www.gradio.app/guides/reactive-interfaces
# should add connection to Turtlebot

import gradio as gr
from transformers import pipeline
import numpy as np
from inference_gradio import main
import json
import ast
from knetworking import DataBridgeServer_TCP
from collections import deque

import subprocess

import gradio as gr
from transformers import pipeline
import numpy as np

launch_demo_ttb = subprocess.Popen('python3 ~/demo-lv1-lv2/demo_ttb.py', stdout=subprocess.DEVNULL, shell=True)

print('Waiting for Turtlbot connection...')
server = DataBridgeServer_TCP()

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

theme = gr.themes.Default(primary_hue= gr.themes.colors.emerald, secondary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate).set(
    button_primary_background_fill="*primary_200",
    button_primary_background_fill_hover="*primary_200",
)

css = """
.color_btn textarea  {background-color: #228B22; !important}
"""

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def nlnt (vid_check, prompt, video, history="None", progress=gr.Progress()):
  if vid_check == True:
      return level3_model(prompt, video)
  else:

        # Level 2 Model
        x_dict = ast.literal_eval(main(prompt))
        
        history = deque([])

        state_number = 0
        while x_dict["instruction complete"] == "#ongoing":       # JSON bug here where it doesn't recognize dictionary 
            # {
            #   "state number": "0x2", 
            #   "orientation": 1.29, 
            #   "distance to next point": 0.001, 
            #   "execution length": 1.402, 
            #   "movement message": (0.0, 0.2), 
            #   "instruction complete": "#complete"
            # }
            
            # x_dict is previous state;

            server.send_data('START')

            lin_x, ang_z = x_dict['movement message']
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
        return "Done!"

def level3_model (textbox, ext_video, rpi_video):
    return "level 3: " + textbox

def show_vid (vid_check):
    if vid_check:
      return gr.update(visible=True)
    else:
      return gr.update(visible=False)

def connecting (connect_to, gz_world, ttb_ip):
    if connect_to == "Gazebo":
        launch_gazebo = subprocess.Popen('export TURTLEBOT3_MODEL=burger ; ros2 launch turtlebot3_gazebo empty_world.launch.py', stdout=subprocess.DEVNULL, shell=True)
        return connect_to_gazebo(gz_world)
    else:
        return connect_to_ttb(ttb_ip)

def connect_to_gazebo(gz_world):
  # quickstart here
  return "Connected to Gazebo"

def connect_to_ttb(ttb_ip):
  # quickstart here
  return "Connected to Turtlebot@" + ttb_ip

def launch(commands):
    launcher = subprocess.Popen(commands, shell=True, text = True)
    
    if commands[0:1] == 'ls':
        print(launcher.stdout)

with gr.Blocks(theme=theme, css=css, title = "NLNT Demo") as demo:
    gr.Markdown(
    """
    # Natural Language Ninja Turtle
    A Natural Language to ROS2 Translator for the Turtlebot V3 Burger
    """, elem_id = "title")
    with gr.Tab('Connect'):
      with gr.Row():
        # Gazebo, SSH
        connect_to = gr.Dropdown(label = "Connect to", choices = ["Gazebo", "Actual Turtlebot"], value = "Gazebo")
      with gr.Row():
        gz_world = gr.Textbox(label = "Gazebo World", placeholder = "turtlebot3_empty_world", interactive = True, visible = True)
        ttb_ip = gr.Textbox(label = "Turtlebot IP", interactive = True, visible = True)
      with gr.Row():
        connect_btn = gr.Button(value = "Connect", elem_classes = "color_btn")
      with gr.Row():
        con_status = gr.Textbox(label = "Connection Status", placeholder = "Not Connected.", interactive = False)
        connect = connect_btn.click(fn=connecting, inputs=[connect_to, gz_world, ttb_ip], outputs=con_status)
    with gr.Tab('NLNT'):
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
      with gr.Row():
        ttbt_btn = gr.Button(value = "Run Instruction", elem_classes = "color_btn")
      with gr.Row():
        video = gr.Image(sources=["webcam"], streaming=True, visible=False)
        ckbx = vid_check.select(fn = show_vid, inputs = vid_check, outputs = video)
      with gr.Row():
        with gr.Column():
          status = gr.Textbox(label = "Status", placeholder = "Please enter your prompt.")
          run_nlnt = ttbt_btn.click(fn=nlnt, inputs=[vid_check, prompt, video], outputs=status)


demo.launch()
