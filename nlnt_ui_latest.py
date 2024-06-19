## with CSS & livestreaming (WORKS !)
# documentation for ASR Demo with Transformers : https://www.gradio.app/guides/real-time-speech-recognition
# documentation for livestreaming: https://www.gradio.app/guides/reactive-interfaces

import gradio as gr
from transformers import pipeline
import numpy as np
from inference_gradio import inference
import ast
from knetworking import DataBridgeServer_TCP
from collections import deque
import subprocess
import os
import time
import json
import math
import sys
import quart_funcs

# transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

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

#head = """<link href="https://fonts.googleapis.com/css2?family=Jersey+25+Charted&display=swap" rel="stylesheet">"""
 #   font-family: "Jersey 25 Charted", sans-serif;
head = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Jersey+10&family=Jersey+25+Charted&family=Rubik+Mono+One&family=Russo+One&display=swap" rel="stylesheet">
    """

css = """
h1 {
    text-align: center;
    font-size: 3vw;
    font-family: "Rubik Mono One", monospace;
    font-weight: 400;
    font-style: normal;
    display:block;
}
p {
    text-align: center;
    font-size: 1.2vw;
    display:block;
}
"""

"""
br {
    display: block;
    content: "";
    margin-top: 0;
}
"""

# print('Waiting for Turtlbot connection...')
# ttb_script_path = os.path.join(os.getcwd(),"demo_ttb.py")
# launch_demo_ttb = subprocess.Popen(f'python3 {ttb_script_path}', stdout=subprocess.DEVNULL, shell=True)
server = DataBridgeServer_TCP()
json_listener = DataBridgeServer_TCP(port_number=15000)
latest_super_json = None

import threading

def super_json_listener():

    t = time.time() + 1.0
    x = 0

    global latest_super_json

    while True:

        if time.time() > t:  # used to calculate framerate for debug purposes
            t = time.time() + 1.0
            x = 0
        x += 1

        data = ast.literal_eval(json_listener.receive_data().decode())

        if data == None:
            continue

        latest_super_json = data

data_listener_thread = threading.Thread(target=super_json_listener)
data_listener_thread.start()


def transcribe(audio):

    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def nlnt (vid_check, prompt, video, history="None", progress=gr.Progress()):

    progress(0, desc="Starting...")

    if vid_check == True:
        #return level3_model(prompt, video)
        return level2_model(prompt, progress=gr.Progress())                            # FOR NOW !!! Change later !!!
    else:
        return level2_model(prompt, progress=gr.Progress())

starting_position, starting_orientation = None, None
total_distance, total_rotation = 0.0, 0.0
normalizing_quat = None
deconstruction = None

def generate_breakdown(prompt:str):

    for_breakdown = f"""
<|user|>I need you to break-down this natural language prompt into a series of steps to figure out whether or not it is possible to complete with a robot, the Turtlebot3.

The Turtlebot3 has the following sensors and capabilities:
1.) Moving around on the floor.
    - It cannot move even over small obstacles
    - It can move forwards and turn left or right, but it cannot be made to move backwards by its own. It needs to turn around first and then move backwards.
2.) Scan the immediate area around it with a planar lidar sensor, up to a maximum of 3.5 meters away from the sensor.
3.) Take a photo (maximum resolution of 1080 x 1920) or a video (720 x 1280 @ 30fps).
    - The Turtlebot3 can take a single photo or start a video stream.
    - Cannot take video in the dark.
    - Cannot be swiveled up and down, and is dependent on the robot to be rotated.
4.) Record audio.
    - Also supported while taking a video stream.
5.) Odometer
6.) IMU

The following device flags are available:

1.) take_photo : takes a photo
2.) livestream : sends a livestream of data frames to the server that the robot is connected to
3.) take_video : records a video of the activity
4.) audio_record : records audio
5.) audio : records audio data with either a video or a livestream; defaults to false so you need to indicate it if it needs to be used
6.) no_special_features : used to signal that the command doesn't need the robot to use any of its special sensors or features

The following actions are not supported:

1.) Object manipulation.
    - The Turtlebot3 does not have an arm or any of the sort to touch and move objects.

2.) Streaming media to and from the internet.
    - The Turtlebot3 does not have access to the internet

3.) Self-tilting or rolling on side
    - the robot is required to stay upright all the time

Add delimeters to allow a Python program to parse your answer afterwards, namely:
"<explanation_start>" and "<explanation_end>" to delimit your prompt breakdown,
"### Possibility: True" or "### Possibility: False" to denote prompt completionability,
and finally "<device_flags_start> and "<device_flags_end>" to enable/disable special sensors.

If the prompt is nonsensical, intentionally confusing, or seems to be just spam, flag it as ### Possibility: False

The natural language prompt that I want you to break-down is: {prompt}<|end|>
<|assistant|>
""".strip()

    breakdown = inference(for_breakdown)
    
    prompt_analysis = breakdown[breakdown.rfind('<explanation_start>') + len('<explanation_start>'):breakdown.rfind('<explanation_end>')]
    if prompt_analysis.rfind('### Possibility: True') == -1:

        return {"possibility" : False, "breakdown" : None}
    
    else:

        return {"possibility" : True, "breakdown" : prompt_analysis}

def predict_next_state(prompt: str, breakdown: str, history: str, current_state_number: int):

    string_prompt = f"""
<|user|>
Given the following instruction breakdown, predict the next state. Use "<next_state_start>" and "<next_state_end>" to delineate your answer.

<prompt> {prompt} <prompt_end>
<state_history> {history} <state_history_end>

<breakdown>
{breakdown}
<breakdown_end><|end|>
<|assistant|>
""".strip()

    response = inference(string_prompt)
    response = response[response.rfind('<next_state_start>') + len('<next_state_start>'):response.rfind('<next_state_end>')]

    return response


def level2_model(user_instruction, progress=gr.Progress()):
    
    global starting_position, starting_orientation, total_distance, total_rotation, normalizing_quat, deconstruction

    progress(0, desc="Starting...")
    server.send_data('START')
    first_run = True
    history = deque([])
    state_number = 0
    max_num_states = 1000

    while latest_super_json == None:

        print("Waiting for state json...")
        time.sleep(0.1)

    while True:

        print('===========================================')

        starting_position = latest_super_json["odometry"]["pose_position"]
        starting_orientation = latest_super_json["odometry"]["pose_orientation_quarternion"]
        normalizing_quat = quart_funcs.inverse_quarternion(starting_orientation)
        total_distance, total_rotation = 0.0, 0.0

        reply = generate_breakdown(user_instruction)
        possibility, prompt_breakdown = reply['possibility'], reply['breakdown']

        if not possibility:
            breakdown = prompt_breakdown
            return "Task deemed impossible"

        while state_number_total < max_num_states:

            if len(history) == 0:

                todo = ast.literal_eval(predict_next_state(user_instruction, prompt_breakdown, '[ None ]', state_number))
                history.append(todo)

                if max_num_states >= 1000:
                    max_num_states = todo['total states']

            else:

                todo = ast.literal_eval(predict_next_state(user_instruction, prompt_breakdown, list(history), state_number))
                history.append(todo)

            state_number += 1
            instruction = str([todo['twist message'], todo['execution length'], False])
            server.send_data(instruction)
            received = server.receive_data().decode()
            print(received)

            if state_number >= max_num_states:

                instruction = str([[0.0, 0.0], 0.0, True])
                server.send_data(instruction)

                return "Instruction accomplished. Waiting for next instruction."

            if len(history) > 5: history.popleft() # keep history at depth = 5


def level3_model (prompt, video):
    return "level 3: " + prompt

#def inference(prompt):
    # should send prompt to the model
    
def show_vid (vid_check):
    if vid_check:
      return gr.update(visible=True)
    else:
      return gr.update(visible=False)

def compute_distance(coord1: list, coord2: list):

    total = 0
    for i in range(3):
        total += (coord2[i] - coord1[i])**2
    
    return math.sqrt(total)


def quaternion_to_yaw(x, y, z, w): # Generated by GPT-4

    """
    Convert a quaternion into yaw (rotation around z-axis in radians)
    """
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

def yaw_difference(quaternion1, quaternion2): # Generated by GPT-4
    """
    Calculate the difference in yaw between two quaternions
    """
    yaw1 = quaternion_to_yaw(*quaternion1)
    yaw2 = quaternion_to_yaw(*quaternion2)
    
    # Calculate the difference and adjust for the circular nature of angles
    difference = yaw2 - yaw1
    difference = (difference + math.pi) % (2 * math.pi) - math.pi
    
    return difference

def normalize_radians(float_num):
    return (float_num + np.pi) % (2 * np.pi) - np.pi

#def prompt_breakdown():

### add webserver to host data from turtlebot
from fastapi import FastAPI
from random import randrange

# demo.launch()

webserver = FastAPI()
print("launching webserver")

@webserver.get("/hello")
def read_root():
    return {"Hello": "World"}


@webserver.get("/metadata")

def read_metadata():

    global total_distance, total_rotation, deconstruction
    # TODO, format the data from server = DataBridgeServer_TCP() to get the different metadata
    battery_percentage = "100%"

    return {
        "total_distance_traveled": total_distance, 
        "total_degrees_rotated": total_rotation,
        "battery_percentage": battery_percentage,
        "prompt": deconstruction
     }

#def deconstruct():
#    #deconstruction = ast.literal_eval(read_metadata())["prompt"]
#    deconstruction = read_metadata()
#    if deconstruction:
#        return gr.update(value = deconstruction)
#    else:
#        return gr.update(value = "Nothing here yet :)")


with gr.Blocks(theme=theme, css=css, title = "NLNT Demo",js="metadata.js", head=head) as demo:
    with gr.Row():
        gr.Markdown(
        """
        # Natural Language Ninja Turtle
        A Natural Language to ROS2 Translator for the Turtlebot V3 Burger
        """)
    with gr.Row():
        vid_check = gr.Checkbox(label = "Connect Live Video")
    with gr.Row(equal_height=True):
        # audio = gr.Audio(sources=["microphone"])
        prompt = gr.Textbox(label = "Instruction", placeholder = "move 1.5 meters forward", interactive = True)
    with gr.Row():
        # clr_audio = gr.ClearButton(value = "Clear Audio", components = [audio])
        # transcribe_btn = gr.Button(value = "Transcribe")
        clr_text = gr.ClearButton(value = "Clear Text", components = [ prompt])# audio
        # transcription = transcribe_btn.click(fn=transcribe, inputs=audio, outputs=prompt)
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
        #deconstruct_btn = gr.Button(value = "Show Prompt Breakdown")
    #    prompt_breakdown = gr.Textbox(label = "Prompt Breakdown", value = read_metadata(), interactive = False, every = 1)
        #run_prompt_brd = deconstruct_btn.click(fn = deconstruct(), inputs = [], outputs = prompt_breakdown)
    #with gr.Row():
    #    total_dist = gr.Textbox(value = next(total_distance()), label = "Total Distance Traveled", interactive=False)
    #    total_rot = gr.Textbox(value = total_rotation(), label = "Total Degrees Rotated", interactive=False, every=0.5)
    with gr.Row():
        gr.HTML("""
        <div style='display: flex; flex-direction: column; height: auto; width: 100%;'>
            <div style='display: flex; flex-direction: column; padding: 20px; background-color: #292524;'>
                <span style='font-weight:bold;display:flex;justify-content:start;'>Prompt Breakdown</span> 
                <span id="prompt" style='background-color: #44403C; padding: 20px;margin-top: 20px'>0</span>
            </div>
        </div>
        <br>
        <br>
        <div style='display: flex; flex-direction: column; height: auto; width: 100%;'>
            <div style='display:flex;justify-content:space-around;'>
                <div>
                    <span style='font-weight:bold;'>Total Distance Traveled:</span> <span id="total-dist">0</span>
                </div>
                <div>
                    <span style='font-weight:bold;'>Total Degrees Rotated:</span> <span id="total-degs">0</span>
                </div>
                 <div>
                    <span style='font-weight:bold;'>Battery Percentage:</span> <span id="batt">100%</span>
                </div>

            </div>
        </div>
        """)

app = gr.mount_gradio_app(webserver, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",port=8001,log_level="critical")
