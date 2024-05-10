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
import quart_funcs

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

def level2_model(user_instruction, progress=gr.Progress()):
    
    global starting_position, starting_orientation, total_distance, total_rotation, normalizing_quat, deconstruction

    progress(0, desc="Starting...")
    server.send_data('START')
    first_run = True
    history = deque([])
    state_number = 0

    while latest_super_json == None:

        print("Waiting for state json...")
        time.sleep(0.1)

    while True:

        print('===========================================')

        starting_position = latest_super_json["odometry"]["pose_position"]
        starting_orientation = latest_super_json["odometry"]["pose_orientation_quarternion"]
        normalizing_quat = quart_funcs.inverse_quarternion(starting_orientation)
        total_distance, total_rotation = 0.0, 0.0
        
        if first_run:

            print("Parsing starting prompt...")
            progress(0, desc="Parsing starting prompt...")
            
        # First Iteration
            prompt = f"""
    Your task is to pilot a Turtlebot3 and predict the next state give a history sequence. First, predict whether or not the task is doable.
    The prompt is: <prompt> <PROMPT> </prompt>. Add delimeters to outline your solution and your answer, "<deconstruction_start>" and "<deconstruction_end>" for your step-by-step breakdown of the natural language prompt, and "<possibility_start>" and "<possibility_end>" to delineate your possibility answer and make it easy to parse in Python.

    Information about the task will be given in JSONs, and it is expected that you will also give your answers in a JSON format.<|end|>""".strip()
            
            prompt = prompt.replace("<PROMPT>", user_instruction)
            predicted = inference(prompt)
            
            # Check whether or not instruction is possible

            possible_start = predicted.rfind('<possibility_start>')
            possible_end = predicted.rfind('<possibility_end>') - 1

            deconstruction = predicted[predicted.rfind("<deconstruction_start>"):predicted.rfind("<deconstruction_end>")].replace("<deconstruction_start>","").strip()
            possible = ast.literal_eval(predicted[possible_start:possible_end].replace("<possibility_start>","").strip())
            print("CoT response received.")

            if possible:

                # Get Expected Number of States
                print("Task is possible.")
                stepnums_start = predicted.rfind('Thus, it will take ')
                stepnums_end = predicted.rfind(' states to complete.') - 1
                first_run = False                                                       # first run done!

            else:
                # Instruction is impossible to accomplish
                return "Task deemed impossible. Waiting for your next instruction."     # stop inferencing here!

        else:
            # Subsequent Inferences once Task is deemed possible
            x_dict = {"instruction complete" : "#ongoing"}
            current_it = 1

            while x_dict["instruction complete"] == "#ongoing":

                if history != deque([]):
                    new_prompt = predicted + f"""The current state history is: {[i for i in history]}. Predict the next state. Use "<answer_start>" and "<answer_end>" to delineate the answer.<|end|>"""
                else:
                    new_prompt = predicted + f"""The current state history is: [ None ]. Predict the next state. Use "<answer_start>" and "<answer_end>" to delineate the answer.<|end|>"""

                # Predict the next state
                predicted = inference(new_prompt)
                # Format the response to make life easier
                next_state_start = predicted.rfind("<answer_start>")
                next_state_end = predicted.rfind("<answer_end>") - 1

                next_state = predicted[next_state_start:next_state_end].replace("<answer_start>","").strip()        
                x_dict = ast.literal_eval(next_state)

                # Send predicted info to Turtlebot
                print('Predicted:', x_dict)
                lin_x, ang_z = x_dict['twist message']
                dt = x_dict['execution length']
                code = 1 if x_dict['instruction complete'] == '#complete' else 0
                expected_states = x_dict['total states']
                
                progress_status = current_it / (int(str(expected_states),16)+1)

                if progress_status < 1:
                    progress(progress_status, desc=f'Ongoing... Next Action: ({str(lin_x)}, {str(ang_z)}, {str(dt)})')
                else:
                    progress(0.99, desc=f'Ongoing... Next Action: ({str(lin_x)}, {str(ang_z)}, {str(dt)})')
                
                mess = str([lin_x, ang_z, dt, code])

                print("Sending:", mess)
                server.send_data(mess.encode())                                     # send the predicted action to turtlebot
                print('------')
                print("Waiting on Turtlebot reply...")
                data = ast.literal_eval(server.receive_data().decode())             # receive actual action from turtlebot

                x_dict['state number'] = hex(state_number)
                x_dict['orientation'] = data['orientation']
                total_distance = data['distance_traveled']
                total_rotation = data['orientation_diff']

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
                current_it += 1
                    
            progress(1, desc="Movement done!")
            startiing_orientation, starting_position = None, None
            deconstruction = None
            return "Instruction accomplished. Waiting for next instruction."

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
    with gr.Row():
        #deconstruct_btn = gr.Button(value = "Show Prompt Breakdown")
        prompt_breakdown = gr.Textbox(label = "Prompt Breakdown", value = read_metadata(), interactive = False, every = 1)
        #run_prompt_brd = deconstruct_btn.click(fn = deconstruct(), inputs = [], outputs = prompt_breakdown)
    #with gr.Row():
    #    total_dist = gr.Textbox(value = next(total_distance()), label = "Total Distance Traveled", interactive=False)
    #    total_rot = gr.Textbox(value = total_rotation(), label = "Total Degrees Rotated", interactive=False, every=0.5)
    with gr.Row():
        gr.HTML("""
        <div style='display: flex; flex-direction: column; height: 100px; width: 100%;'>
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
                <div>
                    <span style='font-weight:bold;'>Prompt Breakdown:</span> <span id="prompt">0</span>
                </div>
            </div>
        </div>
        """)

app = gr.mount_gradio_app(webserver, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",port=8000)
