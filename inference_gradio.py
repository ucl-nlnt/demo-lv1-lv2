from simple_chalk import chalk
import requests
import json
import argparse
import time


def inference(prompt):
    #print(chalk.green("UCL NLNT Level 1 and 2 Inference Terminal"))

    url = 'http://10.158.18.253:11000/send-prompt'
    # url = 'http://localhost:8000/send-prompt'

    # supply username/prompt if they are not provided
    #if username is None:
    #    username = input(chalk.yellow(
    #        "Enter your username (red,gab,mara,rica): "))
    if prompt is None:
        prompt = input(chalk.yellow("Enter your prompt: "))

'''
    # TODO: format prompt

    if history == "None":
        prompt = f"""You are given the task to act as a helpful agent that pilots a robot. Given the the frame history, determine the next frame in the series given the prompt and the previous state. Expect that any given data will be in the form of a JSON, and it is also expected that your reply will be also in JSON format. Set the 'completed' flag to '#complete' when you are done, otherwise leave it as '#ongoing'. Here is your task: {prompt} | History: [ None ] ### Answer:"""
    else:
        prompt = f"""You are given the task to act as a helpful agent that pilots a robot. Given the the frame history, determine the next frame in the series given the prompt and the previous state. Expect that any given data will be in the form of a JSON, and it is also expected that your reply will be also in JSON format. Set the 'completed' flag to '#complete' when you are done, otherwise leave it as '#ongoing'. Here is your task: {prompt} | History: [{str(history)}] ### Answer:"""

    print('================================================')
    print(prompt)
    print('================================================')
''' 
    headers = {'Content-Type': 'application/json'}
    data = {'content': prompt}

    try:
        start_time = time.time()

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        end_time = time.time()
        elapsed_time = end_time - start_time

        json_object = response.json()
        json_formatted_str = json.dumps(json_object, indent=2)


        #json_start = json_formatted_str.rfind('{')
        #json_end = json_formatted_str.rfind('</s>') - 1

        #to_return = json_formatted_str[json_start:json_end].replace("'", '"').strip()
        
        return json_formatted_str
    
    except requests.RequestException as e:
        print("Error:", e)
        return

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--username', type=str,
                        help='Your username', required=False)
    parser.add_argument('--prompt', type=str,
                        help='Your prompt', required=False)
    args = parser.parse_args()
    inference(args.prompt, "None")
