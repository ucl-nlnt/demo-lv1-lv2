from simple_chalk import chalk
import requests
import json
import argparse
import time


def main(prompt, history="None"):
    #print(chalk.green("UCL NLNT Level 1 and 2 Inference Terminal"))

    url = 'http://10.158.18.253:8000/send-prompt'
    # url = 'http://localhost:8000/send-prompt'

    # supply username/prompt if they are not provided
    #if username is None:
    #    username = input(chalk.yellow(
    #        "Enter your username (red,gab,mara,rica): "))
    if prompt is None:
        prompt = input(chalk.yellow("Enter your prompt: "))

    # TODO: format prompt

    prompt = f"""You are given the task to act as a helpful agent that pilots a robot. Given the the frame history, determine the next frame in the series given the prompt and the previous state. Expect that any given data will be in the form of a JSON, and it is also expected that your reply will be also in JSON format. Set the 'completed' flag to '#complete' when you are done, otherwise leave it as '#ongoing'. Here is your task: " + prompt + " | History: [ {str(history)} ] ### Answer:"""

    headers = {'Content-Type': 'application/json'}
    data = {'content': prompt}

    try:
        #print(chalk.magenta("sending prompt to inference server..."))
        start_time = time.time()

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(chalk.magenta("response received!"),
        #      f'{round(elapsed_time, 2)} seconds')

        json_object = response.json()
        json_formatted_str = json.dumps(json_object, indent=2)

        json_start = json_formatted_str.rfind('{')
        json_end = json_formatted_str.rfind('</s>') - 1

        to_return = json_formatted_str[json_start:json_end].replace("'", '"').strip()
        #print(to_return)
        #print(json_formatted_str)
        #json_response = json.loads(to_return)
        #print(json_response)
        #print(json_response["instruction complete"])
        #print(json_response["generated"])
        #gen = json_response["generated"]
        #split_json = gen.split('### Answer: ')
        #print(split_json[-1])
        #returned = split_json[-1].replace("</s>", "")
        #print(returned)
        #ret_json = json.dumps(returned, indent=2)
        #ret = "{'state number': '0x0', 'orientation': 0.0, 'distance to next point': 0.82, 'execution length': 4.208, 'movement message': (0.2, 0.0), 'instruction complete': '#ongoing'}"
        #ret_json = json.dumps(returned, indent=2)
        #ret_dt = json.loads(ret_json)
        #print(ret_dt)
        #print(ret_dt["instruction complete"])
        
        return to_return
    
    except requests.RequestException as e:
        print("Error:", e)
        return "Error: " + e
        #return None

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--username', type=str,
                        help='Your username', required=False)
    parser.add_argument('--prompt', type=str,
                        help='Your prompt', required=False)
    args = parser.parse_args()
    main(args.prompt, "None")
