import openai
from openai import OpenAI
import json
import base64
import ast
import os


def flatten_frames(src_dataset):
    """
    Return a list of dicts — one per frame — with keys:
        frame_id, meta_action (str), waypoints_2d (str), image_paths (dict)
    Scene/agent IDs are discarded.
    """
    out = []

    for scene in src_dataset.values():
        for agent in scene.values():
            for fid, frame in agent.items():
                # ---------- 1) meta_action  ----------
                lat_vals = []
                lon_vals = []
                
                for rec in frame["meta_actions"].values():
                    if rec is not None:
                        lat, lon = rec.get("lateral"), rec.get("longitudinal")
                    lat_vals.append(lat)
                    lon_vals.append(lon)
                meta_action_str = [lat_vals, lon_vals]
                '''  
                meta_action = frame["meta_actions"]['dt_1.00']
                if meta_action is not None:
                    lat, lon = meta_action.get("lateral"), meta_action.get("longitudinal")
                    meta_action_str = str([lat, lon])
                else:
                    meta_action_str = str([None, None])
                '''
                #majority_lat = Counter(lat_vals).most_common(1)[0][0] if lat_vals else None
                #majority_lon = Counter(lon_vals).most_common(1)[0][0] if lon_vals else None
                #meta_action_str = str([majority_lat, majority_lon])

                # ---------- 2) waypoints_2d  ----------
                
                # sort by the integer part of 'dt_X'
                tuples = []
                for wp in frame.get("waypoints_3d", {}).values():
                    if wp and len(wp) >= 3:
                        x, _, z = wp
                        tuples.append((round(x, 1), round(z, 1)))
                waypoints_str = str(tuples)
                
                
                speeds = [st["speed"] for st in frame["agent_state"].values()
                          if "speed" in st]
                speed_val = round(speeds[0],1) if speeds else None
                # ---------- 3) pack result ----------
                out.append({
                    "frame_id":      fid,
                    "image_paths":   frame["image_paths"],  # untouched
                    "meta_action":   meta_action_str,
                    "waypoints_2d":  waypoints_str,
                    "speed": speed_val,
                })

    return out
    


def encode_image(path_to_image: str) -> str:
    """
    Reads an image file from disk and returns a base64-encoded string (JPEG).
    """
    with open(path_to_image, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8")

def build_autonomous_driving_prompt(camera_info_dict, use_base64=False):
    """
    Build a concise prompt for an LLM to perform step‑by‑step scene reasoning
    from six surround‑view images.  The model must output three numbered
    sections—Perception, Prediction, Road—without prescribing any action.
    """

    # -------- System role ---------------------------------------------------
    system_prompt = """
    You are an autonomous‑driving vision analyst.
    Think step‑by‑step and output ONLY the three sections below.
    Do NOT suggest steering or speed commands.
    """.strip()

    # -------- User instructions & demo --------------------------------------
    user_prompt = """
    ### Task
    From the camera images of six views, give a step-by-step thinking process:

    1) **Detected Objects** – main vehicles, pedestrains, traffic lights, and road signs, etc., their state, lane/relative position, ≈distance (m).  
    2) **Predicted Movements** – likely next motion for each key object.  
    3) **Road Condition Ahead** – geometry in front of the ego car (e.g., “straight & clear”, “tight left‑hand curve”).


    ### Example

    Camera Views (sample):
    • front‑left – parked cars at curb  
    • front       – blue sedan 30 m ahead, braking  
    • front‑right – clear sidewalk  
    • back‑left   – black SUV closing in left lane  
    • back        – clear  
    • back‑right  – cyclist 20 m behind

    **Model Output**
    1) Blue sedan ahead braking ~30 m on the front view; black SUV left‑rear closing fast; cyclist right‑rear steady ~20 m.  
    2) Sedan will slow further; SUV may merge right; cyclist continues straight.  
    3) Road ahead straight and unobstructed.

    ---

    Now analyse the new scene:

    """.lstrip()

    # -------- Assemble messages --------------------------------------------
    system_content    = [{"type": "text", "text": system_prompt}]
    user_content      = [{"type": "text", "text": user_prompt}]
    assistant_content = [{"type": "text", "text": "step-by-step reaosoning:"}]   # model’s first token cue

    # Add camera views
    user_content.append({"type": "text", "text": "Camera Views:"})
    if use_base64:
        for view, path in camera_info_dict.items():
            
            user_content.append({"type": "text", "text": f"{view}:"})
            encoded = encode_image(path)                 # assume helper exists
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
            })
    else:
        for view, desc in camera_info_dict.items():
            user_content.append({"type": "text", "text": f"{view}: {desc}"})

    return system_content, user_content, assistant_content

def generate_reasoning_and_action(client, camera_dict, use_base64=False):
    """
    1. Optionally encode images or prepare text descriptions.
    2. Build a prompt using that info.
    3. Call OpenAI GPT to generate chain-of-thought reasoning + final action.
    """

    # Build the prompt
    system_content, user_content, assistant_content = build_autonomous_driving_prompt(
        camera_info_dict=camera_dict,
        use_base64=use_base64
    )

    # Make the ChatCompletion call
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        temperature=1.0,
        max_tokens=2048
    )

    # Extract the GPT response
    answer = response.choices[0].message.content
    return answer

def build_verify_prompt(image_path, reasoning_context: str, speed: float, add_image=True):
    """
    Construct a concise, high‑signal prompt for an LLM that returns a driving
    meta‑action pair and a confidence score from 0‑5.
    """
    system_prompt = """
    You are an autonomous‑driving assistant.  
    Input: camera images from six views of ego + reasoning context + ego speed.  
    Task: decide what the ego vehicle must do **right now**.

    Output format (no extra text):
    (['<LATERAL>', '<LONGITUDINAL>'], <CONFIDENCE>)    # confidence ∈ 0‑5

    Allowed meta‑actions
      • Lateral:   VEER_LEFT | VEER_RIGHT | CHANGE_LANE_LEFT | CHANGE_LANE_RIGHT
                  STRAIGHT  | TURN_LEFT  | TURN_RIGHT
      • Longitudinal: ACCELERATE | MAINTAIN | DECELERATE | REVERSE

    Decision rules
      1. Avoid collisions; keep safe gaps.
      2. Stay on drivable surface.
      3. keep reasonable speed when road is clear
      4. Turning with low speed and deceleration.

    Considerations(IMPORTANT)
    • Lateral:
      1. Evaluate the road geometry first. 
      If the main road curves ahead and continuing in the current direction would cause the ego vehicle to leave the drivable area, choose an action that follows the curve to keep the vehicle on the main road (do not output STRAIGHT in such cases).
      2. Then account for pedestrians, vehicles, or other obstacles and steer to avoid
        any potential collision.

    • Longitudinal:
      1. Begin with the current speed.
      2. Decide on a change:
        - If the vehicle is moving too slowly for conditions, ACCELERATE.  
        - If it’s too fast or needs extra margin, DECELERATE. 
        - Otherwise, MAINTAIN the present speed.

   
    """

    # Assemble chat messages
    user_content = []
    if add_image:
        user_content.append({"type": "text", "text": "Camera Views:"})

        for view, path in image_path.items():
            
            user_content.append({"type": "text", "text": f"{view}:"})
            encoded = encode_image(path)                 # assume helper exists
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
            })
    
    system_content = [{"type": "text", "text": system_prompt.strip()}]
    user_content.append({"type": "text", "text": f"Reasoning context:\n{reasoning_context}"})
    user_content.append({"type": "text", "text": f"Current ego speed: {speed}m/s"})
    
    assistant_content = [{"type": "text", "text": "Meta‑action and confidence:"}]

    return system_content, user_content, assistant_content

def generate_final_action(client, image_path, reasoning_context, speed):
    """
    1. Optionally encode images or prepare text descriptions.
    2. Build a prompt using that info.
    3. Call OpenAI GPT to generate chain-of-thought reasoning + final action.
    """

    # Build the prompt
    system_content, user_content, assistant_content = build_verify_prompt(
        image_path,
        reasoning_context,
        speed
    )

    # Make the ChatCompletion call
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        temperature=0.2,
        max_tokens=512
    )

    # Extract the GPT response
    answer = response.choices[0].message.content
    return answer

def build_refine_prompt(
    camera_info_dict,
    reasoning_context,
    use_base64=False
):
    """
    Builds a text prompt for GPT by either inserting
    base64-encoded images or textual descriptions.
    """
    # Start with a short context
    system_prompt = '''
        You are an advanced autonomous driving assistant specialized in 
        concise reasoning about camera images to determine the correct driving action.
    '''

    # The user message includes demonstration data + the new scenario
    user_prompt = '''
    Below is data from an autonomous driving scenario. You are provided with:
    1) camera images from six views.
    2) A current chain-of-thought reasoning.

    Goal: Produce a shorter, more concise version of the reasoning that only includes details 
    necessary for deriving the final driving action. Remove unnecessary analysis, extraneous 
    tangents, or repeated points. Rephrase any sentences to be more succinct while preserving 
    meaning.

    Instructions:
    1. Review the camera images and the current reasoning.
    2. Delete or omit irrelevant details that do not influence the final driving decision.
    3. Rephrase what's left so it's concise but still logically consistent.
    
    '''
    system_content = [{"type": "text", "text": system_prompt}]
    user_content = [{"type": "text", "text": user_prompt}]
    assistant_content = [{"type": "text", "text": "Concise reasoning:"}]
    user_content.append({"type": "text", "text": "Camera Views:\n"})
    if use_base64:
        # Insert base64 data (GPT-3.5/4 standard models typically cannot decode, but let's show it anyway)
        for view, image_path in camera_info_dict.items():
            if view == 'CAM_FRONT':
                user_content.append({"type": "text", "text": f"{view}:"})
                base64_image = encode_image(image_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })                    
    else:
        # Insert textual descriptions
        for view, desc in camera_info_dict.items():
            user_content.append({"type": "text", "text": f"{view}:"})
            user_content.append({"type": "text", "text": desc})

    user_content.append({"type": "text", "text": "Current reasoning chain:\n"})
    user_content.append({"type": "text", "text": reasoning_context})
    

    return system_content, user_content, assistant_content

def generate_concise_reasoning(client, camera_dict, reasoning_context, use_base64=False):
    """
    1. Optionally encode images or prepare text descriptions.
    2. Build a prompt using that info.
    3. Call OpenAI GPT to generate chain-of-thought reasoning + final action.
    """

    # Build the prompt
    system_content, user_content, assistant_content = build_refine_prompt(
        camera_info_dict=camera_dict,
        reasoning_context=reasoning_context,
        use_base64=use_base64
    )

    # Make the ChatCompletion call
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        temperature=1,
        max_tokens=2048
    )

    # Extract the GPT response
    answer = response.choices[0].message.content
    return answer


client = OpenAI(
    # KEEP IT PRIVATE!
    api_key="sk-***",
)
max_retries = 5
retry_delay = 2.0

json_path = 'dataset_train.json'
with open(json_path, 'r') as file:
    data = file.read()
index_dict = json.loads(data)
dataset = index_dict['dataset']
results = flatten_frames(dataset)

output_json_path = 'reasoning_pair_data.json'
counter = 0
checkpoint_file = 'checkpoint.txt'
start_idx = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        start_idx = int(f.read().strip())

for result in results:
    if counter < start_idx:
        counter = counter+1
        continue
    if counter % 5 != 0:
        counter = counter+1
        continue
    data_entry = {}
    
    image_path = result['image_paths']
    
    data_entry['image_paths'] = image_path
    data_entry['correct_long_reasoning'] = []
    data_entry['correct_medium_reasoning'] = []
    data_entry['correct_short_reasoning'] = []
    data_entry['wrong_long_reasoning'] = []
    data_entry['wrong_medium_reasoning'] = []
    data_entry['wrong_short_reasoning'] = []
    data_entry['long_action'] = []
    data_entry['medium_action'] = []
    data_entry['short_action'] = []
    data_entry['gt_action'] = result['meta_action']
    data_entry['speed'] = result['speed']
    gt_action = result['meta_action']
    speed = result['speed']
    while len(data_entry['correct_long_reasoning'])+len(data_entry['wrong_long_reasoning']) < 2:
        regular_reasoning = generate_reasoning_and_action(client, image_path, use_base64=True)
        regular_action, regular_confidence = ast.literal_eval(generate_final_action(client, image_path, regular_reasoning, speed))
        regular_lar, regular_lon = regular_action
        concise_reasoning_1 = generate_concise_reasoning(client, image_path, regular_reasoning, use_base64=True)
        concise_action_1, concise_confidence_1 = ast.literal_eval(generate_final_action(client, image_path, concise_reasoning_1, speed))
        concise_lar_1, concise_lon_1 = concise_action_1
        concise_reasoning_2 = generate_concise_reasoning(client, image_path, concise_reasoning_1, use_base64=True)
        concise_action_2, concise_confidence_2 = ast.literal_eval(generate_final_action(client, image_path, concise_reasoning_2, speed))
        concise_lar_2, concise_lon_2 = concise_action_2
        if regular_lar in gt_action[0] and regular_lon in gt_action[1] and regular_confidence >= 3:
            data_entry['correct_long_reasoning'].append(regular_reasoning)
        else:
            print('Wrong long reasoning:')
            print(regular_action)
            print(regular_confidence)
            print('---')
            data_entry['wrong_long_reasoning'].append(regular_reasoning)
        
        if concise_lar_1 in gt_action[0] and concise_lon_1 in gt_action[1] and concise_confidence_1 >= 3:
            data_entry['correct_medium_reasoning'].append(concise_reasoning_1)
        else:
            print('Wrong Concise reasoning:')
            print(concise_action_1)
            print('---')
            data_entry['wrong_medium_reasoning'].append(concise_reasoning_1)
   
        if concise_lar_2 in gt_action[0] and concise_lon_2 in gt_action[1] and concise_confidence_2 >= 3:
            data_entry['correct_short_reasoning'].append(concise_reasoning_2)
        else:
            print('Wrong Concise reasoning:')
            print(concise_action_2)
            print('---')
            data_entry['wrong_short_reasoning'].append(concise_reasoning_2)

        data_entry['long_action'].append(regular_action)
        data_entry['medium_action'].append(concise_action_1)
        data_entry['short_action'].append(concise_action_2)

    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    
    data[counter] = data_entry
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)
    
    counter = counter+1
    
    with open(checkpoint_file, 'w') as f:
        f.write(str(counter))
    
    if counter >= 15000:
        break
