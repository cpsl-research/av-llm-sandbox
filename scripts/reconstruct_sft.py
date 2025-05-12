import json
import itertools

def reconstruct_to_conversations(input_json_path, output_json_path):
    """
    Reads a JSON dataset from `input_json_path`, and for each entry:
      - Gathers the 'new_question' + 'new_answer'
      - Gathers the 'image_path' 
      - Iterates over every item in 'correct_long_reasoning' and 'correct_short_reasoning'
      - For each reasoning item, creates a new record with:
          1) ID
          2) images
          3) conversations (2-element array: one from 'human', one from 'gpt')
    Writes the resulting list of records into `output_json_path`.
    """

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_records = []

    for original_id, entry in data.items():
        # Extract fields (use .get() to safely handle missing fields)
        images = entry.get("image_paths", {})
        correct_long_list = entry.get("correct_long_reasoning", [])
        correct_medium_list = entry.get("correct_medium_reasoning", [])
        correct_short_list = entry.get("correct_short_reasoning", [])
        long_actions = entry.get("long_action", "")
        medium_actions = entry.get("medium_action", "")
        short_actions = entry.get("short_action", "")
        gt_action = entry.get("gt_action", "")
        gt_lat = gt_action[0]
        gt_lon = gt_action[1]
        correct_long_action = []
        correct_medium_action = []
        correct_short_action = []
        for long_action in long_actions:
            if (long_action[0] in gt_lat) and (long_action[1] in gt_lon):
                correct_long_action.append(long_action)
        for medium_action in medium_actions:
            if (medium_action[0] in gt_lat) and (medium_action[1] in gt_lon):
                correct_medium_action.append(medium_action)
        for short_action in short_actions:
            if (short_action[0] in gt_lat) and (short_action[1] in gt_lon):
                correct_short_action.append(short_action)
        # We'll create new samples for each item in correct_long_reasoning + correct_short_reasoning.
        # For instance, if correct_long_reasoning has 2 items and correct_short_reasoning has 3,
        # we will produce 5 new conversation records.
        
        # Combine them into a list of (reasoning_text, type_label),
        # just so we know if it's from long or short if needed. (You can ignore 'type_label' if not required.)
        reasoning_candidates = [(r, "long") for r in correct_long_list] + \
                               [(r, "medium") for r in correct_medium_list] + \
                               [(r, "short") for r in correct_short_list]

        # If no correct reasoning is found, skip (or handle differently).
        if not reasoning_candidates:
            continue

        sample_index = 1
        for reasoning_text, reasoning_type in reasoning_candidates:
            # Construct a new ID
            new_id = f"{original_id}_{sample_index}"
            sample_index += 1

            # "human" message combines the new_question plus the additional instructions you mentioned
            human_message_value = (
                "You are provided with six synchronized camera images captured from the ego-vehicle in the following order: "
                "rear, rear-left, and rear-right, front, front-left, front-right. "
                "<task> First, formulate a concise reasoning context that integrates scene perception and short-term motion prediction. "
                "Next, derive the appropriate driving decision and return it **exactly** as a tuple in the form ('<LATERAL>', '<LONGITUDINAL>'). </task> "
                "<meta action pool> Permissible lateral actions: VEER_LEFT | VEER_RIGHT | CHANGE_LANE_LEFT | CHANGE_LANE_RIGHT | STRAIGHT | TURN_LEFT | TURN_RIGHT. "
                "Permissible longitudinal actions: ACCELERATE | MAINTAIN | DECELERATE | REVERSE. </meta action pool>"
            )
            
            # "gpt" message includes the chosen reasoning + final answer
            # We place a newline and then "### Answer: {answer}"
            if reasoning_type == 'long':
                answer = correct_long_action.pop(0)
            elif reasoning_type == 'medium':
                answer = correct_medium_action.pop(0)
            elif reasoning_type == 'short':
                answer = correct_short_action.pop(0)
            gpt_message_value = f"<think> {reasoning_text} </think> \n\n### Correct action: {answer}"

            conversations = [
                {
                    "from": "human",
                    "value": human_message_value
                },
                {
                    "from": "gpt",
                    "value": gpt_message_value
                }
            ]

            record = {
                "ID": new_id,
                "images": images,
                "conversations": conversations
            }

            new_records.append(record)

    # Save to output JSON
    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(new_records, out_f, indent=2, ensure_ascii=False)

    print(f"Reconstructed {len(new_records)} new records into {output_json_path}.")

# Example usage:
if __name__ == "__main__":
    input_file = "reasoning_pair_data.json"   # your single-sample or multi-sample JSON
    output_file = "sft_data.json"
    reconstruct_to_conversations(input_file, output_file)
