import os
import joblib
from typing import List

def summarize_trial(agents):
    correct = [a for a in agents if a.is_correct()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect

def remove_fewshot(prompt_input:List):
    if isinstance(prompt_input, list):
        if len(prompt_input) == 0:
            return ""
        first_text = prompt_input[0] if isinstance(prompt_input[0], str) else str(prompt_input[0])
        last_text = prompt_input[-1] if isinstance(prompt_input[-1], str) else str(prompt_input[-1])
        # imgs
        image_count = len(prompt_input) - 2
        images_placeholder = "\n".join(["<Image>"] * image_count) if image_count > 0 else ""
        full_text = first_text + "\n" + images_placeholder + "\n" + last_text
    else:
        full_text = prompt_input

    if 'Here are some examples:' in full_text and '(END OF EXAMPLES)' in full_text:
        prefix = full_text.split('Here are some examples:')[0]
        suffix = full_text.split('(END OF EXAMPLES)')[1]
        return prefix.strip('\n').strip() + '\n' + suffix.strip('\n').strip()
    else:
        return full_text

def log_trial(agents, trial_n):
    correct, incorrect = summarize_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += remove_fewshot(agent._build_agent_prompt()) 

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += remove_fewshot(agent._build_agent_prompt()) 

    return log

def save_agents(agents, dir: str):
    '''
    save the final agents
    '''
    os.makedirs(dir, exist_ok=True)
    for i, agent in enumerate(agents):
        joblib.dump(agent, os.path.join(dir, f'{i}.joblib'))