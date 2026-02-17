import os
import datetime

from argparse import ArgumentParser
from agents import CoTAgent, ReflexionStrategy
from arguments import ReflexionParams
from utils import summarize_trial, log_trial, save_agents
from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt
from fewshots import COT, COT_REFLECT


def Caption_Reflexion(rp:ReflexionParams):
    '''
    Reflexion for 3D object captioning
    '''
    agents = [CoTAgent(prompt=rp.prompt,
                       obj_path=os.path.join(rp.objs_dir,obj_name),

                       agent_prompt=cot_agent_prompt if rp.strategy == ReflexionStrategy.NONE else cot_reflect_agent_prompt,
                       reflect_prompt=cot_reflect_prompt,

                       cot_examples=COT,
                       reflect_examples=COT_REFLECT,

                       ) for obj_name in os.listdir(rp.objs_dir)]
    
    log = ''

    for trial in range(1,rp.n+1):
        for agent in [a for a in agents if not a.is_correct()]:
            agent.run(reflexion_strategy = rp.strategy)
        log += log_trial(agents, trial)
        correct, incorrect = summarize_trial(agents)
        print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    log_filename = f"{len(agents)}_objects_{rp.n}_trials_{timestamp}.txt"
    log_path = os.path.join('./Log', rp.strategy.value, log_filename)

    agents_folder = f"{len(agents)}_agents_{timestamp}"
    agents_path = os.path.join('./Log', rp.strategy.value, agents_folder)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'w') as f:
        f.write(log)
    save_agents(agents, agents_path)





if __name__ == "__main__" :
    print("Argument parsing...")
    parser = ArgumentParser(description="Reflexion params")
    rp=ReflexionParams(parser)      

    print("Run Captioning...")
    Caption_Reflexion(rp=rp)