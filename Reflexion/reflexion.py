import os
import datetime
import pickle

from argparse import ArgumentParser
from agents import CoTAgent, ReflexionStrategy
from arguments import ReflexionParams
from utils import summarize_trial, log_trial

from prompts import cot_agent_prompt1, cot_agent_prompt2, cot_reflect_agent_prompt1, cot_reflect_agent_prompt2, cot_reflect_prompt1, cot_reflect_prompt2
from fewshots import COT, COT_REFLECT

def Caption_Reflexion(rp:ReflexionParams):
    '''
    Reflexion for 3D object captioning
    '''
    agents = [CoTAgent(prompt=rp.prompt,
                       obj_path=os.path.join(rp.objs_dir,obj_name),
                       obj_name=obj_name,

                       agent_prompts=[cot_agent_prompt1, cot_agent_prompt2] if rp.strategy == ReflexionStrategy.NONE else [cot_reflect_agent_prompt1, cot_reflect_agent_prompt2],
                       reflect_prompts=[cot_reflect_prompt1, cot_reflect_prompt2],

                       cot_examples=COT,
                       reflect_examples=COT_REFLECT,

                       evaluator_lm=rp.Evaluator
                       ) for obj_name in os.listdir(rp.objs_dir)]
    
    log = ''

    for trial in range(1,rp.n+1):
        for agent in [a for a in agents if not a.is_finished()]:
            agent.run(reflexion_strategy = rp.strategy)
        log += log_trial(agents, trial)
        correct, incorrect = summarize_trial(agents)
        print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
    
    # save logs and agents
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    log_filename = f"{len(agents)}_objects_{rp.n}_trials_{timestamp}.txt"
    log_path = os.path.join('./Log', rp.strategy.value, log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'w') as f:
        f.write(log)
    print("Logs saved")

    # save all the captions
    captions_dict={}
    for agent in agents:
        captions_dict[agent.obj_name]=agent.captions_list[-1]
    
    captions_filename=f"{len(agents)}_objects_{timestamp}.pkl"
    captions_path=os.path.join(rp.save_dir,rp.strategy.value,captions_filename)

    os.makedirs(os.path.dirname(captions_path), exist_ok=True)

    with open(captions_path, 'wb') as f:  
        pickle.dump(captions_dict, f)
    print("Captions saved")


if __name__ == "__main__" :
    print("Argument parsing...")
    parser = ArgumentParser(description="Reflexion params")
    rp=ReflexionParams(parser)

    args = parser.parse_args()
    rp = rp.extract(args)      

    print("Run Captioning...")
    Caption_Reflexion(rp=rp)