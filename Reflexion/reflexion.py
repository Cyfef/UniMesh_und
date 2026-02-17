import os
import sys
import pickle
import numpy as np

from argparse import ArgumentParser
from agents import CoTAgent, ReflexionStrategy
from utils import summarize_trial, log_trial, save_agents
from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt
from fewshots import COT, COT_REFLECT

CAPTION_PROMPT=""


        
def Caption_Reflexion(
        prompt:str=CAPTION_PROMPT,  
        n:int=3,
        strategy:ReflexionStrategy=ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
        objs_dir:str=""


        
):
    '''
    Reflexion for 3D object captioning
    '''
    agents = [CoTAgent(prompt=prompt,
                       obj_path=os.path.join(objs_dir,obj_name),

                       agent_prompt=cot_agent_prompt if strategy == ReflexionStrategy.NONE else cot_reflect_agent_prompt,
                       reflect_prompt=cot_reflect_prompt,

                       cot_examples=COT,
                       reflect_examples=COT_REFLECT,

                       ) for obj_name in os.listdir(objs_dir)]
    
    trial = 0
    log = ''

    for i in range(n):
        for agent in [a for a in agents if not a.is_correct()]:
            agent.run(reflexion_strategy = strategy)
            print(f'Answer: {agent.key}')
        trial += 1
        log += log_trial(agents, trial)
        correct, incorrect = summarize_trial(agents)
        print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

    with open(os.path.join(root, 'CoT', 'context', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'), 'w') as f:
        f.write(log)
    save_agents(agents, os.path.join(root, 'CoT', 'context', strategy.value, 'agents'))




if __name__ == "__main__" :
    #参数解析
    print("Argument parsing...")

    parser = ArgumentParser(description="Train script params")
    mo=ModelParams(parser)      #高斯模型参数

    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])     #保存checkpoint迭代次数
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])      #保存模型的迭代次数

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)    #最后一次迭代后保存模型

    scene_imgs_path=args.scene_path