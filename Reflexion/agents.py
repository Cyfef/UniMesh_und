import re,os
import tiktoken
import pickle

from typing import List
from enum import Enum
from PIL import Image
from langchain.prompts import PromptTemplate

from prompts import REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from prompts import cot_reflect_agent_prompt1, cot_reflect_agent_prompt2, cot_reflect_prompt1, cot_reflect_prompt2
from fewshots import COT, COT_REFLECT

from LMs.Bagel import BagelActor,BagelEvaluator,BagelSelfReflection,InterleaveInferencer

class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context 
    REFLEXION: Apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial' 
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'

class CoTAgent:
    '''
    CoTAgent for 3D object captioning
    '''
    def __init__(self,
                 prompt:str,
                 obj_path:str,
                 obj_name:str,

                 agent_prompts: List[PromptTemplate] = [cot_reflect_agent_prompt1, cot_reflect_agent_prompt2],
                 reflect_prompts: List[PromptTemplate] = [cot_reflect_prompt1, cot_reflect_prompt2],
                 
                 cot_examples: str = COT,
                 reflect_examples: str = COT_REFLECT,

                 actor_lm:InterleaveInferencer=BagelActor,
                 evaluator_lm:InterleaveInferencer=BagelEvaluator,
                 self_reflection_lm:InterleaveInferencer=BagelSelfReflection,
                 ) -> None:
        self.prompt = prompt                    #prompt for captioning
        self.imgs_path = self.diffurank_select(obj_path)      #rendered imgs of obj to be captioned
        self.imgs_list=[Image.open(img_path) for img_path in self.imgs_path]
        self.obj_name=obj_name            #name of the object to be captioned
    
        self.agent_prompts = agent_prompts            #
        self.reflect_prompts = reflect_prompts        #

        self.cot_examples = cot_examples            #COT examples
        self.reflect_examples = reflect_examples
        
        self.Actor = actor_lm                     #Actor
        self.Evaluator = evaluator_lm                    #Evaluator
        self.Self_reflection = self_reflection_lm     #Self-reflection
        
        self.reflections = []
        self.reflections_str = ''
        
        self.captions_list=[]
        self.answer=''
        self.correct="INCORRECT"                            
    
        self.step_n = 0                        #number of iter
        self.reset()
    
    @staticmethod    
    def diffurank_select(obj_path):
        '''
        select 6 imgs based on diffurank scores
        '''
        diffu_path = os.path.join(obj_path, "diffurank_scores.pkl")
        with open(diffu_path, 'rb') as f:
            diffu_scores = pickle.load(f)     # numpy array

        indexed = list(enumerate(diffu_scores))
        indexed.sort(key=lambda x: x[1])
        
        lowest_six = indexed[:6]
        indices = [idx for idx, val in lowest_six][::-1] 

        imgs_path=[os.path.join(obj_path,f"{idx:05}.png") for idx in indices] 
        
        return imgs_path

    def run(self,reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy != ReflexionStrategy.NONE:
            self.reflect(reflexion_strategy)
        self.reset()
        self.step()
        self.step_n += 1

    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought:'
        self.scratchpad += ' ' + self._Actor()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction:'
        action = self._Actor()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])  

        self.scratchpad += f'\nObservation: '
        if action_type == 'Finish':
            self.answer = argument
            print(f"Object {self.obj_name}: {self.answer}")
            self.captions_list.append(self.answer)
            self._Evaluator()
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            return
        else:
            print('Invalid action type, please try again.')
    
    def reflect(self,strategy: ReflexionStrategy) -> None:
        '''
        self-reflect with strategy
        '''
        print('Running Reflexion strategy...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.prompt , self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self._Reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.prompt , self.scratchpad)
            self.reflections = [self._Reflection()]
            self.reflections_str += '\n'+ format_reflections(self.reflections, header = REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)
    
    def reset(self) -> None:
        self.scratchpad: str = ''
        self.finished = False       #sign of captioning success

    def _Actor(self) -> str:
        return format_step(self.Actor.interleave_inference(input_lists=self._build_agent_prompt(),
                                                            understanding_output=True,
                                                            max_think_token_n=1000,
                                                            do_sample=False,
                                                            # text_temperature=0.3,
                                                            )[0])
    def _Evaluator(self):
        EVAL_PROMPT="""
        You are given 6 renderings of a 3D object and a caption that describes it. Please determine whether the caption can accurately describe the object. Your output can only be "CORRECT" or "INCORRECT". 
        """

        input_list=self.imgs_list+["The caption:"+self.answer]+[EVAL_PROMPT]

        self.correct=self.Evaluator.interleave_inference(input_lists=input_list,
                                                        understanding_output=True,
                                                        max_think_token_n=1000,
                                                        do_sample=False,
                                                        # text_temperature=0.3,
                                                        )[0]
        self.correct=self.correct.strip().upper()
        print(f"Object {self.obj_name}:{self.correct}")

    def _Reflection(self) -> str:
        return format_step(self.Self_reflection.interleave_inference(input_lists=self._build_reflection_prompt(),
                                                                        understanding_output=True,
                                                                        max_think_token_n=1000,
                                                                        do_sample=False,
                                                                        # text_temperature=0.3,
                                                                        )[0])
    
    def _build_agent_prompt(self) -> List:
        return [self.agent_prompts[0].format(
                    examples = self.cot_examples,
                    reflections = self.reflections_str)]+\
                self.imgs_list+\
                [self.agent_prompts[1].format(
                    prompt = self.prompt,
                    scratchpad = self.scratchpad)]
    
    def _build_reflection_prompt(self) -> List:
        return [self.reflect_prompts[0].format(examples = self.cot_examples)]+\
                self.imgs_list+\
                [self.reflect_prompts[1].format(
                    prompt = self.prompt,
                    scratchpad = self.scratchpad)]
 
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        '''
        return whether the agent gives the right answer
        '''
        if self.correct=="CORRECT":
            return True
        return False   
    
### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    
    else:
        return None

def format_step(step: str) -> str:
    return step.strip('\n').strip()

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_last_attempt(question: str,
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer = gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = [(i, line) for i, line in enumerate(lines) if line.startswith('Observation')]
    observations.sort(key=lambda x: len(tokenizer.encode(x[1])), reverse=True)
    for i, line in observations:
        if len(tokenizer.encode('\n'.join(lines))) <= n_tokens:
            break
        lines[i] = line.split(':')[0] + ': [Content truncated for context window]'
        
    return '\n'.join(lines)