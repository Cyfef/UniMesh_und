import re, string, os
import tiktoken
import pickle

from typing import List, Union, Literal
from enum import Enum
from langchain.prompts import PromptTemplate
from prompts import REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from prompts import cot_reflect_agent_prompt, cot_reflect_prompt
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

                 agent_prompt: PromptTemplate = cot_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = cot_reflect_prompt,
                 cot_examples: str = COT,
                 reflect_examples: str = COT_REFLECT,

                 actor_lm:InterleaveInferencer=BagelActor,
                 evaluator_lm:InterleaveInferencer=BagelEvaluator,
                 self_reflection_lm:InterleaveInferencer=BagelSelfReflection,
                 ) -> None:
        self.prompt = prompt                    #prompt for captioning
        self.imgs_path = self.diffurank_select(obj_path)      #rendered imgs of obj to be captioned
    
        self.agent_prompt = agent_prompt            #
        self.reflect_prompt = reflect_prompt        #
        self.cot_examples = cot_examples            #COT examples
        self.reflect_examples = reflect_examples
        
        self.Actor = actor_lm                     #Actor
        self.Evaluator = evaluator_lm                    #Evaluator
        self.Self_reflection = self_reflection_lm     #Self-reflection
        
        self.reflections = []
        self.reflections_str = ''
        
        self.captions_list=[]
        self.caption=''                             #final caption outcome
    
        self.step_n = 0                        #number of iter
        self.reset()
    
    @staticmethod
    def diffurank_select(obj_path):
        '''
        select 6 imgs based on diffurank scores
        '''
        diffu_scores_path=os.path.join(obj_path,"diffurank_scores.pkl")
        with open(diffu_scores_path, 'rb') as f:
            diffu_scores = pickle.load(f)
        indexed = list(enumerate(diffu_scores))
        indexed.sort(key=lambda x: x[1])
        lowest_six = indexed[:6]
        indices = [idx for idx, _ in lowest_six]
        imgs_path=[os.path.join(obj_path,f"{idx:05}.png") for idx in indices]
        return imgs_path

    def run(self,
            reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy != ReflexionStrategy.NONE:
            self.reflect(reflexion_strategy)
        self.reset()
        self.step()
        self.step_n += 1

    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])  

        self.scratchpad += f'\nObservation: '
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            return
        else:
            print('Invalid action type, please try again.')
    
    def reflect(self,
                strategy: ReflexionStrategy) -> None:
        '''
        self-reflect with strategy
        '''
        print('Running Reflexion strategy...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question , self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.question , self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += '\n'+ format_reflections(self.reflections, header = REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)
    
    def prompt_reflection(self) -> str:
        return format_step(self.Self_reflection(self._build_reflection_prompt()))

    def reset(self) -> None:
        self.scratchpad: str = ''
        self.finished = False       #sign of captioning success

    def prompt_agent(self) -> str:
        return format_step(self.Actor(self._build_agent_prompt()))
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.cot_examples,
                            reflections = self.reflections_str,
                            context = self.context,
                            question = self.question,
                            scratchpad = self.scratchpad)

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            examples = self.reflect_examples,
                            context = self.context,
                            question = self.question,
                            scratchpad = self.scratchpad)
 
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        '''
        return whether the agent gives the right answer
        '''
        return    
    


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
    return step.strip('\n').strip().replace('\n', '')

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
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)