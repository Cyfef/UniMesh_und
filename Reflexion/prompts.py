from langchain.prompts import PromptTemplate

COT_INSTRUCTION1 = """Solve a 3D object captioning task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given 6 rendered images of a 3D object, which you should use to caption the 3D object.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
Images: """

COT_INSTRUCTION2 ="""Prompt: {prompt}{scratchpad}"""

COT_AGENT_REFLECT_INSTRUCTION1 = """Solve a 3D object captioning task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given 6 rendered images of a 3D object, which you should use to caption the 3D object.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Images: """

COT_AGENT_REFLECT_INSTRUCTION2 ="""Prompt: {prompt}{scratchpad}"""

COT_REFLECT_INSTRUCTION1 = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to relevant context and a question to answer. You were unsuccessful in answering the question because you gave the wrong answer with Finish[<answer>] . In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Images:""" 

COT_REFLECT_INSTRUCTION2 ="""
Prompt: {prompt}{scratchpad}

Reflection:"""

cot_agent_prompt1 = PromptTemplate(
                        input_variables=["examples", "reflections"],
                        template = COT_INSTRUCTION1,
                        )

cot_agent_prompt2 = PromptTemplate(
                        input_variables=["prompt", "scratchpad"],
                        template = COT_INSTRUCTION2,
                        )

cot_reflect_agent_prompt1 = PromptTemplate(
                        input_variables=["examples", "reflections"],
                        template = COT_AGENT_REFLECT_INSTRUCTION1,
                        )

cot_reflect_agent_prompt2 = PromptTemplate(
                        input_variables=["prompt", "scratchpad"],
                        template = COT_AGENT_REFLECT_INSTRUCTION2,
                        )

cot_reflect_prompt1 = PromptTemplate(
                        input_variables=["examples"],
                        template = COT_REFLECT_INSTRUCTION1,
                        )

cot_reflect_prompt2 = PromptTemplate(
                        input_variables=["prompt", "scratchpad"],
                        template = COT_REFLECT_INSTRUCTION2,
                        )

REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
REFLECTION_AFTER_LAST_TRIAL_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n'



