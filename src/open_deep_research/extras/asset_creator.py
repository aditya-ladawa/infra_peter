from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate

from langgraph.prebuilt import create_react_agent
from open_deep_research.vector_store_creation import retriever
from dotenv import load_dotenv

load_dotenv()


llm = init_chat_model(model='deepseek-chat', model_provider='deepseek', temperature=0)






from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing_extensions import *
from pydantic import BaseModel, Field


code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert coding assistant specializing in the Manim animation engine for mathematical and scientific visualizations. \n
                Here is the relevant Manim documentation:  \n ------- \n {context} \n ------- \n Answer the user’s question using the documentation above. \n
                Ensure that any code you generate is valid, complete, and can be executed as a standalone Python script. \n
                All required imports, configurations, and object definitions must be included. \n
                Structure your answer as follows:
                1. A brief explanation of the solution and how it works in the context of Manim.
                2. A complete list of imports and configurations.
                3. A clean and executable code block. \n
                Here is the user’s question:
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)



# # Data model
# class code(BaseModel):
#     """Schema for code solutions to questions about LCEL."""

#     prefix: str = Field(description="Description of the problem and approach")
#     imports: str = Field(description="Code block import statements")
#     code: str = Field(description="Code block not including import statements")



# code_gen_chain_oai = code_gen_prompt | llm.with_structured_output(code)



# from langchain_anthropic import ChatAnthropic
# from langchain_core.prompts import ChatPromptTemplate

# ### Anthropic

# # Prompt to enforce tool use
# code_gen_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """<instructions> You are a coding assistant with expertise in Manim, the mathematical animation engine. \n
#               Here is the Manim documentation:  \n ------- \n {context} \n ------- \n Answer the user question based on the \n
#               above provided documentation. Ensure any code you provide can be executed with all required imports, configurations, and \n
#               variables defined. Structure your answer: 1) a prefix describing the code solution, 2) the necessary imports and setup, \n
#               3) the complete and functioning Manim code block. \n
#               Invoke the code tool to structure the output correctly. </instructions> \n Here is the user question:
#             """,
#         ),
#         ("placeholder", "{messages}"),
#     ]
# )


# structured_llm = llm.with_structured_output(code, include_raw=True)


# def check_output(tool_output):
#     """Check for parse error or failure to call the tool"""

#     # Error with parsing
#     if tool_output["parsing_error"]:
#         # Report back output and parsing errors
#         print("Parsing error!")
#         raw_output = str(tool_output["raw"].content)
#         error = tool_output["parsing_error"]
#         raise ValueError(
#             f"Error parsing your output! Be sure to invoke the tool. Output: {raw_output}. \n Parse error: {error}"
#         )

#     # Tool was not invoked
#     elif not tool_output["parsed"]:
#         print("Failed to invoke tool!")
#         raise ValueError(
#             "You did not use the provided tool! Be sure to invoke the tool to structure the output."
#         )
#     return tool_output


# # Chain with output check
# code_chain_raw = (
#     code_gen_prompt | structured_llm | check_output
# )


# def insert_errors(inputs):
#     """Insert errors for tool parsing in the messages"""

#     # Get errors
#     error = inputs["error"]
#     messages = inputs["messages"]
#     messages += [
#         (
#             "assistant",
#             f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool.",
#         )
#     ]
#     return {
#         "messages": messages,
#         "context": inputs["context"],
#     }


# # This will be run as a fallback chain
# fallback_chain = insert_errors | code_chain_raw
# N = 3  # Max re-tries
# code_gen_chain_re_try = code_chain_raw.with_fallbacks(
#     fallbacks=[fallback_chain] * N, exception_key="error"
# )


# def parse_output(solution):
#     """When we add 'include_raw=True' to structured output,
#     it will return a dict w 'raw', 'parsed', 'parsing_error'."""

#     return solution["parsed"]


# # Optional: With re-try to correct for failure to invoke tool
# code_gen_chain = code_gen_chain_re_try | parse_output

# # No re-try
# code_gen_chain = code_gen_prompt | structured_llm | parse_output


# from typing import List
# from typing_extensions import TypedDict


# class GraphState(TypedDict):
#     """
#     Represents the state of our graph.

#     Attributes:
#         error : Binary flag for control flow to indicate whether test error was tripped
#         messages : With user question, error messages, reasoning
#         generation : Code solution
#         iterations : Number of tries
#     """

#     error: str
#     messages: List
#     generation: str
#     iterations: int


# ### Parameter

# # Max tries
# max_iterations = 3
# # Reflect
# # flag = 'reflect'
# flag = "do not reflect"

# ### Nodes


# def generate(state: GraphState):
#     """
#     Generate a code solution

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, generation
#     """

#     print("---GENERATING CODE SOLUTION---")

#     # State
#     messages = state["messages"]
#     iterations = state["iterations"]
#     error = state["error"]

#     # We have been routed back to generation with an error
#     if error == "yes":
#         messages += [
#             (
#                 "user",
#                 "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
#             )
#         ]

#     # Solution
#     code_solution = code_gen_chain.invoke(
#         {"context": concatenated_content, "messages": messages}
#     )
#     messages += [
#         (
#             "assistant",
#             f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
#         )
#     ]

#     # Increment
#     iterations = iterations + 1
#     return {"generation": code_solution, "messages": messages, "iterations": iterations}


# def code_check(state: GraphState):
#     """
#     Check code

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, error
#     """

#     print("---CHECKING CODE---")

#     # State
#     messages = state["messages"]
#     code_solution = state["generation"]
#     iterations = state["iterations"]

#     # Get solution components
#     imports = code_solution.imports
#     code = code_solution.code

#     # Check imports
#     try:
#         exec(imports)
#     except Exception as e:
#         print("---CODE IMPORT CHECK: FAILED---")
#         error_message = [("user", f"Your solution failed the import test: {e}")]
#         messages += error_message
#         return {
#             "generation": code_solution,
#             "messages": messages,
#             "iterations": iterations,
#             "error": "yes",
#         }

#     # Check execution
#     try:
#         exec(imports + "\n" + code)
#     except Exception as e:
#         print("---CODE BLOCK CHECK: FAILED---")
#         error_message = [("user", f"Your solution failed the code execution test: {e}")]
#         messages += error_message
#         return {
#             "generation": code_solution,
#             "messages": messages,
#             "iterations": iterations,
#             "error": "yes",
#         }

#     # No errors
#     print("---NO CODE TEST FAILURES---")
#     return {
#         "generation": code_solution,
#         "messages": messages,
#         "iterations": iterations,
#         "error": "no",
#     }


# def reflect(state: GraphState):
#     """
#     Reflect on errors

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, generation
#     """

#     print("---GENERATING CODE SOLUTION---")

#     # State
#     messages = state["messages"]
#     iterations = state["iterations"]
#     code_solution = state["generation"]

#     # Prompt reflection

#     # Add reflection
#     reflections = code_gen_chain.invoke(
#         {"context": concatenated_content, "messages": messages}
#     )
#     messages += [("assistant", f"Here are reflections on the error: {reflections}")]
#     return {"generation": code_solution, "messages": messages, "iterations": iterations}


# ### Edges


# def decide_to_finish(state: GraphState):
#     """
#     Determines whether to finish.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Next node to call
#     """
#     error = state["error"]
#     iterations = state["iterations"]

#     if error == "no" or iterations == max_iterations:
#         print("---DECISION: FINISH---")
#         return "end"
#     else:
#         print("---DECISION: RE-TRY SOLUTION---")
#         if flag == "reflect":
#             return "reflect"
#         else:
#             return "generate"


# from langgraph.graph import END, StateGraph, START

# workflow = StateGraph(GraphState)

# # Define the nodes
# workflow.add_node("generate", generate)  # generation solution
# workflow.add_node("check_code", code_check)  # check code
# workflow.add_node("reflect", reflect)  # reflect

# # Build graph
# workflow.add_edge(START, "generate")
# workflow.add_edge("generate", "check_code")
# workflow.add_conditional_edges(
#     "check_code",
#     decide_to_finish,
#     {
#         "end": END,
#         "reflect": "reflect",
#         "generate": "generate",
#     },
# )
# workflow.add_edge("reflect", "generate")
# app = workflow.compile()





code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert coding assistant specializing in the Manim animation engine for mathematical and scientific visualizations. \n
                Here is the relevant Manim documentation:  \n ------- \n {context} \n ------- \n Answer the user’s question using the documentation above. \n
                Ensure that any code you generate is valid, complete, and can be executed as a standalone Python script. \n
                All required imports, configurations, and object definitions must be included. \n
                Structure your answer as follows:
                1. A brief explanation of the solution and how it works in the context of Manim.
                2. A complete list of imports and configurations.
                3. A clean and executable code block. \n
                Here is the user’s question:
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)

# === Structured output model for parsing LLM responses ===
class code(BaseModel):
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

# === Compose your LLM chain ===
# This example assumes you have a method `llm.with_structured_output` that returns structured results
# Replace with your actual method of integrating prompt + LLM + parsing
code_gen_chain = code_gen_prompt | llm.with_structured_output(code, include_raw=True)

# === Helper to parse output ===
def parse_output(solution):
    return solution["parsed"]

# === Workflow state type ===
class GraphState(TypedDict):
    error: str
    messages: List
    generation: code
    iterations: int

max_iterations = 3

# === Utility: extract last error from messages ===
def extract_error_from_messages(messages):
    for role, content in reversed(messages):
        if role == "user" and ("failed" in content.lower() or "error" in content.lower()):
            return content
    return ""

# === Utility: get last user question (exclude retry prompts) ===
def extract_latest_user_question(messages):
    for role, content in reversed(messages):
        if role == "user" and "try again" not in content.lower():
            return content
    return ""

# === Main generation function ===
def generate(state: GraphState):
    print(f"\n--- GENERATING CODE SOLUTION (Iteration {state['iterations']+1}) ---")

    messages = state["messages"]
    error = state["error"]

    latest_question = extract_latest_user_question(messages)

    if error == "yes":
        err_msg = extract_error_from_messages(messages)
        improved_query = (
            f"User question: {latest_question}\n"
            f"Error details: {err_msg}\n"
            f"Retrieve Manim docs relevant to fix the error."
        )
        print(f"Retrieving docs with improved query due to error:\n{improved_query}")
        retrieved_docs = retriever.invoke(improved_query)
    else:
        print(f"Retrieving docs based on original question:\n{latest_question}")
        retrieved_docs = retriever.invoke(latest_question)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Invoke LLM code generation with context and messages
    code_solution_raw = code_gen_chain.invoke({"context": context, "messages": messages})

    # Parse structured output
    code_solution = parse_output(code_solution_raw)

    # Append assistant answer to messages
    messages.append((
        "assistant",
        f"{code_solution.prefix}\n\nImports:\n{code_solution.imports}\n\nCode:\n{code_solution.code}"
    ))

    state["messages"] = messages
    state["generation"] = code_solution
    state["iterations"] += 1
    return state

# === Code checking function ===
def code_check(state: GraphState):
    print("--- CHECKING GENERATED CODE ---")

    messages = state["messages"]
    code_solution = state["generation"]

    try:
        exec(code_solution.imports, globals())
    except Exception as e:
        print(f"Import check failed: {e}")
        messages.append(("user", f"Your solution failed the import test: {e}"))
        state.update({"messages": messages, "error": "yes"})
        return state

    try:
        exec(code_solution.imports + "\n" + code_solution.code, globals())
    except Exception as e:
        print(f"Code execution failed: {e}")
        messages.append(("user", f"Your solution failed the code execution test: {e}"))
        state.update({"messages": messages, "error": "yes"})
        return state

    print("Code passed import and execution checks.")
    state.update({"error": "no", "messages": messages})
    return state

# === Decide whether to finish or retry ===
def decide_to_finish(state: GraphState):
    if state["error"] == "no" or state["iterations"] >= max_iterations:
        print("--- DECISION: FINISH ---")
        return "end"
    else:
        print("--- DECISION: RETRY ---")
        return "generate"

# === Main workflow loop ===
def workflow_loop(initial_user_question: str):
    state: GraphState = {
        "error": "no",
        "messages": [("user", initial_user_question)],
        "generation": None,
        "iterations": 0,
    }

    while True:
        state = generate(state)
        state = code_check(state)
        next_action = decide_to_finish(state)
        if next_action == "end":
            break

    return state["generation"]

# === Example usage ===
if __name__ == "__main__":
    # Make sure to replace retriever and llm with your actual initialized objects before running!
    if retriever is None or llm is None:
        print("Please initialize your retriever and llm before running this script.")
    else:
        user_question = "Create a Manim animation showing infinite bezier curves for 10 seconds."
        solution = workflow_loop(user_question)

        print("\n--- FINAL GENERATED SOLUTION ---")
        print("Description:\n", solution.prefix)
        print("Imports:\n", solution.imports)
        print("Code:\n", solution.code)