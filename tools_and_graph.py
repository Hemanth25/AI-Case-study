# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 23:06:51 2025

@author: hemanthn
"""


from langchain.agents import Tool
from logger import setup_logger
# from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from typing import Annotated
import re

logger = setup_logger()

def create_graph(model, scaler, collection, logger):
    llm = init_chat_model("openai:gpt-4.1")

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        
    def query_chroma(collection, query):
        return collection.query(query_texts=[query], n_results=3)

    def check_eligibility(input_str):
        try:
            numbers = re.findall(r'\d+\.?\d*', input_str)
            values = list(map(float, numbers))
            features = scaler.transform([values])
            result = "Eligible" if model.predict(features)[0] == 1 else "Not Eligible"
            logger.info(f"Eligibility result: {result}")
            if result == "Eligible":
                return str("profile is Eligible") + llm.invoke(f"Applicant profile is {result}. Suggest upskilling, job matching, and career counseling for: {input_str}").content
            else:
                return str("profile is Not Eligible") + llm.invoke(f"Applicant profile is {result} for : {input_str}. soft decline financial support and explain why.").content
        except Exception as e:
            logger.error(f"Eligibility check error: {e}")
            return "Invalid input."
    
    def economic_enablement_recommendation_tool(query):
        """Suggest upskilling and career support based on applicant profile."""
        logger.info("Generated economic enablement recommendation.")
        return llm.invoke(f"Suggest upskilling, job matching, and career counseling for: {query}").content
    
    def social_support_recommendation_tool(query):
        """Generate a recommendation for social support based on applicant status."""
        logger.info("Generated social support recommendation.")
        return llm.invoke(f"Applicant status: {query}. Approve or soft decline financial support and explain why.").content

    def format_data_tool(raw):
        prompt = f"""Text: {raw}
        
        You're an assistant helping gather information for eligibility assessment.
        
        Ask the user the following questions **one by one**, only after they have answered the previous one:
        
        1. What is your monthly income (in local currency)?
        2. How many years have you been employed?
        3. How many members are there in your family?
        4. What is your estimated wealth index rank (ask user to enter value between 0 and 10, where 10 is highest wealth)?
        
        Wait for each answer before asking the next question.
        
        After collecting all 4 values, respond with this format:
        "income, employment_years, family_size, wealth_index"
        
        Do not include any explanation in the final response — just output the 4 numbers as comma-separated values.
        """
        print("check_eligibility_tool input is {0}".format(llm.invoke(prompt).content))
        return llm.invoke(prompt).content

    tools = [
        Tool(name="format_data_for_ml", func=format_data_tool, description="Convert raw text to ML input."),
        Tool(name="eligibility_checker", func=check_eligibility, description="Check eligibility."),
        Tool(name="social_support_recommendation", func=social_support_recommendation_tool, description="Approve or soft decline financial support."),
        Tool(name="economic_enablement_recommendation", func=economic_enablement_recommendation_tool, description="Suggest training, job matching, and career counseling."),
        Tool(name="query_documents", func=lambda x: query_chroma(collection, x), description="Query ChromaDB.")
    ]

    llm_tools = llm.bind_tools(tools)

    def tool_llm_node(state: State):
        return {"messages": [llm_tools.invoke(state["messages"]) ]}

    graph = StateGraph(State)
    graph.add_node("tool_llm", tool_llm_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "tool_llm")
    graph.add_conditional_edges("tool_llm", tools_condition)
    graph.add_edge("tools", "tool_llm")
    return graph.compile()