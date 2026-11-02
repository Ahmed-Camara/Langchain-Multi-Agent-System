from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search,scrape_url
from dotenv import load_dotenv
import os
load_dotenv()
import anthropic
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


    
llm = ChatOpenAI(model="qwen3.6-35b-fp8",temperature = 0,base_url = "http://10.30.60.20:8000/v1")
llms = ChatAnthropic(
    model="claude-opus-5-5",
    temperature=0,
    api_key=ANTHROPIC_API_KEY,base_url="https://api.anthropic.com"
)
def create_search_agent():
    return create_agent(model=llm,
                        tools=[web_search])
    
    
def create_reader_agent():
    return create_agent(model=llm,
                            tools=[scrape_url])
    

def create_writer_agent():
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system","You are an expert research writer. Write clear, structured and insightful reports."),
        ("human","""Write a detailed research report on the topic below.
            Topic: {topic}
            
            Research Gathered:
            {research}
            
            Structure the report as:
            - Introduction
            - Key Findings (minimum 3 well-explained points)
            - Conclusion
            - Sources (list all URLs found in the research)
            
            Be detailed, factual and professional.
        """),
        ])

    writer_chain = writer_prompt | llm | StrOutputParser()
    return writer_chain


def create_critic_agent():
    critic_prompt = ChatPromptTemplate.from_messages([
        ("system","You are a sharp and constructive research critic. Be honest and specific."),
        ("human","""Review the research report below and evaluate it strictly.
         
         Report:
         {report}
         
         Respond in this exact form:
         
         Score: X/10
         
         Strengths:
         - ...
         - ...
         
         Areas to Improve:
         - ...
         - ...
         
         One line verdict:
         ...
         """)
    ])
    
    critic_chain = critic_prompt | llm | StrOutputParser()
    return critic_chain