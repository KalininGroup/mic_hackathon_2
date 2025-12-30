from google.adk.agents import Agent

from nanorange.settings import TEXT_MODEL
from nanorange.prompts import NANORANGE_ROOT_INSTR
from nanorange.agent_tools import analyze_cryo_tem, optimize_parameters


cryo_tem_agent = Agent(
    model=TEXT_MODEL,
    name="cryo_tem_agent",
    description="AI assistant for analyzing Cryo-TEM images of nanoparticles.",
    instruction=NANORANGE_ROOT_INSTR,
    tools=[analyze_cryo_tem, optimize_parameters],
)

root_agent = cryo_tem_agent