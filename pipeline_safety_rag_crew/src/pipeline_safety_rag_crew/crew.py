"""
Pipeline Safety RAG Crew
========================
A two-agent CrewAI crew that answers questions about PHMSA pipeline safety
regulations using a FAISS + Amazon Bedrock Titan V2 retrieval pipeline.

Agents
------
retrieval_specialist  — Queries the regulation index via RAGSearchTool.
regulatory_analyst    — Synthesises retrieved passages into a cited answer.

Usage
-----
    from pipeline_safety_rag_crew.crew import PipelineSafetyRAGCrew

    result = PipelineSafetyRAGCrew().crew().kickoff(
        inputs={"question": "What are the pressure testing requirements under §192.505?"}
    )
    print(result.raw)
"""

from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from pipeline_safety_rag_crew.tools.rag_tool import RAGSearchTool


@CrewBase
class PipelineSafetyRAGCrew:
    """Two-agent crew: retrieve regulatory passages → synthesise a cited answer."""

    agents: List[BaseAgent]
    tasks: List[Task]

    # ── Agents ────────────────────────────────────────────────────────────────

    @agent
    def retrieval_specialist(self) -> Agent:
        """Queries the FAISS regulation index using RAGSearchTool."""
        return Agent(
            config=self.agents_config["retrieval_specialist"],  # type: ignore[index]
            tools=[RAGSearchTool()],
            verbose=True,
        )

    @agent
    def regulatory_analyst(self) -> Agent:
        """Synthesises retrieved passages into a clear, cited answer."""
        return Agent(
            config=self.agents_config["regulatory_analyst"],  # type: ignore[index]
            verbose=True,
        )

    # ── Tasks ─────────────────────────────────────────────────────────────────

    @task
    def retrieval_task(self) -> Task:
        """Retrieve the most relevant regulatory passages for the question."""
        return Task(
            config=self.tasks_config["retrieval_task"],  # type: ignore[index]
        )

    @task
    def synthesis_task(self) -> Task:
        """Synthesise a cited answer from the retrieved passages."""
        return Task(
            config=self.tasks_config["synthesis_task"],  # type: ignore[index]
            output_file="output/answer.md",
        )

    # ── Crew ──────────────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """Assemble the crew with sequential task execution."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
