from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase 
class ContentCrew():
    """ content writing crew for guide creator flow """
    
    @agent 
    def content_writer(self) -> Agent:
        """ agent responsible for writing content for the guide """
        return Agent(
            config=self.agents_config['content_writer'],
            verbose=True
        )
    
    @agent
    def content_reviewer(self) -> Agent:
        """ agent responsible for reviewing content created by the content writer agent """
        return Agent(
            config=self.agents_config['content_reviewer'],
            verbose=True
        )
    
    @task 
    def write_section_task(self) -> Task:
        """ task for writing a section of the guide """
        return Task(
            config=self.tasks_config['write_section_task']
        )
    
    @task
    def review_section_task(self) -> Task:
        """ task for reviewing a section of the guide """
        return Task(
            config=self.tasks_config['review_section_task']
        )
    
    
    @crew 
    def crew(self) -> Crew:
        """ creates the content writing crew with the defined agents and tasks """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
    
    