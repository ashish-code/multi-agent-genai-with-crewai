import json
import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from crewai import LLM 
from crewai.flow.flow import Flow, listen, start 
from guide_creator_flow.crews.content_crew.content_crew import ContentCrew


# define our models for structured data
class Section(BaseModel):
    title: str = Field(..., description="Title of the section")
    description: str = Field(..., description="Brief description of what the section covers")
    

class GuideOutline(BaseModel):
    title: str = Field(description="Title of the guide")
    introduction: str = Field(description="Introduction to the topic of the guide")
    target_audience: str = Field(description="Description of the target audience for the guide")
    sections: List[Section] = Field(description="List of sections in the guide")
    conclusion: str = Field(description="Conclusion of the guide")


class GuideCreatorState(BaseModel):
    """ state model for the guide creator flow """
    topic: str = Field(default="", description="The topic for which the guide is being created")
    audience_level: str = Field(default="", description="The expertise level of the target audience (e.g. beginner, intermediate, advanced)")
    guide_outline: Optional[GuideOutline] = Field(default=None, description="The outline of the guide being created")
    sections_content: Dict[str, str] = Field(default_factory=dict, description="Dictionary to hold the content for each section of the guide")
    

class GuideCreatorFlow(Flow[GuideCreatorState]):
    """ flow for creating a guide on a given topic for a specific target audience """
    
    @start()
    def get_user_input(self):
        """ initial step to get the topic and target audience from the user """
        
        # title the console output for better user experience        
        print("Welcome to the Guide Creator! This flow will help you create a comprehensive guide on a topic of your choice for a specific target audience.")
        print("Let's start by gathering some information about the guide you want to create.")
        
        # get the topic and target audience from the user
        self.state.topic = input("Enter the topic for the guide: ")
        
        # get audiencie level from the user with validation to ensure it's one of the expected values
        valid_audience_levels = ["beginner", "intermediate", "advanced"]
        while True:
            audience_level = input("Enter the expertise level of the target audience (e.g. beginner, intermediate, advanced): ")
            if audience_level.lower() in valid_audience_levels:
                self.state.audience_level = audience_level.lower()
                break
            else:
                print(f"Invalid input. Please enter one of the following: {', '.join(valid_audience_levels)}")  
        
        # echo the topic and audience level back to the user for confirmation
        print(f"Great! You've chosen to create a guide on '{self.state.topic}' for a '{self.state.audience_level}' audience.")
        print("Next, we'll generate an outline for the guide based on this information.")
        
        return self.state

    @listen(get_user_input)
    def create_guide_outline(self, state):
        """ step to create the outline of the guide using an LLM agent """
        
        llm = LLM(model='google/gemini-2.5-flash-lite', verbose=True, response_format=GuideOutline)
        
        # create messages for the outline generation agent
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": f"""
            Create a detailed outline for a comprehensive guide on "{state.topic}" for {state.audience_level} level learners.

            The outline should include:
            1. A compelling title for the guide
            2. An introduction to the topic
            3. 4-6 main sections that cover the most important aspects of the topic
            4. A conclusion or summary

            For each section, provide a clear title and a brief description of what it should cover.
            """}
        ]
        
        # make the llm class with JSON response format
        response = llm.call(messages=messages)
        
        # response_format=GuideOutline causes crewai/LiteLLM to return a parsed model directly
        self.state.guide_outline = response
        
        # ensure output directory exists before saving the outline
        os.makedirs("output", exist_ok=True)
        
        # save the outline to a json file for reference
        with open("output/guide_outline.json", "w") as f:
            json.dump(self.state.guide_outline.model_dump(), f, indent=2)
        
        print("Guide outline created successfully! Here's a summary of the outline:")
        print(f"Title: {self.state.guide_outline.title}")
        print(f"Introduction: {self.state.guide_outline.introduction}")
        print(f"Target Audience: {self.state.guide_outline.target_audience}")
        print("Sections:")
        for idx, section in enumerate(self.state.guide_outline.sections, 1):
            print(f"  {idx}. {section.title} - {section.description}")
        print(f"Conclusion: {self.state.guide_outline.conclusion}")
        
        return self.state.guide_outline
    
    @listen(create_guide_outline)
    def write_and_compile_guide(self, outline):
        """ step to write the content for each section of the guide using the ContentCrew and compile the final guide """
        
        print("Now we'll move on to writing the content for each section of the guide using our ContentCrew.")
        completed_sections = []
        
        # process sections sequentially to ensure the content reviewer can review each section after it's written
        for section in outline.sections:
            print(f"Working on section: {section.title}")
            
            # build context from previous sections to provide to the content writer agent
            previous_sections_text = ""
            if completed_sections:
                previous_sections_text = "# Previously Written Sections\n\n"
                for title in completed_sections:
                    previous_sections_text += f"## {title}\n\n"
                    previous_sections_text += self.state.sections_content.get(title, "") + "\n\n"
            else:
                previous_sections_text = "No previous sections written yet.\n\n"
            
            
            # run the content crew for the current section, providing the section title, description, and context from previous sections
            result = ContentCrew().crew().kickoff(inputs={
                "section_title": section.title,
                "section_description": section.description,
                "previous_sections": previous_sections_text,
                "audience_level": self.state.audience_level,
                })
            
            # store the content
            self.state.sections_content[section.title] = result.raw 
            completed_sections.append(section.title)
            print(f"Completed section: {section.title}")
            
        # compile the final guide 
        guide_content = f"# {outline.title}\n\n"
        guide_content += f"## Introduction\n\n{outline.introduction}\n\n"
        
        # Add each section in order
        for section in outline.sections:
            guide_content += f"## {section.title}\n\n"
            guide_content += self.state.sections_content.get(section.title, "") + "\n\n"
        
        # add conclusion
        guide_content += f"## Conclusion\n\n{outline.conclusion}\n\n"
        
        # save the final guide to a markdown file
        with open("output/final_guide.md", "w") as f:
            f.write(guide_content)
        
        print("Guide writing and compilation completed! The final guide has been saved to 'output/final_guide.md'.")
        return guide_content


def kickoff():
    """ function to kickoff the guide creator flow """
    flow = GuideCreatorFlow()
    flow.kickoff()
    print("Guide creation process completed successfully!")
    print("You can find the generated guide and outline in the 'output' directory.")
    

def plot():
    """ function to plot the flow graph for visualization purposes """
    os.makedirs("output", exist_ok=True)
    flow = GuideCreatorFlow()
    flow.plot("output/guide_creator_flow_graph")
    print("Flow graph has been saved to 'output/guide_creator_flow_graph'.")


if __name__ == "__main__":
    kickoff()
