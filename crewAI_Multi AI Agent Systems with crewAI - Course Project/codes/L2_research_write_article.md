You can download the `requirements.txt` for this course from the workspace of this lab. `File --> Open...`

# L2: Create Agents to Research and Write an Article

In this lesson, you will be introduced to the foundational concepts of multi-agent systems and get an overview of the crewAI framework.

The libraries are already installed in the classroom. If you're running this notebook on your own machine, you can install the following:
```Python
!pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29
```


```python
# Warning control
import warnings
warnings.filterwarnings('ignore')
```

- Import from the crewAI libray.


```python
from crewai import Agent, Task, Crew
```

- As a LLM for your agents, you'll be using OpenAI's `gpt-3.5-turbo`.

**Optional Note:** crewAI also allow other popular models to be used as a LLM for your Agents. You can see some of the examples at the [bottom of the notebook](#1).


```python
import os
from utils import get_openai_api_key

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
```

## Creating Agents

- Define your Agents, and provide them a `role`, `goal` and `backstory`.
- It has been seen that LLMs perform better when they are role playing.

### Agent: Planner

**Note**: The benefit of using _multiple strings_ :
```Python
varname = "line 1 of text"
          "line 2 of text"
```

versus the _triple quote docstring_:
```Python
varname = """line 1 of text
             line 2 of text
          """
```
is that it can avoid adding those whitespaces and newline characters, making it better formatted to be passed to the LLM.


```python
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation=False,
	verbose=True
)
```

### Agent: Writer


```python
writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True
)
```

### Agent: Editor


```python
editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    allow_delegation=False,
    verbose=True
)
```

## Creating Tasks

- Define your Tasks, and provide them a `description`, `expected_output` and `agent`.

### Task: Plan


```python
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.",
    agent=planner,
)
```

### Task: Write


```python
write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer,
)
```

### Task: Edit


```python
edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=editor
)
```

## Creating the Crew

- Create your crew of Agents
- Pass the tasks to be performed by those agents.
    - **Note**: *For this simple example*, the tasks will be performed sequentially (i.e they are dependent on each other), so the _order_ of the task in the list _matters_.
- `verbose=2` allows you to see all the logs of the execution. 


```python
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2
)
```

## Running the Crew

**Note**: LLMs can provide different outputs for they same input, so what you get might be different than what you see in the video.


```python
result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})
```

    [1m[95m [DEBUG]: == Working Agent: Content Planner[00m
    [1m[95m [INFO]: == Starting Task: 1. Prioritize the latest trends, key players, and noteworthy news on Artificial Intelligence.
    2. Identify the target audience, considering their interests and pain points.
    3. Develop a detailed content outline including an introduction, key points, and a call to action.
    4. Include SEO keywords and relevant data or sources.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer
    
    Final Answer:
    
    Title: Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    Introduction:
    - Brief overview of the rapid advancements in artificial intelligence (AI)
    - Importance of staying updated on the latest trends in AI
    
    Key Points:
    1. Latest Trends in Artificial Intelligence:
    - Breakthroughs in deep learning and neural networks
    - Expansion of AI applications in various industries such as healthcare, finance, and retail
    - Integration of AI with Internet of Things (IoT) technology
    - Rise of explainable AI for increased transparency and accountability
    
    2. Key Players in Artificial Intelligence:
    - Google's DeepMind leading in AI research and development
    - Microsoft's Azure AI platform for business solutions
    - IBM Watson offering cognitive computing services
    - Startups like OpenAI and Element AI making significant contributions
    
    3. Noteworthy News in Artificial Intelligence:
    - Ethical concerns surrounding AI development and deployment
    - Regulatory efforts to govern AI technologies
    - Collaborations between tech giants and research institutions for AI innovation
    
    Audience Analysis:
    Target Audience: Tech enthusiasts, industry professionals, entrepreneurs, and students interested in AI
    Interests: Latest technological advancements, AI applications across industries, ethical implications of AI
    Pain Points: Keeping up with rapidly evolving AI landscape, understanding complex AI concepts, accessing reliable sources for AI news
    
    Content Outline:
    - Introduction to the importance of staying updated on AI trends
    - Latest Trends in Artificial Intelligence
    - Key Players in Artificial Intelligence
    - Noteworthy News in Artificial Intelligence
    - Audience-specific insights and analysis
    - Call to Action encouraging further exploration of AI resources and communities
    
    SEO Keywords: artificial intelligence trends, AI industry players, latest AI news, explainable AI, AI applications, AI ethics
    
    Resources:
    1. "The State of AI in 2021" report by Forbes
    2. "Deep Learning" course on Coursera by Andrew Ng
    3. "AI Ethics: Global Perspectives" book by Markus Dubber and Frank Pasquale
    
    By following this comprehensive content plan, the blog article on Artificial Intelligence will engage the target audience with informative insights and relevant resources, helping them make informed decisions in the ever-evolving world of AI.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Content Planner] Task output: Title: Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    Introduction:
    - Brief overview of the rapid advancements in artificial intelligence (AI)
    - Importance of staying updated on the latest trends in AI
    
    Key Points:
    1. Latest Trends in Artificial Intelligence:
    - Breakthroughs in deep learning and neural networks
    - Expansion of AI applications in various industries such as healthcare, finance, and retail
    - Integration of AI with Internet of Things (IoT) technology
    - Rise of explainable AI for increased transparency and accountability
    
    2. Key Players in Artificial Intelligence:
    - Google's DeepMind leading in AI research and development
    - Microsoft's Azure AI platform for business solutions
    - IBM Watson offering cognitive computing services
    - Startups like OpenAI and Element AI making significant contributions
    
    3. Noteworthy News in Artificial Intelligence:
    - Ethical concerns surrounding AI development and deployment
    - Regulatory efforts to govern AI technologies
    - Collaborations between tech giants and research institutions for AI innovation
    
    Audience Analysis:
    Target Audience: Tech enthusiasts, industry professionals, entrepreneurs, and students interested in AI
    Interests: Latest technological advancements, AI applications across industries, ethical implications of AI
    Pain Points: Keeping up with rapidly evolving AI landscape, understanding complex AI concepts, accessing reliable sources for AI news
    
    Content Outline:
    - Introduction to the importance of staying updated on AI trends
    - Latest Trends in Artificial Intelligence
    - Key Players in Artificial Intelligence
    - Noteworthy News in Artificial Intelligence
    - Audience-specific insights and analysis
    - Call to Action encouraging further exploration of AI resources and communities
    
    SEO Keywords: artificial intelligence trends, AI industry players, latest AI news, explainable AI, AI applications, AI ethics
    
    Resources:
    1. "The State of AI in 2021" report by Forbes
    2. "Deep Learning" course on Coursera by Andrew Ng
    3. "AI Ethics: Global Perspectives" book by Markus Dubber and Frank Pasquale
    
    By following this comprehensive content plan, the blog article on Artificial Intelligence will engage the target audience with informative insights and relevant resources, helping them make informed decisions in the ever-evolving world of AI.
    
    [00m
    [1m[95m [DEBUG]: == Working Agent: Content Writer[00m
    [1m[95m [INFO]: == Starting Task: 1. Use the content plan to craft a compelling blog post on Artificial Intelligence.
    2. Incorporate SEO keywords naturally.
    3. Sections/Subtitles are properly named in an engaging manner.
    4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.
    5. Proofread for grammatical errors and alignment with the brand's voice.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer
    
    Final Answer:
    
    # Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    ## Introduction
    Artificial intelligence (AI) has been rapidly evolving in recent years, with groundbreaking advancements in deep learning and neural networks. It's crucial for tech enthusiasts, industry professionals, entrepreneurs, and students interested in AI to stay updated on the latest trends in this dynamic field. By understanding the latest developments in AI, individuals can harness the power of this technology to drive innovation and create impactful solutions.
    
    ## Latest Trends in Artificial Intelligence
    One of the most significant trends in AI is the expansion of its applications across various industries such as healthcare, finance, and retail. AI is revolutionizing these sectors by enabling predictive analytics, personalized recommendations, and automation of repetitive tasks. Additionally, the integration of AI with Internet of Things (IoT) technology is opening up new possibilities for interconnected smart devices and systems. Another notable trend is the rise of explainable AI, which focuses on enhancing transparency and accountability in AI decision-making processes.
    
    ## Key Players in Artificial Intelligence
    Several key players are leading the way in AI research and development. Google's DeepMind has been at the forefront of AI innovation, pushing boundaries in machine learning and artificial general intelligence. Microsoft's Azure AI platform offers robust business solutions, empowering organizations to leverage AI for improved efficiency and productivity. IBM Watson is known for its cognitive computing services, providing advanced AI capabilities for data analysis and natural language processing. Moreover, startups like OpenAI and Element AI are making significant contributions to the AI landscape, driving innovation and pushing the boundaries of what is possible with artificial intelligence.
    
    ## Noteworthy News in Artificial Intelligence
    Alongside the rapid advancements in AI, there are ethical concerns surrounding its development and deployment. As AI technologies become more pervasive, there is a growing need for regulatory efforts to govern their use and mitigate potential risks. Collaborations between tech giants and research institutions are also driving AI innovation, fostering a collaborative ecosystem for the development of cutting-edge AI solutions.
    
    In conclusion, staying informed about the latest trends and key players in artificial intelligence is essential for anyone looking to navigate the rapidly evolving landscape of AI. By exploring the latest advancements, understanding the key players driving innovation, and staying abreast of noteworthy news in AI, individuals can harness the power of this transformative technology to drive positive change in their respective fields. Let's continue to explore the exciting world of artificial intelligence and unlock its full potential for the benefit of society.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Content Writer] Task output: # Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    ## Introduction
    Artificial intelligence (AI) has been rapidly evolving in recent years, with groundbreaking advancements in deep learning and neural networks. It's crucial for tech enthusiasts, industry professionals, entrepreneurs, and students interested in AI to stay updated on the latest trends in this dynamic field. By understanding the latest developments in AI, individuals can harness the power of this technology to drive innovation and create impactful solutions.
    
    ## Latest Trends in Artificial Intelligence
    One of the most significant trends in AI is the expansion of its applications across various industries such as healthcare, finance, and retail. AI is revolutionizing these sectors by enabling predictive analytics, personalized recommendations, and automation of repetitive tasks. Additionally, the integration of AI with Internet of Things (IoT) technology is opening up new possibilities for interconnected smart devices and systems. Another notable trend is the rise of explainable AI, which focuses on enhancing transparency and accountability in AI decision-making processes.
    
    ## Key Players in Artificial Intelligence
    Several key players are leading the way in AI research and development. Google's DeepMind has been at the forefront of AI innovation, pushing boundaries in machine learning and artificial general intelligence. Microsoft's Azure AI platform offers robust business solutions, empowering organizations to leverage AI for improved efficiency and productivity. IBM Watson is known for its cognitive computing services, providing advanced AI capabilities for data analysis and natural language processing. Moreover, startups like OpenAI and Element AI are making significant contributions to the AI landscape, driving innovation and pushing the boundaries of what is possible with artificial intelligence.
    
    ## Noteworthy News in Artificial Intelligence
    Alongside the rapid advancements in AI, there are ethical concerns surrounding its development and deployment. As AI technologies become more pervasive, there is a growing need for regulatory efforts to govern their use and mitigate potential risks. Collaborations between tech giants and research institutions are also driving AI innovation, fostering a collaborative ecosystem for the development of cutting-edge AI solutions.
    
    In conclusion, staying informed about the latest trends and key players in artificial intelligence is essential for anyone looking to navigate the rapidly evolving landscape of AI. By exploring the latest advancements, understanding the key players driving innovation, and staying abreast of noteworthy news in AI, individuals can harness the power of this transformative technology to drive positive change in their respective fields. Let's continue to explore the exciting world of artificial intelligence and unlock its full potential for the benefit of society.
    
    [00m
    [1m[95m [DEBUG]: == Working Agent: Editor[00m
    [1m[95m [INFO]: == Starting Task: Proofread the given blog post for grammatical errors and alignment with the brand's voice.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m


    [32;1m[1;3mI now can give a great answer
    
    Final Answer: 
    
    # Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    ## Introduction
    Artificial intelligence (AI) has seen remarkable advancements in recent years, particularly in deep learning and neural networks. Staying informed about the latest trends in AI is essential for tech enthusiasts, industry professionals, entrepreneurs, and students alike. By keeping up with the dynamic field of AI, individuals can harness its potential to drive innovation and create impactful solutions.
    
    ## Latest Trends in Artificial Intelligence
    AI's applications are expanding rapidly across various industries like healthcare, finance, and retail. This expansion is revolutionizing sectors by enabling predictive analytics, personalized recommendations, and automating repetitive tasks. The integration of AI with Internet of Things (IoT) technology is also creating new opportunities for interconnected smart devices and systems. Another important trend is the emergence of explainable AI, which aims to improve transparency and accountability in AI decision-making processes.
    
    ## Key Players in Artificial Intelligence
    Leading the charge in AI research and development are key players such as Google's DeepMind, known for its groundbreaking work in machine learning and artificial general intelligence. Microsoft's Azure AI platform offers robust business solutions, empowering organizations to enhance efficiency and productivity with AI. IBM Watson's cognitive computing services provide advanced AI capabilities for data analysis and natural language processing. Additionally, startups like OpenAI and Element AI are driving innovation and expanding the possibilities of artificial intelligence.
    
    ## Noteworthy News in Artificial Intelligence
    While AI continues to advance rapidly, ethical concerns surrounding its development and deployment are becoming more prominent. As AI technologies become more widespread, there is a growing need for regulatory efforts to govern their use and address potential risks. Collaborations between tech giants and research institutions are also driving AI innovation, creating a collaborative ecosystem for cutting-edge AI solutions.
    
    In conclusion, staying informed about the latest trends and key players in artificial intelligence is crucial for navigating the evolving landscape of AI. By exploring the latest advancements, understanding the key players shaping innovation, and staying informed about noteworthy news in AI, individuals can leverage this transformative technology for positive change in their respective fields. Let's continue to delve into the exciting world of artificial intelligence and unlock its full potential for the benefit of society.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Editor] Task output: # Unveiling the Latest Trends and Key Players in Artificial Intelligence
    
    ## Introduction
    Artificial intelligence (AI) has seen remarkable advancements in recent years, particularly in deep learning and neural networks. Staying informed about the latest trends in AI is essential for tech enthusiasts, industry professionals, entrepreneurs, and students alike. By keeping up with the dynamic field of AI, individuals can harness its potential to drive innovation and create impactful solutions.
    
    ## Latest Trends in Artificial Intelligence
    AI's applications are expanding rapidly across various industries like healthcare, finance, and retail. This expansion is revolutionizing sectors by enabling predictive analytics, personalized recommendations, and automating repetitive tasks. The integration of AI with Internet of Things (IoT) technology is also creating new opportunities for interconnected smart devices and systems. Another important trend is the emergence of explainable AI, which aims to improve transparency and accountability in AI decision-making processes.
    
    ## Key Players in Artificial Intelligence
    Leading the charge in AI research and development are key players such as Google's DeepMind, known for its groundbreaking work in machine learning and artificial general intelligence. Microsoft's Azure AI platform offers robust business solutions, empowering organizations to enhance efficiency and productivity with AI. IBM Watson's cognitive computing services provide advanced AI capabilities for data analysis and natural language processing. Additionally, startups like OpenAI and Element AI are driving innovation and expanding the possibilities of artificial intelligence.
    
    ## Noteworthy News in Artificial Intelligence
    While AI continues to advance rapidly, ethical concerns surrounding its development and deployment are becoming more prominent. As AI technologies become more widespread, there is a growing need for regulatory efforts to govern their use and address potential risks. Collaborations between tech giants and research institutions are also driving AI innovation, creating a collaborative ecosystem for cutting-edge AI solutions.
    
    In conclusion, staying informed about the latest trends and key players in artificial intelligence is crucial for navigating the evolving landscape of AI. By exploring the latest advancements, understanding the key players shaping innovation, and staying informed about noteworthy news in AI, individuals can leverage this transformative technology for positive change in their respective fields. Let's continue to delve into the exciting world of artificial intelligence and unlock its full potential for the benefit of society.
    
    [00m


- Display the results of your execution as markdown in the notebook.


```python
from IPython.display import Markdown
Markdown(result)
```




# Unveiling the Latest Trends and Key Players in Artificial Intelligence

## Introduction
Artificial intelligence (AI) has seen remarkable advancements in recent years, particularly in deep learning and neural networks. Staying informed about the latest trends in AI is essential for tech enthusiasts, industry professionals, entrepreneurs, and students alike. By keeping up with the dynamic field of AI, individuals can harness its potential to drive innovation and create impactful solutions.

## Latest Trends in Artificial Intelligence
AI's applications are expanding rapidly across various industries like healthcare, finance, and retail. This expansion is revolutionizing sectors by enabling predictive analytics, personalized recommendations, and automating repetitive tasks. The integration of AI with Internet of Things (IoT) technology is also creating new opportunities for interconnected smart devices and systems. Another important trend is the emergence of explainable AI, which aims to improve transparency and accountability in AI decision-making processes.

## Key Players in Artificial Intelligence
Leading the charge in AI research and development are key players such as Google's DeepMind, known for its groundbreaking work in machine learning and artificial general intelligence. Microsoft's Azure AI platform offers robust business solutions, empowering organizations to enhance efficiency and productivity with AI. IBM Watson's cognitive computing services provide advanced AI capabilities for data analysis and natural language processing. Additionally, startups like OpenAI and Element AI are driving innovation and expanding the possibilities of artificial intelligence.

## Noteworthy News in Artificial Intelligence
While AI continues to advance rapidly, ethical concerns surrounding its development and deployment are becoming more prominent. As AI technologies become more widespread, there is a growing need for regulatory efforts to govern their use and address potential risks. Collaborations between tech giants and research institutions are also driving AI innovation, creating a collaborative ecosystem for cutting-edge AI solutions.

In conclusion, staying informed about the latest trends and key players in artificial intelligence is crucial for navigating the evolving landscape of AI. By exploring the latest advancements, understanding the key players shaping innovation, and staying informed about noteworthy news in AI, individuals can leverage this transformative technology for positive change in their respective fields. Let's continue to delve into the exciting world of artificial intelligence and unlock its full potential for the benefit of society.



## Try it Yourself

- Pass in a topic of your choice and see what the agents come up with!


```python
topic = "Embodied AI"
result = crew.kickoff(inputs={"topic": topic})
```

    [1m[95m [DEBUG]: == Working Agent: Content Planner[00m
    [1m[95m [INFO]: == Starting Task: 1. Prioritize the latest trends, key players, and noteworthy news on Embodied AI.
    2. Identify the target audience, considering their interests and pain points.
    3. Develop a detailed content outline including an introduction, key points, and a call to action.
    4. Include SEO keywords and relevant data or sources.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer
    
    Final Answer: 
    
    **Content Plan: Embodied AI**
    
    **Introduction:**
    - Brief explanation of Embodied AI and its significance in the field of artificial intelligence.
    - Mention of key players and recent trends in the industry.
    
    **Key Points:**
    1. **Latest Trends in Embodied AI:**
       - Integration of sensory perception and motor control in AI systems.
       - Advancements in robotics and virtual agents with physical bodies.
       - Impact of Embodied AI on industries such as healthcare, manufacturing, and entertainment.
    
    2. **Key Players in Embodied AI:**
       - Companies like Boston Dynamics, SoftBank Robotics, and Hanson Robotics leading the way in developing embodied AI technologies.
       - Research institutions and universities contributing to the advancement of Embodied AI.
    
    3. **Noteworthy News on Embodied AI:**
       - Recent breakthroughs in human-robot interaction and collaboration.
       - Ethical considerations and challenges in the development of Embodied AI technologies.
    
    **Target Audience:**
    - Tech enthusiasts, researchers, industry professionals, and students interested in artificial intelligence and robotics.
    - Pain points include staying updated on the latest developments in AI, understanding the implications of Embodied AI on society, and seeking opportunities to learn more about this emerging field.
    
    **Content Outline:**
    1. Introduction to Embodied AI
    2. Latest Trends in Embodied AI
    3. Key Players in the Industry
    4. Noteworthy News and Developments
    5. Ethical Considerations and Future Implications
    6. Call to Action: Resources for further reading and learning about Embodied AI
    
    **SEO Keywords:**
    - Embodied AI, artificial intelligence with physical bodies, robotics, human-robot interaction, AI trends, key players in AI, AI news, ethical AI, future of AI.
    
    **Resources:**
    1. "Embodied AI: A Brief Overview" - Stanford University AI Lab
    2. "The Next Frontier in AI: Embodied Intelligence" - MIT Technology Review
    3. "Robotics and AI: The Future of Human-Machine Collaboration" - World Economic Forum
    
    By following this comprehensive content plan, the blog article on Embodied AI will engage the target audience with relevant information, insights on the latest trends, and valuable resources for further exploration of this exciting field.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Content Planner] Task output: **Content Plan: Embodied AI**
    
    **Introduction:**
    - Brief explanation of Embodied AI and its significance in the field of artificial intelligence.
    - Mention of key players and recent trends in the industry.
    
    **Key Points:**
    1. **Latest Trends in Embodied AI:**
       - Integration of sensory perception and motor control in AI systems.
       - Advancements in robotics and virtual agents with physical bodies.
       - Impact of Embodied AI on industries such as healthcare, manufacturing, and entertainment.
    
    2. **Key Players in Embodied AI:**
       - Companies like Boston Dynamics, SoftBank Robotics, and Hanson Robotics leading the way in developing embodied AI technologies.
       - Research institutions and universities contributing to the advancement of Embodied AI.
    
    3. **Noteworthy News on Embodied AI:**
       - Recent breakthroughs in human-robot interaction and collaboration.
       - Ethical considerations and challenges in the development of Embodied AI technologies.
    
    **Target Audience:**
    - Tech enthusiasts, researchers, industry professionals, and students interested in artificial intelligence and robotics.
    - Pain points include staying updated on the latest developments in AI, understanding the implications of Embodied AI on society, and seeking opportunities to learn more about this emerging field.
    
    **Content Outline:**
    1. Introduction to Embodied AI
    2. Latest Trends in Embodied AI
    3. Key Players in the Industry
    4. Noteworthy News and Developments
    5. Ethical Considerations and Future Implications
    6. Call to Action: Resources for further reading and learning about Embodied AI
    
    **SEO Keywords:**
    - Embodied AI, artificial intelligence with physical bodies, robotics, human-robot interaction, AI trends, key players in AI, AI news, ethical AI, future of AI.
    
    **Resources:**
    1. "Embodied AI: A Brief Overview" - Stanford University AI Lab
    2. "The Next Frontier in AI: Embodied Intelligence" - MIT Technology Review
    3. "Robotics and AI: The Future of Human-Machine Collaboration" - World Economic Forum
    
    By following this comprehensive content plan, the blog article on Embodied AI will engage the target audience with relevant information, insights on the latest trends, and valuable resources for further exploration of this exciting field.
    
    [00m
    [1m[95m [DEBUG]: == Working Agent: Content Writer[00m
    [1m[95m [INFO]: == Starting Task: 1. Use the content plan to craft a compelling blog post on Embodied AI.
    2. Incorporate SEO keywords naturally.
    3. Sections/Subtitles are properly named in an engaging manner.
    4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.
    5. Proofread for grammatical errors and alignment with the brand's voice.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer
    
    Final Answer:
    
    # Exploring the Future of Artificial Intelligence: The Rise of Embodied AI
    
    **Introduction**
    
    Artificial Intelligence (AI) has undergone remarkable advancements in recent years, with one of the most intriguing developments being the emergence of Embodied AI. This innovative approach integrates sensory perception and motor control into AI systems, allowing them to interact with the physical world in a more human-like manner. Key players in the industry, such as Boston Dynamics, SoftBank Robotics, and Hanson Robotics, have been at the forefront of developing technologies that bridge the gap between the virtual and physical realms. In this blog post, we delve into the latest trends, key players, noteworthy news, and ethical considerations surrounding Embodied AI.
    
    **Latest Trends in Embodied AI**
    
    The integration of sensory perception and motor control in AI systems has opened up new possibilities for the field. Advancements in robotics and virtual agents equipped with physical bodies have paved the way for applications in various industries, including healthcare, manufacturing, and entertainment. These technologies are not only enhancing efficiency and productivity but also transforming the way humans interact with machines. The impact of Embodied AI on society is profound, as we witness the blurring of boundaries between humans and machines.
    
    **Key Players in the Industry**
    
    Companies like Boston Dynamics, known for their groundbreaking work in robotics, and SoftBank Robotics, a pioneer in developing humanoid robots, are leading the charge in the realm of Embodied AI. Research institutions and universities are also playing a crucial role in pushing the boundaries of what is possible in this field. Their collaborative efforts are driving innovation and shaping the future of AI technologies. The expertise and resources invested by these key players are instrumental in driving the momentum of Embodied AI forward.
    
    **Noteworthy News and Developments**
    
    Recent breakthroughs in human-robot interaction and collaboration have captured the attention of the AI community. From robots assisting in surgical procedures to virtual agents enhancing customer service experiences, the potential applications of Embodied AI are vast. However, with these advancements come ethical considerations and challenges. As AI technologies become more integrated into our daily lives, questions surrounding privacy, security, and accountability must be addressed. It is essential to navigate these complexities thoughtfully and responsibly to ensure a future where AI benefits society as a whole.
    
    **Ethical Considerations and Future Implications**
    
    As we look towards the future of Embodied AI, it is crucial to consider the ethical implications of these technologies. Ensuring that AI systems are designed and implemented ethically is paramount to building trust and acceptance among users. The potential societal impact of Embodied AI is vast, from revolutionizing healthcare to transforming the way we work and play. By addressing ethical concerns and actively engaging in discussions on the future implications of AI, we can steer the development of Embodied AI towards a more inclusive and beneficial future for all.
    
    **Call to Action**
    
    For those interested in delving deeper into the world of Embodied AI, there are numerous resources available for further reading and learning. From academic papers to industry reports, there is a wealth of information waiting to be explored. Stay informed on the latest trends, engage in discussions on ethical AI practices, and be part of shaping the future of artificial intelligence. The journey towards a more embodied and intelligent AI future begins with each of us taking an active interest in this evolving field.
    
    In conclusion, the rise of Embodied AI represents a significant milestone in the evolution of artificial intelligence. By embracing the integration of physical bodies into AI systems, we are opening up new possibilities for human-machine interaction and collaboration. As we navigate the challenges and opportunities presented by Embodied AI, let us strive to build a future where AI technologies enhance our lives and empower us to create a more inclusive and sustainable world.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Content Writer] Task output: # Exploring the Future of Artificial Intelligence: The Rise of Embodied AI
    
    **Introduction**
    
    Artificial Intelligence (AI) has undergone remarkable advancements in recent years, with one of the most intriguing developments being the emergence of Embodied AI. This innovative approach integrates sensory perception and motor control into AI systems, allowing them to interact with the physical world in a more human-like manner. Key players in the industry, such as Boston Dynamics, SoftBank Robotics, and Hanson Robotics, have been at the forefront of developing technologies that bridge the gap between the virtual and physical realms. In this blog post, we delve into the latest trends, key players, noteworthy news, and ethical considerations surrounding Embodied AI.
    
    **Latest Trends in Embodied AI**
    
    The integration of sensory perception and motor control in AI systems has opened up new possibilities for the field. Advancements in robotics and virtual agents equipped with physical bodies have paved the way for applications in various industries, including healthcare, manufacturing, and entertainment. These technologies are not only enhancing efficiency and productivity but also transforming the way humans interact with machines. The impact of Embodied AI on society is profound, as we witness the blurring of boundaries between humans and machines.
    
    **Key Players in the Industry**
    
    Companies like Boston Dynamics, known for their groundbreaking work in robotics, and SoftBank Robotics, a pioneer in developing humanoid robots, are leading the charge in the realm of Embodied AI. Research institutions and universities are also playing a crucial role in pushing the boundaries of what is possible in this field. Their collaborative efforts are driving innovation and shaping the future of AI technologies. The expertise and resources invested by these key players are instrumental in driving the momentum of Embodied AI forward.
    
    **Noteworthy News and Developments**
    
    Recent breakthroughs in human-robot interaction and collaboration have captured the attention of the AI community. From robots assisting in surgical procedures to virtual agents enhancing customer service experiences, the potential applications of Embodied AI are vast. However, with these advancements come ethical considerations and challenges. As AI technologies become more integrated into our daily lives, questions surrounding privacy, security, and accountability must be addressed. It is essential to navigate these complexities thoughtfully and responsibly to ensure a future where AI benefits society as a whole.
    
    **Ethical Considerations and Future Implications**
    
    As we look towards the future of Embodied AI, it is crucial to consider the ethical implications of these technologies. Ensuring that AI systems are designed and implemented ethically is paramount to building trust and acceptance among users. The potential societal impact of Embodied AI is vast, from revolutionizing healthcare to transforming the way we work and play. By addressing ethical concerns and actively engaging in discussions on the future implications of AI, we can steer the development of Embodied AI towards a more inclusive and beneficial future for all.
    
    **Call to Action**
    
    For those interested in delving deeper into the world of Embodied AI, there are numerous resources available for further reading and learning. From academic papers to industry reports, there is a wealth of information waiting to be explored. Stay informed on the latest trends, engage in discussions on ethical AI practices, and be part of shaping the future of artificial intelligence. The journey towards a more embodied and intelligent AI future begins with each of us taking an active interest in this evolving field.
    
    In conclusion, the rise of Embodied AI represents a significant milestone in the evolution of artificial intelligence. By embracing the integration of physical bodies into AI systems, we are opening up new possibilities for human-machine interaction and collaboration. As we navigate the challenges and opportunities presented by Embodied AI, let us strive to build a future where AI technologies enhance our lives and empower us to create a more inclusive and sustainable world.
    
    [00m
    [1m[95m [DEBUG]: == Working Agent: Editor[00m
    [1m[95m [INFO]: == Starting Task: Proofread the given blog post for grammatical errors and alignment with the brand's voice.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m


    [32;1m[1;3mI now can give a great answer
    
    Final Answer: 
    
    # Exploring the Future of Artificial Intelligence: The Rise of Embodied AI
    
    **Introduction**
    
    Artificial Intelligence (AI) has undergone remarkable advancements in recent years, with one of the most intriguing developments being the emergence of Embodied AI. This innovative approach integrates sensory perception and motor control into AI systems, allowing them to interact with the physical world in a more human-like manner. Key players in the industry, such as Boston Dynamics, SoftBank Robotics, and Hanson Robotics, have been at the forefront of developing technologies that bridge the gap between the virtual and physical realms. In this blog post, we delve into the latest trends, key players, noteworthy news, and ethical considerations surrounding Embodied AI.
    
    **Latest Trends in Embodied AI**
    
    The integration of sensory perception and motor control in AI systems has opened up new possibilities for the field. Advancements in robotics and virtual agents equipped with physical bodies have paved the way for applications in various industries, including healthcare, manufacturing, and entertainment. These technologies are not only enhancing efficiency and productivity but also transforming the way humans interact with machines. The impact of Embodied AI on society is profound, as we witness the blurring of boundaries between humans and machines.
    
    **Key Players in the Industry**
    
    Companies like Boston Dynamics, known for their groundbreaking work in robotics, and SoftBank Robotics, a pioneer in developing humanoid robots, are leading the charge in the realm of Embodied AI. Research institutions and universities are also playing a crucial role in pushing the boundaries of what is possible in this field. Their collaborative efforts are driving innovation and shaping the future of AI technologies. The expertise and resources invested by these key players are instrumental in driving the momentum of Embodied AI forward.
    
    **Noteworthy News and Developments**
    
    Recent breakthroughs in human-robot interaction and collaboration have captured the attention of the AI community. From robots assisting in surgical procedures to virtual agents enhancing customer service experiences, the potential applications of Embodied AI are vast. However, with these advancements come ethical considerations and challenges. As AI technologies become more integrated into our daily lives, questions surrounding privacy, security, and accountability must be addressed. It is essential to navigate these complexities thoughtfully and responsibly to ensure a future where AI benefits society as a whole.
    
    **Ethical Considerations and Future Implications**
    
    As we look towards the future of Embodied AI, it is crucial to consider the ethical implications of these technologies. Ensuring that AI systems are designed and implemented ethically is paramount to building trust and acceptance among users. The potential societal impact of Embodied AI is vast, from revolutionizing healthcare to transforming the way we work and play. By addressing ethical concerns and actively engaging in discussions on the future implications of AI, we can steer the development of Embodied AI towards a more inclusive and beneficial future for all.
    
    **Call to Action**
    
    For those interested in delving deeper into the world of Embodied AI, there are numerous resources available for further reading and learning. From academic papers to industry reports, there is a wealth of information waiting to be explored. Stay informed on the latest trends, engage in discussions on ethical AI practices, and be part of shaping the future of artificial intelligence. The journey towards a more embodied and intelligent AI future begins with each of us taking an active interest in this evolving field.
    
    In conclusion, the rise of Embodied AI represents a significant milestone in the evolution of artificial intelligence. By embracing the integration of physical bodies into AI systems, we are opening up new possibilities for human-machine interaction and collaboration. As we navigate the challenges and opportunities presented by Embodied AI, let us strive to build a future where AI technologies enhance our lives and empower us to create a more inclusive and sustainable world.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Editor] Task output: # Exploring the Future of Artificial Intelligence: The Rise of Embodied AI
    
    **Introduction**
    
    Artificial Intelligence (AI) has undergone remarkable advancements in recent years, with one of the most intriguing developments being the emergence of Embodied AI. This innovative approach integrates sensory perception and motor control into AI systems, allowing them to interact with the physical world in a more human-like manner. Key players in the industry, such as Boston Dynamics, SoftBank Robotics, and Hanson Robotics, have been at the forefront of developing technologies that bridge the gap between the virtual and physical realms. In this blog post, we delve into the latest trends, key players, noteworthy news, and ethical considerations surrounding Embodied AI.
    
    **Latest Trends in Embodied AI**
    
    The integration of sensory perception and motor control in AI systems has opened up new possibilities for the field. Advancements in robotics and virtual agents equipped with physical bodies have paved the way for applications in various industries, including healthcare, manufacturing, and entertainment. These technologies are not only enhancing efficiency and productivity but also transforming the way humans interact with machines. The impact of Embodied AI on society is profound, as we witness the blurring of boundaries between humans and machines.
    
    **Key Players in the Industry**
    
    Companies like Boston Dynamics, known for their groundbreaking work in robotics, and SoftBank Robotics, a pioneer in developing humanoid robots, are leading the charge in the realm of Embodied AI. Research institutions and universities are also playing a crucial role in pushing the boundaries of what is possible in this field. Their collaborative efforts are driving innovation and shaping the future of AI technologies. The expertise and resources invested by these key players are instrumental in driving the momentum of Embodied AI forward.
    
    **Noteworthy News and Developments**
    
    Recent breakthroughs in human-robot interaction and collaboration have captured the attention of the AI community. From robots assisting in surgical procedures to virtual agents enhancing customer service experiences, the potential applications of Embodied AI are vast. However, with these advancements come ethical considerations and challenges. As AI technologies become more integrated into our daily lives, questions surrounding privacy, security, and accountability must be addressed. It is essential to navigate these complexities thoughtfully and responsibly to ensure a future where AI benefits society as a whole.
    
    **Ethical Considerations and Future Implications**
    
    As we look towards the future of Embodied AI, it is crucial to consider the ethical implications of these technologies. Ensuring that AI systems are designed and implemented ethically is paramount to building trust and acceptance among users. The potential societal impact of Embodied AI is vast, from revolutionizing healthcare to transforming the way we work and play. By addressing ethical concerns and actively engaging in discussions on the future implications of AI, we can steer the development of Embodied AI towards a more inclusive and beneficial future for all.
    
    **Call to Action**
    
    For those interested in delving deeper into the world of Embodied AI, there are numerous resources available for further reading and learning. From academic papers to industry reports, there is a wealth of information waiting to be explored. Stay informed on the latest trends, engage in discussions on ethical AI practices, and be part of shaping the future of artificial intelligence. The journey towards a more embodied and intelligent AI future begins with each of us taking an active interest in this evolving field.
    
    In conclusion, the rise of Embodied AI represents a significant milestone in the evolution of artificial intelligence. By embracing the integration of physical bodies into AI systems, we are opening up new possibilities for human-machine interaction and collaboration. As we navigate the challenges and opportunities presented by Embodied AI, let us strive to build a future where AI technologies enhance our lives and empower us to create a more inclusive and sustainable world.
    
    [00m



```python
Markdown(result)
```




# Exploring the Future of Artificial Intelligence: The Rise of Embodied AI

**Introduction**

Artificial Intelligence (AI) has undergone remarkable advancements in recent years, with one of the most intriguing developments being the emergence of Embodied AI. This innovative approach integrates sensory perception and motor control into AI systems, allowing them to interact with the physical world in a more human-like manner. Key players in the industry, such as Boston Dynamics, SoftBank Robotics, and Hanson Robotics, have been at the forefront of developing technologies that bridge the gap between the virtual and physical realms. In this blog post, we delve into the latest trends, key players, noteworthy news, and ethical considerations surrounding Embodied AI.

**Latest Trends in Embodied AI**

The integration of sensory perception and motor control in AI systems has opened up new possibilities for the field. Advancements in robotics and virtual agents equipped with physical bodies have paved the way for applications in various industries, including healthcare, manufacturing, and entertainment. These technologies are not only enhancing efficiency and productivity but also transforming the way humans interact with machines. The impact of Embodied AI on society is profound, as we witness the blurring of boundaries between humans and machines.

**Key Players in the Industry**

Companies like Boston Dynamics, known for their groundbreaking work in robotics, and SoftBank Robotics, a pioneer in developing humanoid robots, are leading the charge in the realm of Embodied AI. Research institutions and universities are also playing a crucial role in pushing the boundaries of what is possible in this field. Their collaborative efforts are driving innovation and shaping the future of AI technologies. The expertise and resources invested by these key players are instrumental in driving the momentum of Embodied AI forward.

**Noteworthy News and Developments**

Recent breakthroughs in human-robot interaction and collaboration have captured the attention of the AI community. From robots assisting in surgical procedures to virtual agents enhancing customer service experiences, the potential applications of Embodied AI are vast. However, with these advancements come ethical considerations and challenges. As AI technologies become more integrated into our daily lives, questions surrounding privacy, security, and accountability must be addressed. It is essential to navigate these complexities thoughtfully and responsibly to ensure a future where AI benefits society as a whole.

**Ethical Considerations and Future Implications**

As we look towards the future of Embodied AI, it is crucial to consider the ethical implications of these technologies. Ensuring that AI systems are designed and implemented ethically is paramount to building trust and acceptance among users. The potential societal impact of Embodied AI is vast, from revolutionizing healthcare to transforming the way we work and play. By addressing ethical concerns and actively engaging in discussions on the future implications of AI, we can steer the development of Embodied AI towards a more inclusive and beneficial future for all.

**Call to Action**

For those interested in delving deeper into the world of Embodied AI, there are numerous resources available for further reading and learning. From academic papers to industry reports, there is a wealth of information waiting to be explored. Stay informed on the latest trends, engage in discussions on ethical AI practices, and be part of shaping the future of artificial intelligence. The journey towards a more embodied and intelligent AI future begins with each of us taking an active interest in this evolving field.

In conclusion, the rise of Embodied AI represents a significant milestone in the evolution of artificial intelligence. By embracing the integration of physical bodies into AI systems, we are opening up new possibilities for human-machine interaction and collaboration. As we navigate the challenges and opportunities presented by Embodied AI, let us strive to build a future where AI technologies enhance our lives and empower us to create a more inclusive and sustainable world.



<a name='1'></a>
 ## Other Popular Models as LLM for your Agents

#### Hugging Face (HuggingFaceHub endpoint)

```Python
from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token="<HF_TOKEN_HERE>",
    task="text-generation",
)

### you will pass "llm" to your agent function
```

#### Mistral API

```Python
OPENAI_API_KEY=your-mistral-api-key
OPENAI_API_BASE=https://api.mistral.ai/v1
OPENAI_MODEL_NAME="mistral-small"
```

#### Cohere

```Python
from langchain_community.chat_models import ChatCohere
# Initialize language model
os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
llm = ChatCohere()

### you will pass "llm" to your agent function
```

### For using Llama locally with Ollama and more, checkout the crewAI documentation on [Connecting to any LLM](https://docs.crewai.com/how-to/LLM-Connections/).


```python

```


```python

```


```python

```
