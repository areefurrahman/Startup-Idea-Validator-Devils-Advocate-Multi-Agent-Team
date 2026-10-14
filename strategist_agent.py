from crewai import Agent, Task


def create_strategist_agent(llm) -> Agent:
    return Agent(
        role="The Pivot & Launch Strategist",
        goal=(
            "Turn the research and the critique into a practical Survival & Pivot "
            "Plan, a 30-day MVP roadmap and an honest Viability Score."
        ),
        backstory=(
            "You are a pragmatic startup operator. You respect the critic's hard "
            "points and the detective's facts. You do not flatter. A low score is "
            "fine when the evidence says so."
        ),
        llm=llm,
        allow_delegation=False,
        max_iter=4,
        verbose=False,
    )


def create_strategy_task(agent, idea: str, research_task: Task, critique_task: Task) -> Task:
    return Task(
        description=(
            f"STARTUP IDEA: {idea}\n\n"
            "Read the Market Detective's report and the Devil's Advocate's critique. "
            "Then write these sections:\n"
            "## Survival & Pivot Plan\n2-3 concrete options with the trade-off of each.\n"
            "## 30-Day MVP Roadmap\nWeek 1 to Week 4, small steps that can be tested.\n"
            "## Final Verdict\nTwo or three sentences, honest and direct.\n\n"
            "The LAST line of your answer must be exactly in this format:\n"
            "VIABILITY_SCORE: XX/100\n"
            "(XX is a whole number from 0 to 100.)\n\n"
            "Keep the whole answer between 300 and 400 words. Use simple words."
        ),
        expected_output=(
            "A pivot plan, a 30-day roadmap and a verdict (300-400 words), ending with "
            "the line 'VIABILITY_SCORE: XX/100'."
        ),
        agent=agent,
        context=[research_task, critique_task],
    )
