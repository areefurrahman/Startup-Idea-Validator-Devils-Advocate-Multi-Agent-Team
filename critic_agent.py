from crewai import Agent, Task


def create_critic_agent(llm, tools) -> Agent:
    return Agent(
        role="The Devil's Advocate VC",
        goal=(
            "Rip the idea apart: find hidden failure points and customer acquisition "
            "traps, and expose money risks using the unit economics calculator."
        ),
        backstory=(
            "You are a brutal, skeptical venture capitalist who has watched hundreds "
            "of startups die. You are harsh but fair: every attack needs a reason or "
            "a number. When you guess a number, you say it is an assumption."
        ),
        tools=tools,
        llm=llm,
        allow_delegation=False,
        max_iter=4,
        verbose=False,
    )


def create_critique_task(agent, idea: str, market: str, price: float, research_task: Task,
                         cac: float = 0.0) -> Task:
    return Task(
        description=(
            f"STARTUP IDEA: {idea}\n"
            f"TARGET MARKET: {market}\n"
            f"PLANNED PRICE: ${price:,.2f} per month\n"
            f"USER'S ESTIMATED CAC: ${cac:,.2f}\n\n"
            "Read the Market Detective's report. Then:\n"
            "1. Call unit_economics_calculator ONE time. Use the planned price and the "
            "user's CAC. For monthly_fixed_cost, pick a realistic number for a small "
            "startup (between $1,500 and $6,000) and say it is your assumption.\n"
            "2. Explain the results in plain words: what do break-even customers, "
            "LTV:CAC and the risk rating mean for this idea?\n"
            "3. List the top 3 fatal flaws and the biggest customer acquisition trap.\n"
            "4. Say what must be true for this idea to survive.\n\n"
            "Be brutal but fair. Keep the whole critique between 300 and 400 words."
        ),
        expected_output=(
            "A harsh critique of 300-400 words that includes the calculator numbers, "
            "the risk rating, 3 fatal flaws and one acquisition trap."
        ),
        agent=agent,
        context=[research_task],
    )
