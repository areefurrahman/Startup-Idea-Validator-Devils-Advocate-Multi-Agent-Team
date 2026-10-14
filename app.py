import os

# Turn telemetry off BEFORE crewai is imported.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACKING"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import re
import time

import plotly.graph_objects as go
import streamlit as st
from crewai import LLM, Crew, Process

from critic_agent import create_critic_agent, create_critique_task
from researcher_agent import create_research_task, create_researcher_agent
from strategist_agent import create_strategist_agent, create_strategy_task
from tools import (
    get_last_economics,
    get_search_count,
    reset_tool_state,
    unit_economics_calculator,
    web_search_tool,
)

st.set_page_config(page_title="Startup Idea Validator", page_icon="🧨", layout="wide")


def build_llm(api_key: str) -> LLM:
    return LLM(
        model="openai/openai/gpt-oss-120b",
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        temperature=0.3,
        max_tokens=3000,
    )


# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
.hero {background: linear-gradient(135deg,#0f172a 0%,#312e81 55%,#7c3aed 100%);
       padding: 2.2rem 2rem; border-radius: 18px; margin-bottom: 1.2rem; color: #fff;}
.hero h1 {margin: 0 0 .4rem 0; font-size: 2.1rem; color: #fff;}
.hero p {margin: 0 0 1rem 0; color: #cbd5e1; font-size: 1.02rem;}
.badge {display:inline-block; padding:.35rem .8rem; margin:0 .5rem .4rem 0; border-radius:999px;
        background: rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28);
        font-size:.88rem; color:#fff;}
.acard {padding:.8rem .9rem; border-radius:12px; margin-bottom:.7rem;
        background: rgba(124,58,237,.10); border:1px solid rgba(124,58,237,.35);}
.acard b {font-size:.98rem;}
.acard small {opacity:.85;}
.step {padding:.8rem 1rem; border-radius:12px; margin-bottom:.55rem; font-size:1rem;
       border:1px solid rgba(128,128,128,.3);}
.step.pending {opacity:.55;}
.step.running {background: rgba(124,58,237,.14); border-color:#7c3aed; animation: pulse 1.4s infinite;}
.step.done {background: rgba(34,197,94,.13); border-color:#22c55e;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(124,58,237,.5);} 70%{box-shadow:0 0 0 10px rgba(124,58,237,0);} 100%{box-shadow:0 0 0 0 rgba(124,58,237,0);}}
</style>
<div class="hero">
  <h1>🧨 Startup Idea Validator &amp; Devil's Advocate</h1>
  <p>Three AI agents research your market, tear your idea apart, then build you a survival plan.</p>
  <span class="badge">🕵️ Market Detective</span>
  <span class="badge">😈 Devil's Advocate VC</span>
  <span class="badge">🧭 Pivot Strategist</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = ""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("Groq API key loaded from Secrets ✅")
    except Exception:
        api_key = st.text_input("Groq API key", type="password", help="Get a free key at console.groq.com")
    st.caption("Model: openai/gpt-oss-120b on Groq")

    st.header("🤖 The Crew")
    st.markdown(
        """
<div class="acard"><b>🕵️ Agent 1: Market Detective</b><br>
<small>Finds real competitors, pricing and market saturation.<br>
🔧 Tool: <code>web_search_tool</code> (live search, max 4 per run)</small></div>
<div class="acard"><b>😈 Agent 2: Devil's Advocate VC</b><br>
<small>Finds fatal flaws and customer acquisition traps.<br>
🔧 Tool: <code>unit_economics_calculator</code></small></div>
<div class="acard"><b>🧭 Agent 3: Pivot Strategist</b><br>
<small>Builds the survival plan, 30-day MVP roadmap and viability score.<br>
🔧 Tool: none (reasoning only)</small></div>
""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- inputs
PRESETS = {
    "AI Resume Tailor for Graduates": {
        "idea": "An AI web app that tailors a graduate's resume and cover letter to each job posting in under 2 minutes.",
        "market": "Fresh university graduates in the US and UK",
        "price": 12.0, "cac": 25.0,
    },
    "Subscription Specialty Coffee Box": {
        "idea": "A monthly subscription box of freshly roasted single-origin specialty coffee from small roasters, delivered to your door.",
        "market": "Coffee enthusiasts aged 25-40 in the US",
        "price": 35.0, "cac": 45.0,
    },
    "B2B WhatsApp Order Bot for Restaurants": {
        "idea": "A WhatsApp chatbot that takes food orders from customers and sends them straight to the restaurant's kitchen.",
        "market": "Small and mid-size restaurants in emerging markets (Pakistan, UAE, India)",
        "price": 49.0, "cac": 120.0,
    },
}
CHOOSE = "Choose a demo idea..."

st.session_state.setdefault("idea_in", "")
st.session_state.setdefault("market_in", "")
st.session_state.setdefault("price_in", 29.0)
st.session_state.setdefault("cac_in", 40.0)


def apply_preset():
    p = PRESETS.get(st.session_state["preset"])
    if p:
        st.session_state["idea_in"] = p["idea"]
        st.session_state["market_in"] = p["market"]
        st.session_state["price_in"] = p["price"]
        st.session_state["cac_in"] = p["cac"]


st.subheader("💡 Your idea")
st.selectbox("⚡ Quick Demo Ideas (1-click test)", [CHOOSE] + list(PRESETS), key="preset", on_change=apply_preset)
st.text_area("Startup idea description", key="idea_in", height=110,
             placeholder="Describe what you want to build and who it helps...")
c1, c2, c3 = st.columns([2, 1, 1])
c1.text_input("Target market / region", key="market_in", placeholder="e.g. Small Shopify stores in the US")
c2.number_input("Expected monthly price ($)", min_value=0.0, step=1.0, key="price_in")
c3.number_input("Est. customer acquisition cost ($)", min_value=0.0, step=1.0, key="cac_in")
run_clicked = st.button("🔥 Validate my idea", type="primary", width="stretch")

# ---------------------------------------------------------------- live tracker
STEPS = [
    ("🕵️", "Agent 1 (Market Detective)", "is scanning live competitors..."),
    ("😈", "Agent 2 (Devil's Advocate VC)", "is stress-testing business flaws & unit economics..."),
    ("🧭", "Agent 3 (Pivot Strategist)", "is crafting the survival roadmap..."),
]


def render_tracker(placeholder, states):
    html = ""
    for (icon, name, doing), state in zip(STEPS, states):
        if state == "done":
            html += f'<div class="step done">✅ {icon} {name} — Done</div>'
        elif state == "running":
            html += f'<div class="step running">{icon} {name} {doing}</div>'
        else:
            html += f'<div class="step pending">⏳ {icon} {name} — waiting</div>'
    placeholder.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------- run the crew
if run_clicked:
    idea = st.session_state["idea_in"].strip()
    market = st.session_state["market_in"].strip()
    price = float(st.session_state["price_in"])
    cac = float(st.session_state["cac_in"])

    if not api_key:
        st.error("Add your Groq API key in the sidebar (or in Streamlit Secrets).")
    elif not idea or not market or price <= 0:
        st.error("Please fill in the idea, the target market and a price above 0.")
    else:
        st.session_state.pop("result", None)
        reset_tool_state(idea, market)

        llm = build_llm(api_key)
        researcher = create_researcher_agent(llm, [web_search_tool])
        critic = create_critic_agent(llm, [unit_economics_calculator])
        strategist = create_strategist_agent(llm)

        research_task = create_research_task(researcher, idea, market, price)
        critique_task = create_critique_task(critic, idea, market, price, research_task, cac=cac)
        strategy_task = create_strategy_task(strategist, idea, research_task, critique_task)

        st.subheader("🛰️ Live Agent Tracker")
        tracker = st.empty()
        states = ["running", "pending", "pending"]
        render_tracker(tracker, states)
        finished = [0]

        def on_task_done(_output):
            i = finished[0]
            states[i] = "done"
            if i + 1 < len(states):
                states[i + 1] = "running"
            finished[0] += 1
            render_tracker(tracker, states)

        crew = Crew(
            agents=[researcher, critic, strategist],
            tasks=[research_task, critique_task, strategy_task],
            process=Process.sequential,
            task_callback=on_task_done,
            verbose=False,
        )

        started = time.time()
        try:
            crew.kickoff()
            st.session_state["result"] = {
                "idea": idea, "market": market, "price": price, "cac": cac,
                "research": research_task.output.raw,
                "critique": critique_task.output.raw,
                "strategy": strategy_task.output.raw,
                "economics": get_last_economics(),
                "searches": get_search_count(),
                "seconds": round(time.time() - started),
            }
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                st.error("Groq says you hit the rate limit (too many tokens per minute). "
                         "Wait about 60 seconds and click the button again.")
            elif "401" in msg or "api key" in msg.lower():
                st.error("Groq rejected the API key. Check that it is correct.")
            else:
                st.error(f"Something went wrong: {msg[:400]}")

# ---------------------------------------------------------------- results
res = st.session_state.get("result")
if res:
    strategy_raw = res["strategy"]
    m = re.search(r"VIABILITY_SCORE:\s*\**\s*(\d{1,3})", strategy_raw)
    score = min(int(m.group(1)), 100) if m else None
    strategy_clean = re.sub(r"\**VIABILITY_SCORE:.*", "", strategy_raw).strip()

    if score is None:
        verdict = "n/a"
    elif score >= 70:
        verdict = "Promising"
    elif score >= 40:
        verdict = "Needs a pivot"
    else:
        verdict = "High risk"

    eco = res.get("economics") or {}
    st.divider()
    st.subheader("📊 Validation Dashboard")

    left, right = st.columns([1, 1.3])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score if score is not None else 0,
            title={"text": "Viability Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#7c3aed"},
                "steps": [
                    {"range": [0, 40], "color": "#fecaca"},
                    {"range": [40, 70], "color": "#fde68a"},
                    {"range": [70, 100], "color": "#bbf7d0"},
                ],
            },
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        a, b = st.columns(2)
        a.metric("Verdict", verdict)
        b.metric("Financial risk", eco.get("risk", "n/a"))
        c, d = st.columns(2)
        c.metric("Break-even customers", f"{eco['breakeven']:,}" if eco.get("breakeven") is not None else "n/a")
        d.metric("LTV:CAC", eco["ltv_cac"] if eco.get("ltv_cac") is not None else "n/a")
        st.caption(f"Live searches used: {res['searches']}/4 · Run time: {res['seconds']}s")

    if score is None:
        st.warning("The strategist did not give a clean score line. Read the plan in Tab 3.")

    t1, t2, t3 = st.tabs([
        "🕵️ Market Detective Report",
        "😈 Devil's Advocate Roast",
        "🧭 Pivot & Launch Strategy",
    ])
    t1.markdown(res["research"])
    t2.markdown(res["critique"])
    t3.markdown(strategy_clean)

    report = f"""# Startup Validation Report

**Idea:** {res['idea']}
**Target market:** {res['market']}
**Monthly price:** ${res['price']:,.2f} | **Estimated CAC:** ${res['cac']:,.2f}
**Viability score:** {score if score is not None else 'n/a'}/100 ({verdict})

---

## 🕵️ Market Detective Report

{res['research']}

---

## 😈 Devil's Advocate Roast

{res['critique']}

---

## 🧭 Pivot & Launch Strategy

{strategy_clean}
"""
    st.download_button("⬇️ Download full report (.md)", report,
                       file_name="startup_validation_report.md", mime="text/markdown")
