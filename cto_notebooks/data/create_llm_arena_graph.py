# %%

import pickle
import urllib.request
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go

from cto_notebooks.utils.config import CONFIG

# %%

# https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard/tree/main
TARGET_DATE = "20250128"
TARGET_FILE = f"elo_results_{TARGET_DATE}.pkl"
DATE_STR = datetime.strptime(TARGET_DATE, "%Y%m%d").strftime("Stand %d.%m.%y")
url = (
    "https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard/"
    f"resolve/main/{TARGET_FILE}?download=true"
)

if not Path(CONFIG.cache_dir / TARGET_FILE).exists():
    if not url.startswith(("http:", "https:")):
        msg = "URL must start with 'http:' or 'https:'"
        raise ValueError(msg)
    # S310 is handled by the if condition above
    urllib.request.urlretrieve(url, CONFIG.cache_dir / TARGET_FILE)  # noqa: S310

with open(CONFIG.cache_dir / TARGET_FILE, "rb") as f:
    elo_results = pickle.load(f)  # noqa: S301
# %%
ratings = elo_results["text"]["full"]["elo_rating_final"]
prop = [
    "chatgpt-4o-latest-20241120",
    "gemini-2.0-flash-thinking-exp-01-21",
    "claude-3-5-sonnet-20241022",
    "gpt-4-turbo-2024-04-09",
    "mistral-large-2407",
    "gpt-3.5-turbo-0613",
]

not_properietary = ["deepseek-r1", "llama-3.1-405b-instruct-bf16", "vicuna-7b"]

# %%
fig = go.Figure()
fig.add_trace(go.Bar(name="proprietary", x=prop, y=ratings[prop]))
fig.add_trace(go.Bar(name="open", x=not_properietary, y=ratings[not_properietary]))
fig.update_yaxes(
    title_text='<span style="font-size: 1.5rem;">Elo rating</span>', range=[900, 1400]
)
fig.update_xaxes(tickangle=20, categoryorder="total descending")
fig.update_layout(
    title=(
        '<span style="font-size: 2rem;">Chatbot Arena</span><br>'
        f"<sup>{DATE_STR}</sup>"
    ),
    font={"size": 20},
    barmode="group",
    height=500,
    width=1200,
    # margin=dict(l=20, r=20, t=60, b=20),
)
for b in fig.data:
    b.x = tuple("-".join(t.split("-")[:2]) for t in b.x)
fig.show()
fig.write_image(f"{CONFIG.cache_dir}/elo_ratings_{TARGET_DATE}.png")


# %%
