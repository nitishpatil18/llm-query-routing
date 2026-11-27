import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv(".env")

MODELS = {"small": "openai/gpt-oss-20b", "large": "openai/gpt-oss-120b"}
PRICE = {  # USD per token (prompt, completion), from the provider's listing
    "small": (0.000000075, 0.0000003),
    "large": (0.00000015, 0.0000006),
}
CAP = 4096

st.set_page_config(page_title="Cost-aware LLM cascade", layout="wide")


@st.cache_resource
def get_client():
    return Groq(api_key=os.environ["GROQ_API_KEY"])


@st.cache_data
def load_bench():
    d = np.load("data/router_features_v2.npz", allow_pickle=True)
    tiers = [str(t) for t in d["tiers"]]
    S, L = tiers.index("small"), tiers.index("large")
    c = d["correct"]
    return (c[:, S].astype(float), c[:, L].astype(float),
            d["cost"][:, S], d["cost"][:, L], d["ctok"][:, S])


def call(tier, question, cap=CAP):
    r = get_client().chat.completions.create(
        model=MODELS[tier],
        messages=[{"role": "user", "content": question}],
        max_completion_tokens=cap,
    )
    pp, cp = PRICE[tier]
    ch = r.choices[0]
    return {
        "text": ch.message.content or "",
        "finish": ch.finish_reason,
        "ptok": r.usage.prompt_tokens,
        "ctok": r.usage.completion_tokens,
        "cost": r.usage.prompt_tokens * pp + r.usage.completion_tokens * cp,
    }


st.title("Adaptive cost-aware LLM query routing")
st.caption("Cascade: call the small model first; escalate to the large model only if the small one runs out of its token budget without answering.")

with st.sidebar:
    st.header("Settings")
    mult = st.slider("Hypothetical large-model price multiplier", 1, 20, 1,
                     help="Scales the large model's price to show how the saving changes if the big model were costlier. 1 = real Groq prices.")
    compare = st.checkbox("Also run the large model on every question (measured baseline, uses quota)")
    demo_cap = st.slider("Small model token cap (demo only)", 128, CAP, CAP, step=128,
                         help="The frozen rule uses 4096. Lower it to force the small model to run out of budget and watch the escalation.")
    st.write("Small: openai/gpt-oss-20b")
    st.write("Large: openai/gpt-oss-120b")

tab1, tab2 = st.tabs(["Live cascade", "Benchmark evidence"])

with tab1:
    q = st.text_area("Question", height=120, placeholder="Ask anything, e.g. a math word problem or a knowledge question")
    if st.button("Run cascade", type="primary") and q.strip():
        try:
            with st.spinner("Calling the small model..."):
                s = call("small", q, demo_cap)
            escalate = s["finish"] == "length"
            big = None
            if escalate or compare:
                with st.spinner("Calling the large model..."):
                    big = call("large", q)
        except Exception as e:
            code = getattr(e, "status_code", None)
            if code == 429:
                st.error("Rate limit reached (429): the daily token quota for one model is used up. Wait and retry.")
            else:
                st.error(f"API call failed ({type(e).__name__}, status {code}).")
            st.stop()

        if escalate:
            st.warning(f"The small model hit the {demo_cap}-token cap without finishing, so the rule escalates to the large model.")
        else:
            st.success("The small model finished within the cap, so its answer is kept.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Small model")
            st.write(f"{s['ctok']} completion tokens | finish: {s['finish']} | cost ${s['cost']:.6f}")
            st.write(s["text"] or "(no final answer: the token budget ran out while reasoning)")
        with c2:
            st.subheader("Large model")
            if big is None:
                st.write("Not called.")
            else:
                tag = "used for the final answer" if escalate else "reference only"
                st.write(f"{big['ctok']} completion tokens | finish: {big['finish']} | cost ${big['cost']:.6f} ({tag})")
                st.write(big["text"] or "(no final answer: the token budget ran out while reasoning)")

        casc = s["cost"] + (mult * big["cost"] if escalate else 0.0)
        m1, m2, m3 = st.columns(3)
        m1.metric("Cascade cost", f"${casc:.6f}")
        if big is not None:
            always_large = mult * big["cost"]
            m2.metric("Always-large cost (measured)", f"${always_large:.6f}")
            m3.metric("Saved vs always-large", f"{(1 - casc / always_large) * 100:.1f}%")
        st.caption(f"Large-model price multiplier in use: x{mult}" + (" (hypothetical)" if mult != 1 else " (real prices)"))

with tab2:
    cs, cl, ps, pl, cts = load_bench()
    n = len(cs)
    trunc = cts >= CAP
    st.caption(f"{n} benchmark questions answered by both models (MMLU, MMLU-Pro, GSM8K, MATH-500 numeric). Truncated answers count as wrong.")
    acc_c = np.where(trunc, cl, cs).mean()

    def policy_costs(m):
        return {
            "always-small": ps.sum(),
            "always-large": m * pl.sum(),
            "cascade (escalate if capped)": ps.sum() + m * pl[trunc].sum(),
        }

    costs = policy_costs(mult)
    accs = {"always-small": cs.mean(), "always-large": cl.mean(), "cascade (escalate if capped)": acc_c}
    rows = [{"policy": k, "accuracy": f"{accs[k]:.3f}", "cost (USD)": f"{costs[k]:.4f}",
             "saved vs always-large": f"{(1 - costs[k] / costs['always-large']) * 100:.1f}%"} for k in costs]
    st.subheader(f"Results at large-model price x{mult}")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    a, b, c = st.columns(3)
    a.metric("Small truncated", f"{trunc.sum()} of {n}", f"{trunc.mean() * 100:.1f}%", delta_color="off")
    b.metric("Large correct on those", f"{cl[trunc].mean() * 100:.0f}%" if trunc.any() else "n/a")
    c.metric("Cost ratio large/small", f"{mult * pl.sum() / ps.sum():.1f}x")

    grid = list(range(1, 21))
    chart = pd.DataFrame({k: [policy_costs(m)[k] for m in grid] for k in costs}, index=pd.Index(grid, name="large price multiplier"))
    st.subheader("Total cost vs large-model price (hypothetical multiplier)")
    st.line_chart(chart)
    st.info("At real prices the large model is only about 1.3x the cost of the small one, so the saving is small. As the price gap grows, the cascade's saving grows. The multiplier is hypothetical; accuracy and token counts are the measured values.")
