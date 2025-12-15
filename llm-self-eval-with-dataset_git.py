# pip install langgraph langchain-openai python-dotenv
# .env: AZURE_OPENAI_API_KEY=...

from __future__ import annotations

import os
import json
import re
import uuid
import difflib
from typing import Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# =========================
# 0) DEBUG / VERBOSE
# =========================
VERBOSE = True
SHOW_FULL_PROMPTS = False


# =========================
# 0.1) EXAMPLES (DICT input -> output)
# =========================
# Sem si dej vlastní páry; node select_example vybere nejpodobnější vstupu.
EXAMPLE_PAIRS: dict[str, str] = {
    "Nemám je": "Nemám je",
    "Prodají mi ho": "Prodají mi ho",
    "Jsem nesoustředěný v práci.": "Jsem nesoustředěný",
    "Přijdeme pozdě": "Přijdeme pozdě",
    "V práci nestíhám termíny.": "Nestíhám to",
    "Hádáme se doma kvůli penězům.": "Hádáme se",
    "Mám toho moc": "Mám toho moc",
    "Že dělám chyby.": "Dělám chyby",
    "Nemůžu je najít": "Nemůžu je najít",
    "Neovládám se": "Neovládám se",
    "Když se najím, je mi těžko": "Je mi těžko",
    "Že je tu nepořádek.": "Je tu nepořádek",
    "Nevycházíme spolu.": "Nevycházíme spolu",
    "Budu muset": "Budu muset",
    "Pořád se musím starat o děti.": "Musím se o ně starat",
    "Maminka by byla smutná.": "Je smutná",
    "Neumím se ovládat.": "Neumím se ovládat.",
    "Nevím jak na to": "Nevím jak na to",
    "Zapomněl jsem.": "Zapomněl jsem.",
    "Nestíhám to": "Nestíhám to",
    "Nevím, jak si říct o pomoc.": "Nevím jak si o to říct",
    "připadám si neschopně": "Jsem neschopný",
    "Nestihnu svoji práci": "Nestihnu to",
    "Nemůžu to dělat podle sebe.": "Nemůžu to dělat podle sebe.",
    "Nedělám to co bych měl.": "Nedělám to co bych měl.",
    "Nikdo mě nebude chtít.": "Nikdo mě nebude chtít.",
    "Nebudou mě potřebovat.": "Nebudou mě potřebovat.",
    "Mám strach, že se rozhodnu špatně.": "Mám strach, že se rozhodnu špatně.",
    "Musím se rychle rozhodnout.": "Musím se rychle rozhodnout.",
    "No že to ztrácím.": "Ztrácím to",
    "Zapomínám na to.": "Zapomínám na to",
}


# =========================
# 1) KONFIG + PROMPTY (NECHÁVÁM BEZE ZMĚNY)
# =========================

load_dotenv()

HUMAN_INPUT_FIELD = "question"

GEN_SYSTEM_PROMPT = """
Jsi jazykový model, který AKTIVNĚ kondenzuje české věty.
Tvým cílem je odstranit nadbytečný kontext a vysvětlení,
ale vždy zachovat význam a emoční jádro.

Tvým úkolem NENÍ ponechávat věty beze změny,
pokud existuje bezpečný způsob, jak je zestručnit.
""".strip()

GEN_USER_TEMPLATE = """
Postupuj vždy ve DVOU KROCÍCH (kroky NEPOPISUJ):

KROK 1 – AKTIVNĚ HLEDEJ ZKRÁCENÍ
Zkus aplikovat alespoň jedno z pravidel níže,
pokud je to možné bez změny významu.

KROK 2 – BEZPEČNOSTNÍ KONTROLA
Pokud by zkrácení změnilo význam nebo emoci,
vrať původní větu.

────────────────────────
PRAVIDLO 1 – Odstraň zjevný kontext (kde / kdy / s kým).
Příklad:
Vstup: Jsem nesoustředěný v práci.
Výstup: Jsem nesoustředěný

────────────────────────
PRAVIDLO 2 – Odstraň úvodní markery („že“, „protože“, „nic“, „no že“),
pokud nejsou významové.
Příklad:
Vstup: Že dělám chyby.
Výstup: Dělám chyby

────────────────────────
PRAVIDLO 3 – Nahraď objekt zájmenem (to / ho / ji / je / ně),
pokud objekt není identitní, vztahový ani hodnotový.
PŘÍKLADY:
Nemám auto. → Nemám to
Bolí mě noha. → Bolí mě to

NEPOUŽÍT:
Nemůžu to dělat podle sebe. → beze změny

────────────────────────
PRAVIDLO 4 – Zkrať vysvětlení na jednoznačné emoční jádro.
Příklad:
Jde o to vaření, manžel se na mě zlobí. → Zlobí se na mě

────────────────────────
PRAVIDLO 5 – Použij běžnou zkrácenou formulaci, pokud existuje.
Příklad:
Mám moc úkolů. → Mám toho moc.

────────────────────────
OMEZENÍ:
• Použij maximálně 2–3 pravidla.
• Nikdy neměň osobu, čas ani emoci.
• Nepřidávej nové informace.
• Vrať právě jednu českou větu.

────────────────────────
Vstup:
"{INPUT}"

Výstup:

{feedback_block}
""".strip()

EVAL_SYSTEM_PROMPT = """
Jsi hodnoticí jazykový model (LLM-judge).
Tvým úkolem je posoudit, zda výstup zachovává význam
a dodržuje transformační pravidla.
Buď přísný.
""".strip()

EVAL_USER_TEMPLATE = """
Zhodnoť transformaci vstupní věty na výstupní.

Vstup:
"{INPUT}"

Výstup:
"{OUTPUT}"

Odpověz ve STRUKTUROVANÉM JSONU přesně v tomto formátu:

{
  "verdict": "PASS" | "FAIL",
  "reasons": [
    "stručný důvod selhání nebo prázdné pole"
  ],
  "violations": [
    "změna významu",
    "ztráta emoce",
    "neoprávněné zkrácení",
    "nesprávné zájmeno",
    "přidání informace",
    "jiné"
  ]
}

KRITÉRIA HODNOCENÍ:
• Význam a emoční jádro musí zůstat zachovány.
• Osoba a čas se nesmí změnit.
• Výstup nesmí obsahovat nové informace.
• Zkrácení je POVOLENO, pokud je významově bezpečné.
• Beze změny je přijatelné, pouze pokud neexistuje zjevná bezpečná kondenzace.
• Příliš agresivní zkrácení, které zahodí podstatnou část sdělení, je chyba.

DŮLEŽITÁ VÝJIMKA (NEPENALIZOVAT):
• Odstranění diskurzních/úvodních markerů (např. „že“, „no že“, „protože“, „nic“, „no že to…“)
  se NEPOVAŽUJE za „ztrátu emoce“, pokud zůstane stejné jádrové tvrzení.
  Příklad správně: „No že to ztrácím“ → „Ztrácím to“.

POKYNY:
• Nehodnoť negativně samotný fakt, že došlo ke změně.
• „Ztráta emoce“ nastává jen tehdy, když se oslabí nebo změní emoční obsah
  (např. odstraní se slova jako „bojím se“, „je mi těžko“, „mám strach“, „jsem smutný“, „vadí mi“, apod.).
• Pokud je výstup kratší, jasnější a významově shodný, považuj jej za správný.

Pokud je vše v pořádku, nastav:
"verdict": "PASS"
a "reasons" i "violations" ponech prázdné.
""".strip()

MAX_ATTEMPTS = 5

# ===== CHECKPOINTER =====
checkpointer = InMemorySaver()

# ===== LLMS (AZURE) =====
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="",
    azure_endpoint="",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.2,
)

evaluator_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-12-01-preview",
    azure_endpoint="",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.0,
)


# =========================
# 2) STATE
# =========================
class State(TypedDict, total=False):
    question: str
    answer: str
    ok: bool
    attempts: int
    final: str

    # komunikace eval -> generate
    advice: str

    # debug
    judge: dict

    # nové: vybraný příklad pro generate
    example_input: str
    example_output: str
    example_score: float


# =========================
# 3) HELPERS
# =========================
def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Evaluator nevrátil JSON. Raw: {text[:300]}")
    return json.loads(m.group(0))


def _render_prompt(template: str, *, INPUT: str = "", OUTPUT: str = "", feedback_block: str = "") -> str:
    return (
        template
        .replace("{INPUT}", INPUT)
        .replace("{OUTPUT}", OUTPUT)
        .replace("{feedback_block}", feedback_block)
    )


def _dbg(title: str, text: str):
    if not VERBOSE:
        return
    print(f"\n===== {title} =====")
    if SHOW_FULL_PROMPTS:
        print(text)
    else:
        snippet = text if len(text) <= 600 else (text[:600] + "\n... [TRUNCATED]")
        print(snippet)


def _advice_from_judge(data: dict) -> str:
    violations = data.get("violations") or []
    reasons = data.get("reasons") or []

    vset = {str(v).strip().lower() for v in violations if str(v).strip()}
    advice_lines: list[str] = []

    advice_lines.append("Zkrať méně agresivně; když si nejsi jistý, vrať původní větu beze změny.")

    if "změna významu" in vset or "ztráta emoce" in vset:
        advice_lines.append("Zachovej význam i emoční tón; neměň osobu ani čas.")
    if "přidání informace" in vset:
        advice_lines.append("Odstraň jakoukoliv informaci, která nebyla explicitně ve vstupu.")
    if "nesprávné zájmeno" in vset:
        advice_lines.append("Nepřepisuj objekt na zájmeno, pokud je identitní/vztahový/hodnotový; raději objekt ponech.")
    if "neoprávněné zkrácení" in vset:
        advice_lines.append("Neškrtat klíčové části; ponech to, co nese význam a emoci.")

    if isinstance(reasons, list):
        for r in reasons[:3]:
            r = str(r).strip()
            if r:
                advice_lines.append(f"Judge důvod: {r}")

    # dedup
    seen = set()
    out = []
    for line in advice_lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return "\n".join(out).strip()


def _best_example(query: str) -> tuple[str, str, float]:
    """
    Vybere nejpodobnější příklad z EXAMPLE_PAIRS pomocí SequenceMatcher.
    Vrací (example_input, example_output, score).
    """
    q = query.strip().lower()
    best_in, best_out, best_score = "", "", 0.0

    for ex_in, ex_out in EXAMPLE_PAIRS.items():
        score = difflib.SequenceMatcher(a=q, b=ex_in.strip().lower()).ratio()
        if score > best_score:
            best_in, best_out, best_score = ex_in, ex_out, float(score)

    return best_in, best_out, best_score


def _feedback_block(state: State) -> str:
    """
    Jediný text, který se vloží do {feedback_block}.
    Kombinuje:
    - vybraný příklad (z node select_example)
    - advice od evaluátoru (z node evaluate)
    """
    parts: list[str] = []

    ex_in = (state.get("example_input") or "").strip()
    ex_out = (state.get("example_output") or "").strip()
    ex_score = state.get("example_score", 0.0)

    # příklad přidáme jen pokud to dává smysl (prahově)
    if ex_in and ex_out and ex_score >= 0.45:
        parts.append(
            "Příklad podobné transformace (použij jako vodítko stylu, ne jako obsah):\n"
            f'Vstup: "{ex_in}"\n'
            f'Výstup: "{ex_out}"\n'
        )

    advice = (state.get("advice") or "").strip()
    if advice:
        parts.append("Advice pro úpravu výstupu:\n" + advice)
    else:
        parts.append("Advice pro úpravu výstupu:\n(není)")

    return "\n\n".join(parts).strip()


# =========================
# 4) NODES
# =========================
def select_example(state: State) -> State:
    """
    NOVÝ NODE:
    Vybere nejlepší (input -> output) příklad z EXAMPLE_PAIRS pro aktuální vstup.
    """
    ex_in, ex_out, score = _best_example(state["question"])

    if VERBOSE:
        print("\n[select_example]")
        print(f'[select_example] INPUT = {state["question"]}')
        if ex_in:
            print(f'[select_example] picked example (score={score:.2f}): "{ex_in}" -> "{ex_out}"')
        else:
            print("[select_example] no example available")

    return {
        "example_input": ex_in,
        "example_output": ex_out,
        "example_score": score,
    }


def generate(state: State) -> State:
    attempts = int(state.get("attempts", 0)) + 1

    system = SystemMessage(content=GEN_SYSTEM_PROMPT)
    user_text = _render_prompt(
        GEN_USER_TEMPLATE,
        INPUT=state["question"],
        feedback_block=_feedback_block(state),
    )
    user = HumanMessage(content=user_text)

    if VERBOSE:
        print(f"\n[generate] attempt={attempts}")
        print(f"[generate] INPUT  = {state['question']}")
        print(f"[generate] advice = {state.get('advice','(none)')}")
        if state.get("example_input"):
            print(
                f'[generate] example(score={state.get("example_score",0.0):.2f}) = '
                f'"{state.get("example_input")}" -> "{state.get("example_output")}"'
            )

    _dbg("generate.user_message (to LLM)", user_text)

    resp = llm.invoke([system, user])

    if VERBOSE:
        print(f"[generate] OUTPUT = {resp.content}")

    return {"attempts": attempts, "answer": resp.content}


def evaluate(state: State) -> State:
    system = SystemMessage(content=EVAL_SYSTEM_PROMPT)
    user_text = _render_prompt(
        EVAL_USER_TEMPLATE,
        INPUT=state["question"],
        OUTPUT=state.get("answer", ""),
    )
    user = HumanMessage(content=user_text)

    if VERBOSE:
        print(f"\n[evaluate] evaluating attempt={state.get('attempts')}")
        print(f"[evaluate] INPUT  = {state['question']}")
        print(f"[evaluate] OUTPUT = {state.get('answer','')}")

    _dbg("evaluate.user_message (to Judge)", user_text)

    try:
        resp = evaluator_llm.invoke([system, user])
        if VERBOSE:
            _dbg("evaluate.raw_judge_response", resp.content)

        data = _extract_json(resp.content)

        verdict = str(data.get("verdict", "")).strip().upper()
        ok = verdict == "PASS"

        if ok:
            if VERBOSE:
                print("[evaluate] verdict=PASS -> advice cleared")
            return {"ok": True, "advice": "", "judge": data}

        advice = _advice_from_judge(data) or "Zkrať opatrněji; zachovej význam a emoci."
        if VERBOSE:
            print("[evaluate] verdict=FAIL -> creating advice for next generate()")
            _dbg("evaluate.advice (to next generate)", advice)

        return {"ok": False, "advice": advice, "judge": data}

    except Exception as e:
        advice = (
            f"Evaluator error: {type(e).__name__}: {e}. "
            "V dalším pokusu vrať raději původní větu beze změny."
        )
        if VERBOSE:
            print("[evaluate] ERROR -> forcing FAIL with advice")
            _dbg("evaluate.advice (from error)", advice)
        return {"ok": False, "advice": advice}


def route_after_eval(state: State) -> Literal["output", "select_example", "fallback"]:
    if state.get("ok") is True:
        return "output"
    if int(state.get("attempts", 0)) < MAX_ATTEMPTS:
        # před dalším generate znovu vybereme příklad (může být stejný, ale můžeš to dál vylepšit)
        return "select_example"
    return "fallback"


def output(state: State) -> State:
    return {"final": state.get("answer", "")}


def fallback(state: State) -> State:
    return {
        "final": (
            f"Fallback: nepodařilo se projít evaluací ani po {MAX_ATTEMPTS} pokusech.\n\n"
            f"Poslední odpověď:\n{state.get('answer','')}\n\n"
            f"Poslední advice:\n{state.get('advice','')}"
        )
    }


# =========================
# 5) GRAPH
# =========================
def build_graph():
    g = StateGraph(State)

    # NOVÉ: node pro výběr příkladu
    g.add_node("select_example", select_example)

    g.add_node("generate", generate)
    g.add_node("evaluate", evaluate)
    g.add_node("output", output)
    g.add_node("fallback", fallback)

    # Start -> select_example -> generate -> evaluate
    g.add_edge(START, "select_example")
    g.add_edge("select_example", "generate")
    g.add_edge("generate", "evaluate")

    # při FAIL: evaluate -> select_example -> generate (loop)
    g.add_conditional_edges("evaluate", route_after_eval)

    g.add_edge("output", END)
    g.add_edge("fallback", END)

    return g.compile(checkpointer=checkpointer)


# =========================
# 6) RUN (TERMINÁL INPUT LOOP)
# =========================
if __name__ == "__main__":
    app = build_graph()

    print("LangGraph demo (exit/quit/q = konec).")
    while True:
        q = input("\nZadej větu: ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            print("Konec.")
            break

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = app.invoke({HUMAN_INPUT_FIELD: q}, config=config)

        print("\n--- FINAL ---\n")
        print(result["final"])
        print("\n(meta) attempts:", result.get("attempts"), "ok:", result.get("ok"))
