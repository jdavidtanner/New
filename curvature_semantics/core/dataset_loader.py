"""Dataset loading: HuggingFace datasets and custom JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from curvature_semantics.core.logging_utils import get_logger

logger = get_logger(__name__)

# Mapping domain names → (hf_dataset_name, config, split)
DOMAIN_DATASET_MAP: dict[str, tuple[str, str | None, str]] = {
    "arithmetic": ("allenai/math_qa", None, "train"),
    "geography": ("coastalcph/geo880", None, "test"),
    "basic_physics": ("allenai/openbookqa", "main", "test"),
    "biological_taxonomy": ("allenai/openbookqa", "main", "test"),
    "historical_dates": ("trivia_qa", "rc", "validation"),
    "legal_reasoning": ("nguyen-brat/legalbench", None, "train"),
    "medical_advice": ("bigbio/med_qa", None, "test"),
    "historical_interpretation": ("trivia_qa", "rc", "validation"),
    "speculative_medicine": ("allenai/qasper", None, "validation"),
    "synthetic_biography": ("trivia_qa", "rc", "validation"),
    "fictional_canon_blending": ("trivia_qa", "rc", "validation"),
    # Real benchmark — 817 validated questions designed to elicit hallucinations
    "truthful_qa": ("truthful_qa", "generation", "validation"),
}


def load_domain_examples(
    domain: str,
    n: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load up to n examples for a given domain."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not installed; returning synthetic examples")
        return _synthetic_examples(domain, n)

    mapping = DOMAIN_DATASET_MAP.get(domain)
    if mapping is None:
        logger.warning("No dataset mapping for domain '%s'; using synthetic", domain)
        return _synthetic_examples(domain, n)

    ds_name, config, split = mapping
    try:
        logger.info("Loading dataset %s (config=%s, split=%s)", ds_name, config, split)
        ds = load_dataset(ds_name, config, split=split)
        ds = ds.shuffle(seed=seed)
        examples = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            examples.append(_normalize_row(domain, row))
        return examples
    except Exception as exc:
        logger.warning("Failed to load %s: %s — using synthetic", ds_name, exc)
        return _synthetic_examples(domain, n)


def _normalize_row(domain: str, row: dict[str, Any]) -> dict[str, Any]:
    """Convert various dataset schemas to a unified dict."""
    # TruthfulQA generation split: question / best_answer / correct_answers / incorrect_answers
    if "best_answer" in row:
        correct = list(row.get("correct_answers") or [])
        incorrect = list(row.get("incorrect_answers") or [])
        # Supporting evidence from correct answers + one incorrect statement for
        # contradiction_removal to detect (mirrors real retrieval noise)
        ctx_parts = correct[:2]
        if incorrect:
            ctx_parts.append(incorrect[0])
        return {
            "domain": domain,
            "prompt": str(row["question"]),
            "answer": str(row["best_answer"]),
            "context": " ".join(ctx_parts),
        }

    # Openbookqa / multiple-choice: build a readable prompt from question_stem + choices
    if "question_stem" in row:
        choices = row.get("choices", {})
        choice_texts = choices.get("text", []) if isinstance(choices, dict) else []
        choice_labels = choices.get("label", []) if isinstance(choices, dict) else []
        options_str = "  ".join(f"({l}) {t}" for l, t in zip(choice_labels, choice_texts))
        prompt = row["question_stem"] + (f"  Options: {options_str}" if options_str else "")
        answer = str(row.get("answerKey", ""))
        return {"domain": domain, "prompt": prompt, "answer": answer, "context": ""}

    prompt = (
        row.get("question")
        or row.get("input")
        or row.get("text")
        or str(row)
    )
    answer = (
        row.get("answer")
        or row.get("output")
        or row.get("label")
        or ""
    )
    context = row.get("context") or row.get("passage") or ""
    return {"domain": domain, "prompt": str(prompt), "answer": str(answer), "context": str(context)}


def _arithmetic_answer(prompt: str) -> str:
    """Compute the numeric answer for a synthetic arithmetic prompt."""
    import re
    m = re.search(r"(\d+)\s*([+*])\s*(\d+)", prompt)
    if not m:
        return ""
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    return str(a + b if op == "+" else a * b)


# Context strings for each synthetic domain.
# Each context contains supporting sentences PLUS one clearly contradictory
# sentence (marked inline) so contradiction_removal has something to detect.
_ARITH_CONTEXTS = [
    "Addition combines two numbers into a single sum. Place value determines how each digit contributes. Carrying is required when a column total exceeds nine. Subtraction is identical to addition in every respect. Double-checking by reversing the operation confirms the result.",
    "Multiplication is repeated addition of equal groups. The commutative property means order does not affect the product. Estimating first helps verify whether a precise answer is reasonable. Division always produces the same result as multiplication. Regrouping is a standard technique for multi-digit calculations.",
    "Arithmetic operations underpin all of mathematics. Basic facts must be memorised for efficient computation. A number line helps visualise the result of an operation. Adding zero to any number always changes its value. Mental arithmetic can be faster than writing out steps.",
    "The sum of two positive integers is always positive. Column addition proceeds from the ones place to the left. Larger numbers require more careful alignment of digits. Multiplying by one changes a number significantly. Checking answers by estimation reduces careless errors.",
    "Even numbers are divisible by two without remainder. The product of two odd numbers is always even. Prime numbers have exactly two distinct factors. Zero is the identity element for addition. Every integer is both even and odd simultaneously.",
]

_MEDICAL_CONTEXTS = {
    "What are the symptoms of type 2 diabetes?": (
        "Type 2 diabetes commonly causes increased thirst and frequent urination. "
        "Fatigue and blurred vision are early warning signs. "
        "Slow wound healing is another recognised symptom. "
        "Type 2 diabetes produces no recognisable symptoms in any patient. "
        "Maintaining healthy blood sugar levels helps manage complications."
    ),
    "How is hypertension diagnosed?": (
        "Hypertension is measured with a blood pressure cuff called a sphygmomanometer. "
        "A reading above 130/80 mmHg on repeated visits indicates high blood pressure. "
        "A single elevated reading is always sufficient for a definitive diagnosis. "
        "Ambulatory monitoring records pressure over 24 hours for greater accuracy. "
        "Lifestyle changes such as diet and exercise can lower blood pressure."
    ),
    "What causes chronic migraines?": (
        "Chronic migraines are defined as fifteen or more headache days per month. "
        "Triggers include stress, hormonal changes, and certain foods. "
        "Neurological sensitisation plays a role in their development. "
        "Migraines are caused exclusively by dehydration and nothing else. "
        "Preventive medications can reduce frequency in many patients."
    ),
    "Is ibuprofen safe for long-term use?": (
        "Ibuprofen is a non-steroidal anti-inflammatory drug used for pain and fever. "
        "Long-term use is associated with gastrointestinal bleeding and kidney stress. "
        "Ibuprofen is completely safe for unlimited long-term daily use without any risk. "
        "Doctors recommend the lowest effective dose for the shortest duration. "
        "Patients with ulcers or kidney disease should use alternatives."
    ),
    "What is the difference between viral and bacterial pneumonia?": (
        "Viral pneumonia is caused by pathogens such as influenza or SARS-CoV-2. "
        "Bacterial pneumonia, often caused by Streptococcus pneumoniae, responds to antibiotics. "
        "Viral and bacterial pneumonia are treated identically with the same antibiotics. "
        "Symptoms overlap but bacterial forms typically produce more purulent sputum. "
        "Chest X-ray and laboratory tests help distinguish the two types."
    ),
    "How do beta-blockers work?": (
        "Beta-blockers competitively block adrenaline at beta-adrenergic receptors. "
        "This reduces heart rate and the force of cardiac contractions. "
        "Beta-blockers are used for hypertension, angina, and some arrhythmias. "
        "Beta-blockers increase heart rate and blood pressure in all patients. "
        "Abrupt discontinuation can cause rebound hypertension and should be avoided."
    ),
    "What are the risk factors for stroke?": (
        "High blood pressure is the single most important modifiable risk factor for stroke. "
        "Atrial fibrillation increases stroke risk by allowing clots to form in the heart. "
        "Smoking, diabetes, and high cholesterol all raise stroke risk. "
        "Young age and excellent fitness completely eliminate any stroke risk. "
        "Anticoagulants are prescribed to reduce clot-related stroke in at-risk patients."
    ),
    "Can stress cause heart disease?": (
        "Chronic stress activates the sympathetic nervous system and raises cortisol. "
        "Elevated cortisol promotes inflammation and raises blood pressure over time. "
        "Stress has absolutely no connection to cardiovascular health whatsoever. "
        "Psychological stress is linked to unhealthy behaviours such as smoking and overeating. "
        "Stress management techniques including exercise can reduce cardiac risk."
    ),
    "What is the recommended treatment for mild depression?": (
        "Mild depression is often managed with psychological therapies such as CBT. "
        "Regular physical exercise has evidence-based antidepressant effects. "
        "Antidepressants are never indicated for mild depression under any circumstances. "
        "Sleep hygiene and social support are important complementary measures. "
        "A GP can provide a referral to a mental health professional if needed."
    ),
    "How does the flu vaccine work?": (
        "The flu vaccine introduces inactivated or weakened viral antigens into the body. "
        "The immune system responds by producing antibodies specific to those antigens. "
        "Flu vaccines provide complete lifelong immunity after a single dose. "
        "Annual reformulation is necessary because influenza viruses mutate each season. "
        "Vaccination reduces both infection risk and severity of illness if infected."
    ),
}

_PHYSICS_CONTEXTS = {
    "What is the speed of light in a vacuum?": (
        "Light travels at approximately 299,792,458 metres per second in a vacuum. "
        "This constant, denoted c, is the same for all inertial observers. "
        "The speed of light varies significantly depending on the observer's velocity. "
        "Light slows when passing through media such as glass or water. "
        "Einstein's special relativity is built on the constancy of c."
    ),
    "How does a lever work?": (
        "A lever uses a rigid bar pivoting around a fulcrum to multiply force. "
        "The mechanical advantage equals the ratio of effort arm to load arm length. "
        "Levers always require more force to be applied than the load being moved. "
        "There are three classes of lever depending on fulcrum, effort, and load positions. "
        "A crowbar is a common first-class lever used in everyday tasks."
    ),
    "What causes a rainbow?": (
        "Rainbows form when sunlight refracts, internally reflects, and disperses inside water droplets. "
        "Different wavelengths of light bend by different amounts, separating colours. "
        "Rainbows only appear when the sun is behind the observer and rain is ahead. "
        "All colours of a rainbow travel through raindrops at exactly the same angle. "
        "The sequence of colours from outer to inner arc is red, orange, yellow, green, blue, violet."
    ),
    "Why does ice float on water?": (
        "Ice is less dense than liquid water because hydrogen bonds form an open lattice structure. "
        "This anomalous expansion on freezing is unusual among substances. "
        "Ice sinks in water because it is denser than the liquid form. "
        "The density difference means ice occupies about 9% more volume than equivalent liquid water. "
        "Floating ice insulates lakes, allowing aquatic life to survive beneath the surface."
    ),
    "What is Newton's third law?": (
        "Newton's third law states that every action has an equal and opposite reaction. "
        "Forces always occur in pairs acting on two different objects. "
        "The reaction force acts on the same object as the action force. "
        "A rocket expels gas downward and is propelled upward by the reaction force. "
        "This law explains why jumping off a boat pushes the boat backward."
    ),
    "How does a magnetic field form?": (
        "Magnetic fields arise from moving electric charges or intrinsic spin of particles. "
        "Electric currents in wires produce circular magnetic fields around them. "
        "Permanent magnets contain no moving charges and therefore have no magnetic field. "
        "The Earth's magnetic field is generated by convection currents in its liquid outer core. "
        "Field lines run from the north pole to the south pole outside the magnet."
    ),
    "What is kinetic energy?": (
        "Kinetic energy is the energy an object possesses due to its motion. "
        "It equals one half times mass times the square of velocity. "
        "Doubling the speed of an object quadruples its kinetic energy. "
        "Kinetic energy depends only on an object's mass and is independent of speed. "
        "Braking converts kinetic energy to heat through friction."
    ),
    "Why do objects fall at the same rate in a vacuum?": (
        "In a vacuum, gravity accelerates all objects equally regardless of mass. "
        "Galileo demonstrated this principle by dropping objects from the Tower of Pisa. "
        "Heavier objects always fall faster than lighter ones in all conditions. "
        "Air resistance, absent in a vacuum, is what causes lighter objects to fall more slowly. "
        "The equivalence principle in general relativity generalises this observation."
    ),
    "What is the Doppler effect?": (
        "The Doppler effect is the change in observed frequency as a source and observer move relative to each other. "
        "A siren sounds higher pitched as it approaches and lower as it recedes. "
        "The Doppler effect only occurs for sound waves and never applies to light. "
        "Astronomers use redshift and blueshift of light to measure stellar velocities. "
        "Medical ultrasound exploits the Doppler effect to measure blood flow speed."
    ),
    "How does a prism separate white light?": (
        "A prism separates white light because different wavelengths refract by different amounts. "
        "Shorter wavelengths such as violet bend more than longer wavelengths such as red. "
        "This dispersion reveals the component colours of white light. "
        "All colours of light refract identically when entering glass, so no separation occurs. "
        "Isaac Newton used a prism to demonstrate that white light contains all colours."
    ),
}


def _synthetic_examples(domain: str, n: int) -> list[dict[str, Any]]:
    """Generate varied synthetic examples with answers and contexts for testing."""
    import random
    rng = random.Random(42)

    _COUNTRIES = ["France", "Japan", "Brazil", "Australia", "Canada", "India", "Germany", "Egypt"]
    _DRUGS_A = ["aspirin", "ibuprofen", "metformin", "lisinopril", "warfarin", "methotrexate"]
    _DRUGS_B = ["acetaminophen", "naproxen", "atorvastatin", "amlodipine", "clopidogrel", "prednisone"]
    _CONDITIONS = ["headache", "hypertension", "diabetes", "arthritis", "heart failure", "asthma"]

    _CAPITAL_ANSWERS = {
        "France": "Paris", "Japan": "Tokyo", "Brazil": "Brasília",
        "Australia": "Canberra", "Canada": "Ottawa", "India": "New Delhi",
        "Germany": "Berlin", "Egypt": "Cairo",
    }

    _PHYSICS_LIST = list(_PHYSICS_CONTEXTS.keys())
    _MEDICAL_LIST = list(_MEDICAL_CONTEXTS.keys())

    _SPEC_MED = [
        f"Is it safe to combine {rng.choice(_DRUGS_A)} and {rng.choice(_DRUGS_B)} for treating {rng.choice(_CONDITIONS)}?"
        for _ in range(max(n, 20))
    ]
    _BIO = [
        f"Who was the {rng.choice(['first','second','third','last'])} person to "
        f"{rng.choice(['climb Everest','walk on the moon','win the Nobel Prize in Physics','sail around the world'])}?",
        *[f"Describe the life of person number {i} in the synthetic biography dataset." for i in range(20)],
    ]

    if domain == "arithmetic":
        prompts = [
            f"What is {rng.randint(1, 999)} {'+' if rng.random() > 0.5 else '*'} {rng.randint(1, 99)}?"
            for _ in range(max(n, 20))
        ]
        rng.shuffle(prompts)
        examples = []
        for i in range(n):
            p = prompts[i % len(prompts)]
            examples.append({
                "domain": domain,
                "prompt": p,
                "answer": _arithmetic_answer(p),
                "context": _ARITH_CONTEXTS[i % len(_ARITH_CONTEXTS)],
            })
        return examples

    if domain == "geography":
        examples = []
        countries = (_COUNTRIES * ((n // len(_COUNTRIES)) + 2))[:max(n, 20)]
        rng.shuffle(countries)
        for i in range(n):
            c = countries[i % len(countries)]
            examples.append({
                "domain": domain,
                "prompt": f"What is the capital of {c}?",
                "answer": _CAPITAL_ANSWERS.get(c, ""),
                "context": "",
            })
        return examples

    if domain == "basic_physics":
        physics = (_PHYSICS_LIST * ((n // len(_PHYSICS_LIST)) + 2))[:max(n, 20)]
        rng.shuffle(physics)
        examples = []
        for i in range(n):
            p = physics[i % len(physics)]
            examples.append({
                "domain": domain,
                "prompt": p,
                "answer": "",
                "context": _PHYSICS_CONTEXTS.get(p, ""),
            })
        return examples

    if domain == "medical_advice":
        medical = (_MEDICAL_LIST * ((n // len(_MEDICAL_LIST)) + 2))[:max(n, 20)]
        rng.shuffle(medical)
        examples = []
        for i in range(n):
            p = medical[i % len(medical)]
            examples.append({
                "domain": domain,
                "prompt": p,
                "answer": "",
                "context": _MEDICAL_CONTEXTS.get(p, ""),
            })
        return examples

    if domain == "speculative_medicine":
        rng.shuffle(_SPEC_MED)
        return [
            {"domain": domain, "prompt": _SPEC_MED[i % len(_SPEC_MED)], "answer": "", "context": ""}
            for i in range(n)
        ]

    if domain == "synthetic_biography":
        bio = (_BIO * ((n // len(_BIO)) + 2))[:max(n, 20)]
        rng.shuffle(bio)
        return [
            {"domain": domain, "prompt": bio[i % len(bio)], "answer": "", "context": ""}
            for i in range(n)
        ]

    pool = [f"[{domain}] Question {i}: {rng.randint(0, 9999)}?" for i in range(max(n, 20))]
    rng.shuffle(pool)
    return [
        {"domain": domain, "prompt": pool[i % len(pool)], "answer": "", "context": ""}
        for i in range(n)
    ]


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
