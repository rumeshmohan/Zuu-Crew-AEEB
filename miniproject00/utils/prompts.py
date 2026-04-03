import string
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class PromptSpec:
    """Standardized structure for LLM prompts."""
    id: str
    purpose: str
    template: str
    temperature: float = 0.0
    max_tokens: int = 500

class Template(string.Template):
    """Custom template engine allowing for ${var} or $var syntax."""
    delimiter = "$"

PROMPTS: Dict[str, PromptSpec] = {
    # Phase 1: Few-Shot Classification 
    "few_shot.v1": PromptSpec(
        id="few_shot.v1",
        purpose="Strict classification with location validation for Sri Lanka.",
        template=(
            "Role: Emergency Response AI\n"
            "Task: Classify the query. Be precise about Sri Lankan locations.\n\n"
            "KNOWLEDGE BASE (TOWN -> DISTRICT):\n"
            "- Kaduwela, Kolonnawa, Wellampitiya -> Colombo\n"
            "- Wattala, Ja-Ela, Ragama, Kelaniya -> Gampaha\n"
            "- Knuckles -> Matale\n"
            "- Peradeniya -> Kandy\n"
            "- Kitulgala -> Kegalle\n"
            "- Galle, Hikkaduwa -> Galle\n\n"
            "RULES:\n"
            "1. DISTRICT: Look for city names and map to the District above. "
            "If NO city is mentioned, output 'District: None'.\n"
            "DO NOT assume a Words like 'Dengue' or 'Typhoid' or 'washed away' as District.\n"
            "DO NOT assume a District based on landmarks like 'temple', 'highway', or 'river'.\n"
            "2. CONTRACT: Return ONLY the output in this exact format: "
            "District: [Name] | Intent: [Category] | Priority: [Level]\n"
            "3. NO reasoning or explanations.\n\n"
            "Examples:\n$examples\n\n"
            "Message: $query\n"
            "Output:"
        ),
        temperature=0.0,
        max_tokens=100
    ),

    # Phase 2: Chain-of-Thought Stability Experiment 
    "cot_reasoning.v1": PromptSpec(
        id="cot_reasoning.v1",
        purpose="Reasoning-based analysis for complex and ambiguous scenarios.",
        template=(
            "Role: $role\n"
            "Task: Analyze the following crisis scenario using Chain-of-Thought reasoning.\n"
            "Scenario: $problem\n\n"
            "Step-by-Step Analysis:\n"
            "1. Identify the core threat and potential for escalation.\n"
            "2. Extract location details (District). If none found, state 'None'.\n"
            "3. Assess urgency and required resource types.\n"
            "4. Final Decision logic.\n\n"
            "Format:\n"
            "Reasoning: [Your logical steps]\n"
            "Output: District: [Name] | Intent: [Category] | Priority: [High/Low]"
        ),
        temperature=0.0,
        max_tokens=2000
    ),

    # Phase 3 - Step A: Scoring (CoT) 
    "cot_scoring.v1": PromptSpec(
        id="cot_scoring.v1",
        purpose="Column-aware triage scoring.",
        template=(
            "Role: Data Analyst & Triage AI\n"
            "Input Row: $incident\n\n"
            "Step 1: Data Mapping (Extract from columns separated by '|'):\n"
            "- Column 3 (Area): [Value]\n"
            "- Column 5 (Ages): [Value]\n"
            "- Column 6 (Main Need): [Value]\n\n"
            "Step 2: Apply Mission Rules:\n"
            "- Base Score: 5\n"
            "- Age Bonus (+2): Is Age < 5 or > 60?\n"
            "- Rescue Bonus (+3): Is Main Need 'Rescue'?\n"
            "- Medical Bonus (+1): Is Main Need 'Insulin' or 'Medicine'?\n\n"
            "Step 3: Final Math:\n"
            "Math: 5 (Base) + [Age Bonus] + [Need Bonus] = Total\n"
            "Result: Score: X/10"
        ),
        temperature=0.0,
        max_tokens=600
    ),

    # Phase 3 - Stage B: Strategy (Tree-of-Thought) 
    "tot_strategy.v1": PromptSpec(
        id="tot_strategy.v1",
        purpose="Optimal route selection using three logical branches.",
        template=(
            "Role: Logistics Commander\n"
            "Context: One rescue boat at Ragama.\n"
            "Travel: Ragama -> Ja-Ela (10m), Ja-Ela -> Gampaha (40m).\n\n"
            "Scored Incidents:\n$scored_incidents\n\n"
            "Explore 3 Branches:\n"
            "1. Branch 1 (Greedy): Start with highest priority score first.\n"
            "2. Branch 2 (Speed): Start with closest location (Ragama is 0m away).\n"
            "3. Branch 3 (Logistics): Start with furthest location first.\n\n"
            "Task: For each branch, calculate Total Time and Total Priority Score saved.\n\n"
            "Final Conclusion: Summarize the logic and provide the mission directive at the end.\n"
            "Outcome Format:\n"
            "Optimal Route: ID [X] (Location) -> ID [Y] (Location) -> ID [Z] (Location)"
        ),
        temperature=0.0,
        max_tokens=1500
    ),

    # Phase 4: Token Economics 
    "overflow_summarize.v1": PromptSpec(
        id="overflow_summarize.v1",
        purpose="Summarize long/spam messages to stay within token budget.",
        template=(
            "The following message is too long for the system. "
            "Summarize it into 50 words or less, keeping ONLY the "
            "Location, Number of People, and specific Medical/Rescue needs.\n\n"
            "Message: $message"
        ),
        temperature=0.0,
        max_tokens=100
    ),

    # Phase 5: JSON Extraction Pipeline 
    "json_extract.v1": PromptSpec(
        id="json_extract.v1",
        purpose="High-precision JSON extraction for Sri Lankan crisis data.",
        template=(
            "Extract JSON from this news snippet. Use these exact rules:\n"
            "1. DISTRICT: Must be one of [Colombo, Gampaha, Kandy, Kalutara, Galle, Matara, Jaffna, Kegalle, Ratnapura, Nuwara Eliya]. "
            "If a town is mentioned, map it to its parent district from this list.\n"
            "2. FLOOD LEVEL: Float only. If unknown, use null.\n"
            "3. VICTIMS: Integer only. Default to 0.\n"
            "4. STATUS: [Critical, Warning, Stable].\n"
            "5. NEED: Provide a short string. If none mentioned, use 'General Assistance/Informational'.\n\n"
            "Snippet: $text\n\n"
            "Return ONLY the JSON object. Example: {\"district\": \"Colombo\", \"flood_level_meters\": 1.2, \"victim_count\": 0, \"main_need\": \"Food\", \"status\": \"Warning\"}"
        ),
        temperature=0.0
    )
}

def render(prompt_id: str, **vars) -> Tuple[str, PromptSpec]:
    """
    Renders a prompt template with provided variables.
    Returns a tuple of (rendered_text, prompt_spec).
    """
    if prompt_id not in PROMPTS:
        raise KeyError(f"Prompt '{prompt_id}' not found in registry.")
    
    spec = PROMPTS[prompt_id]
    # safe_substitute prevents crashes if a variable is missing
    text = Template(spec.template).safe_substitute(**vars)
    return text, spec

def list_prompts() -> List[str]:
    """Returns all registered prompt IDs."""
    return list(PROMPTS.keys())