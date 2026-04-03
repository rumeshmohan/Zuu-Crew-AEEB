import os, re, json, time, random, requests, yaml
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

def load_config(path: str = "../src/config/config.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_prompts(path: str = "../src/config/prompts.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def format_prompt(template_name, prompts_dict, **kwargs):
    return prompts_dict[template_name]['template'].format(**kwargs)

def clean_json_output(text: str) -> str:
    text = re.sub(r"```json\s*|```", "", text)
    match = re.search(r'([\[\{].*[\]\}])', text, re.DOTALL)
    return match.group(0).strip() if match else text.strip()

def query_broker(config: Dict[str, Any], prompt: str, role: str = "llm_b", format_json: bool = False):
    provider_name = config['roles'][role]
    prov_cfg = config['providers'][provider_name]
    
    params = config['generation']
    timeout = params.get('request_timeout', 60)
    max_retries = params.get('max_retries', 3)

    model_id = prov_cfg.get(f"{role}_model") or prov_cfg.get("model") if provider_name == "openrouter" else prov_cfg.get("model")

    for attempt in range(max_retries):
        try:
            headers = {"X-Title": "Operation Ledger-Mind"}
            api_key = os.getenv(f"{provider_name.upper()}_API_KEY") or os.getenv("HF_ACCESS_TOKEN")
            if api_key: headers["Authorization"] = f"Bearer {api_key}"

            if provider_name == "google":
                url = f"{prov_cfg['base_url']}?key={os.getenv('GOOGLE_API_KEY')}"
                payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": params['temperature']}}
                response = requests.post(url, json=payload, timeout=timeout)
            
            elif provider_name == "huggingface":
                url = f"{prov_cfg['base_url']}{model_id}"
                payload = {"inputs": prompt, "parameters": {"temperature": params['temperature']}}
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            
            else:
                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": params['temperature']
                }
                if format_json and provider_name == "ollama": payload["format"] = "json"
                response = requests.post(prov_cfg['base_url'], json=payload, headers=headers, timeout=timeout)

            if response.status_code == 429:
                time.sleep((2 ** attempt) + random.random())
                continue
                
            response.raise_for_status()
            res = response.json()

            if provider_name == "google": return res['candidates'][0]['content']['parts'][0]['text']
            if provider_name == "groq": return res["choices"][0]["message"]["content"]
            if provider_name == "ollama": return res["message"]["content"]
            if provider_name == "huggingface": return res[0]['generated_text'] if isinstance(res, list) else res.get('generated_text')
            
            return res["choices"][0]["message"]["content"]
            
        except Exception as e:
            if attempt == max_retries - 1: return None
            time.sleep(2)
            
    return None