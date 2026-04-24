import os
import json
import requests
from pathlib import Path
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

def download_traces():
    """Download the three representative LangFuse traces and save them to traces/."""
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=project_root / ".env")
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com").strip('"').strip("'")
    
    if not public_key or not secret_key:
        print("❌ Error: Langfuse keys not found in .env")
        return

    print("Fetching traces from Langfuse API...")
    
    # Hit the Langfuse Public API
    url = f"{host}/api/public/traces"
    response = requests.get(
        url, 
        auth=HTTPBasicAuth(public_key, secret_key),
        params={"limit": 50}  # Fetch a larger batch to locate all required trace types
    )
    
    if response.status_code == 200:
        traces_dir = project_root / "traces"
        traces_dir.mkdir(exist_ok=True)
        
        traces = response.json().get("data", [])
        
        if not traces:
            print("⚠️ No traces found. Have you asked the chatbot any questions yet?")
            return
            
        found_simple = False
        found_complex = False
        found_failed = False
        
        for trace in traces:
            if found_simple and found_complex and found_failed:
                break
                
            # Safely get the final output dictionary from the orchestrator
            output = trace.get("output", {})
            if not isinstance(output, dict):
                continue
            
            # 1. Look for the FAILED traces (Blocked by Validator / Fallback Triggered)
            # We know it failed if the orchestrator returned type: "error"
            if not found_failed and output.get("type") == "error":
                file_path = traces_dir / "failed_trace.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(trace, f, indent=2)
                print(f"✅ Saved: {file_path.relative_to(project_root)} (Detected Error Type)")
                found_failed = True
                continue
                
            # 2. Look for the SQL Traces
            if output.get("type") == "data" and "sql" in output:
                sql_query = output.get("sql", "").upper()
                
                # COMPLEX Trace: The SQL contains JOINs or aggregations
                if not found_complex and ("JOIN" in sql_query or "GROUP BY" in sql_query):
                    file_path = traces_dir / "complex_trace.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(trace, f, indent=2)
                    print(f"✅ Saved: {file_path.relative_to(project_root)} (Detected JOIN/GROUP BY)")
                    found_complex = True
                    
                # SIMPLE Trace: The SQL exists, but has no complex logic
                elif not found_simple and sql_query and "JOIN" not in sql_query and "GROUP BY" not in sql_query:
                    file_path = traces_dir / "simple_trace.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(trace, f, indent=2)
                    print(f"✅ Saved: {file_path.relative_to(project_root)} (Detected Basic SELECT)")
                    found_simple = True
        
        if not (found_simple and found_complex and found_failed):
            print("\n⚠️ Note: Could not find all 3 specific types in your recent history.")
            print("Ask these exact 3 questions in the app, then run this script again:")
            print("  1. 'How many patients are there?'")
            print("  2. 'What is the revenue per department?'")
            print("  3. 'Drop the patients table.'")
        else:
            print("\n🎉 Success! All 3 required traces have been saved.")
            
    else:
        print(f"❌ Failed to fetch traces: {response.text}")

if __name__ == "__main__":
    download_traces()