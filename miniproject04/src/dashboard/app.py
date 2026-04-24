import sys
import time
import datetime
import re
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from src.engine.orchestrator import NL2SQLPipeline

st.set_page_config(page_title="Nawaloka Analytics", page_icon="🏥", layout="wide")
st.title("🏥 Nawaloka Hospital AI Database Assistant")

@st.cache_resource
def load_pipeline():
    """Load and cache the NL2SQL pipeline using Streamlit's cache resource."""
    return NL2SQLPipeline()

pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! I am the Nawaloka Hospital Data Assistant. Select a pre-built panel from the sidebar or ask a question below!", "display": "Welcome! I am the Nawaloka Hospital Data Assistant. Select a pre-built panel from the sidebar or ask a question below!"}]
    
if "prompt_trigger" not in st.session_state:
    st.session_state.prompt_trigger = None

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "💬 AI Chatbot"

def sync_view_mode():
    """Sync the view mode session state from the radio button widget."""
    st.session_state.view_mode = st.session_state._radio_view

def trigger_quick_prompt(prompt_text):
    """Set a quick prompt trigger and switch to the AI Chatbot view."""
    st.session_state.prompt_trigger = prompt_text
    st.session_state.view_mode = "💬 AI Chatbot"

def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Chat cleared! How can I help you next?", "display": "Chat cleared! How can I help you next?"}]
    st.session_state.prompt_trigger = None
    st.session_state.view_mode = "💬 AI Chatbot"

with st.sidebar:
    st.header("⚙️ Navigation")
    options = ["💬 AI Chatbot", "📊 Operations Dashboard"]
    st.radio("Select View Mode:", options, index=options.index(st.session_state.view_mode), key="_radio_view", on_change=sync_view_mode)
    st.divider()
    
    default_start = datetime.date(2020, 1, 1)
    default_end = datetime.date.today()
    
    if st.session_state.view_mode == "📊 Operations Dashboard":
        st.header("📊 Dashboard Filters")
        active_dash_dept = st.selectbox("Department Filter:", ["All Departments", "Cardiology", "Neurology", "Orthopedics", "Pediatrics", "Oncology"], key="dash_dept")
        
        use_dash_date = st.toggle("Filter by Date Range", value=False, key="dash_date_tog")
        if use_dash_date:
            dash_dates = st.date_input("Select Dates:", value=(default_start, default_end), key="dash_dates")
            dash_start = dash_dates[0] if len(dash_dates) > 0 else default_start
            dash_end = dash_dates[1] if len(dash_dates) > 1 else dash_start
        else:
            dash_start = datetime.date(1900, 1, 1)
            dash_end = datetime.date(2100, 1, 1)
            
        use_dash_val = st.toggle("Filter by Minimum Value", value=False, key="dash_val_tog")
        if use_dash_val:
            active_dash_min_val = st.slider("Min Value ($):", min_value=0, max_value=10000, value=0, step=100, key="dash_min_val")
        else:
            active_dash_min_val = 0
            
        active_chat_dept, use_chat_date, use_chat_val, active_chat_min_val = "All Departments", False, False, 0 
    else:
        st.header("💬 Chatbot Filters")
        active_chat_dept = st.selectbox("Focus AI Query on Department:", ["All Departments", "Cardiology", "Neurology", "Orthopedics", "Pediatrics", "Oncology"], key="chat_dept")
        
        use_chat_date = st.toggle("Focus AI on Date Range", value=False, key="chat_date_tog")
        if use_chat_date:
            chat_dates = st.date_input("Select Dates:", value=(default_start, default_end), key="chat_dates")
            chat_start = chat_dates[0] if len(chat_dates) > 0 else default_start
            chat_end = chat_dates[1] if len(chat_dates) > 1 else chat_start
        else:
            chat_start, chat_end = None, None
            
        use_chat_val = st.toggle("Focus AI on Minimum Value", value=False, key="chat_val_tog")
        if use_chat_val:
            active_chat_min_val = st.slider("Min Value ($):", min_value=0, max_value=10000, value=0, step=100, key="chat_min_val")
        else:
            active_chat_min_val = 0
            
        active_dash_dept, use_dash_date, use_dash_val, active_dash_min_val, dash_start, dash_end = "All Departments", False, False, 0, default_start, default_end

    st.divider()
    
    st.header("💬 Quick Chat Prompts")
    st.button("💰 Revenue Trend", on_click=trigger_quick_prompt, args=("What is the total billed amount per department?",), width="stretch")
    st.button("🩺 Top Diagnoses", on_click=trigger_quick_prompt, args=("What are the top 5 most common diagnoses?",), width="stretch")
    st.button("👨‍⚕️ Doctor Load", on_click=trigger_quick_prompt, args=("Show me a bar chart of the top 5 doctors with the most appointments.",), width="stretch")
    st.button("💳 Payment Methods", on_click=trigger_quick_prompt, args=("Show me the total payment amount grouped by payment method.",), width="stretch")
    st.button("📅 Monthly Admissions", on_click=trigger_quick_prompt, args=("Show me a line chart of the number of admissions grouped by month.",), width="stretch")
        
    st.divider()
    st.button("🗑️ Clear Chat", type="primary", on_click=clear_chat, width="stretch")

def render_visuals(raw_data: list, chart_config: dict):
    """Render the appropriate Plotly chart based on the chart config type."""
    if not raw_data: return
    df = pd.DataFrame(raw_data)
    if df.empty: return

    chart_type = chart_config.get("type", "none")
    if chart_type == "metric":
        st.metric(label="Result", value=df.iloc[0, 0])
        return

    if chart_type in ["bar", "line", "pie"]:
        x_col = chart_config.get("x_axis")
        y_col = chart_config.get("y_axis")
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            st.warning("Chart axes were not clearly defined. Showing raw data instead.")
            return

        x_name = x_col.replace('_', ' ').title()
        y_name = y_col.replace('_', ' ').title()
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

        try:
            if chart_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, color=x_col, title=f"{y_name} by {x_name}", labels={x_col: x_name, y_col: y_name})
            elif chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, markers=True, title=f"{y_name} Trend", labels={x_col: x_name, y_col: y_name})
            elif chart_type == "pie":
                fig = px.pie(df, names=x_col, values=y_col, title=f"{y_name} Distribution")
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"Could not render chart: {e}")

def render_all_time_dashboard(dept, start_d, end_d, min_amt, use_date, use_val):
    """Render the pre-built 5-panel hospital operations dashboard with filters."""
    st.subheader(f"📊 Hospital Operations Overview")
    
    date_label = f" | Dates: {start_d} to {end_d}" if use_date else ""
    val_label = f" | Min Value: ${min_amt}" if use_val else ""
    st.caption(f"**Filters Active:** Department: {dept}{date_label}{val_label}")
    
    dept_cond = f"AND d.department_name = '{dept}'" if dept != "All Departments" else ""
    
    if dept == "All Departments":
        queries = {
            "total_patients": f"SELECT COUNT(*) as count FROM patients WHERE registered_date >= '{start_d}' AND registered_date <= '{end_d}';",
            "total_revenue": f"SELECT COALESCE(SUM(amount), 0) as revenue FROM payments WHERE payment_date >= '{start_d}' AND payment_date <= '{end_d}' AND amount >= {min_amt};",
            "total_doctors": "SELECT COUNT(*) as count FROM doctors;",
            "top_diagnoses": f"SELECT diagnosis_description, COUNT(*) as count FROM diagnoses WHERE diagnosis_date >= '{start_d}' AND diagnosis_date <= '{end_d}' GROUP BY diagnosis_description ORDER BY count DESC LIMIT 5;",
            "payment_methods": f"SELECT payment_method, COALESCE(SUM(amount), 0) as total FROM payments WHERE payment_date >= '{start_d}' AND payment_date <= '{end_d}' AND amount >= {min_amt} GROUP BY payment_method;",
            "monthly_admissions": f"SELECT TO_CHAR(admission_date, 'YYYY-MM') as month, COUNT(*) as count FROM admissions WHERE admission_date >= '{start_d}' AND admission_date <= '{end_d}' GROUP BY month ORDER BY month;",
            "dept_revenue": f"SELECT d.department_name, COALESCE(SUM(b.total_amount), 0) as total FROM departments d JOIN admissions a ON d.department_id = a.department_id JOIN billing_invoices b ON a.admission_id = b.admission_id WHERE b.invoice_date >= '{start_d}' AND b.invoice_date <= '{end_d}' AND b.total_amount >= {min_amt} GROUP BY d.department_name;",
            "doctor_load": f"SELECT doc.full_name, COUNT(a.appointment_id) as appointments FROM doctors doc JOIN appointments a ON doc.doctor_id = a.doctor_id WHERE a.appointment_date >= '{start_d}' AND a.appointment_date <= '{end_d}' GROUP BY doc.full_name ORDER BY appointments DESC LIMIT 5;",
        }
    else:
        queries = {
            "total_patients": f"SELECT COUNT(DISTINCT p.patient_id) as count FROM patients p JOIN admissions a ON p.patient_id = a.patient_id JOIN departments d ON a.department_id = d.department_id WHERE p.registered_date >= '{start_d}' AND p.registered_date <= '{end_d}' {dept_cond};",
            "total_revenue": f"SELECT COALESCE(SUM(p.amount), 0) as revenue FROM payments p JOIN billing_invoices b ON p.invoice_id = b.invoice_id JOIN admissions a ON b.admission_id = a.admission_id JOIN departments d ON a.department_id = d.department_id WHERE p.payment_date >= '{start_d}' AND p.payment_date <= '{end_d}' AND p.amount >= {min_amt} {dept_cond};",
            "total_doctors": f"SELECT COUNT(DISTINCT doc.doctor_id) as count FROM doctors doc LEFT JOIN departments d ON doc.department_id = d.department_id WHERE 1=1 {dept_cond};",
            "top_diagnoses": f"SELECT diag.diagnosis_description, COUNT(*) as count FROM diagnoses diag JOIN admissions a ON diag.admission_id = a.admission_id JOIN departments d ON a.department_id = d.department_id WHERE diag.diagnosis_date >= '{start_d}' AND diag.diagnosis_date <= '{end_d}' {dept_cond} GROUP BY diag.diagnosis_description ORDER BY count DESC LIMIT 5;",
            "payment_methods": f"SELECT p.payment_method, COALESCE(SUM(p.amount), 0) as total FROM payments p JOIN billing_invoices b ON p.invoice_id = b.invoice_id JOIN admissions a ON b.admission_id = a.admission_id JOIN departments d ON a.department_id = d.department_id WHERE p.payment_date >= '{start_d}' AND p.payment_date <= '{end_d}' AND p.amount >= {min_amt} {dept_cond} GROUP BY p.payment_method;",
            "monthly_admissions": f"SELECT TO_CHAR(a.admission_date, 'YYYY-MM') as month, COUNT(*) as count FROM admissions a JOIN departments d ON a.department_id = d.department_id WHERE a.admission_date >= '{start_d}' AND a.admission_date <= '{end_d}' {dept_cond} GROUP BY month ORDER BY month;",
            "dept_revenue": f"SELECT d.department_name, COALESCE(SUM(b.total_amount), 0) as total FROM departments d JOIN admissions a ON d.department_id = a.department_id JOIN billing_invoices b ON a.admission_id = b.admission_id WHERE b.invoice_date >= '{start_d}' AND b.invoice_date <= '{end_d}' AND b.total_amount >= {min_amt} {dept_cond} GROUP BY d.department_name;",
            "doctor_load": f"SELECT doc.full_name, COUNT(a.appointment_id) as appointments FROM doctors doc JOIN appointments a ON doc.doctor_id = a.doctor_id JOIN departments d ON doc.department_id = d.department_id WHERE a.appointment_date >= '{start_d}' AND a.appointment_date <= '{end_d}' {dept_cond} GROUP BY doc.full_name ORDER BY appointments DESC LIMIT 5;",
        }
    
    m1, m2, m3 = st.columns(3)
    with m1:
        res = pipeline.db_client.execute_query(queries["total_patients"])
        st.metric(label="Registered Patients", value=res[0].get('count', 0) if res and "error" not in res[0] else "Err")
    with m2:
        res = pipeline.db_client.execute_query(queries["total_revenue"])
        rev = res[0].get('revenue', 0) if res and "error" not in res[0] else "Err"
        st.metric(label="Total Revenue", value=f"${rev:,.2f}" if isinstance(rev, (int, float)) else rev)
    with m3:
        res = pipeline.db_client.execute_query(queries["total_doctors"])
        st.metric(label="Total Doctors", value=res[0].get('count', 0) if res and "error" not in res[0] else "Err")
        
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        # Panel 1: Monthly Admissions Trend
        data = pipeline.db_client.execute_query(queries["monthly_admissions"])
        if data and "error" not in data[0] and pd.DataFrame(data).shape[0] > 0:
            st.plotly_chart(px.line(pd.DataFrame(data), x="month", y="count", title="Monthly Admissions Trend", markers=True, labels={"month": "Month", "count": "Admissions"}), width="stretch")
            
        # Panel 2: Top 5 Diagnoses
        data2 = pipeline.db_client.execute_query(queries["top_diagnoses"])
        if data2 and "error" not in data2[0] and pd.DataFrame(data2).shape[0] > 0:
            st.plotly_chart(px.bar(pd.DataFrame(data2), x="diagnosis_description", y="count", color="diagnosis_description", title="Top 5 Diagnoses", labels={"diagnosis_description": "Diagnosis", "count": "Count"}), width="stretch")
            
    with c2:
        # Panel 3: Payment Methods
        data3 = pipeline.db_client.execute_query(queries["payment_methods"])
        if data3 and "error" not in data3[0] and pd.DataFrame(data3).shape[0] > 0:
            st.plotly_chart(px.pie(pd.DataFrame(data3), names="payment_method", values="total", title="Revenue by Payment Method"), width="stretch")
            
        # Panel 4: Revenue by Department
        data4 = pipeline.db_client.execute_query(queries["dept_revenue"])
        if data4 and "error" not in data4[0] and pd.DataFrame(data4).shape[0] > 0:
            st.plotly_chart(px.bar(pd.DataFrame(data4), x="department_name", y="total", title="Revenue by Department", color="department_name", labels={"department_name": "Department", "total": "Total Revenue ($)"}), width="stretch")

    st.divider()

    # Panel 5: Doctor Load (top 5 doctors by appointment count)
    data5 = pipeline.db_client.execute_query(queries["doctor_load"])
    if data5 and "error" not in data5[0] and pd.DataFrame(data5).shape[0] > 0:
        st.plotly_chart(
            px.bar(
                pd.DataFrame(data5),
                x="full_name",
                y="appointments",
                color="full_name",
                title="Top 5 Doctors by Appointment Load",
                labels={"full_name": "Doctor", "appointments": "Total Appointments"},
            ),
            width="stretch",
        )

if st.session_state.view_mode == "📊 Operations Dashboard":
    render_all_time_dashboard(active_dash_dept, dash_start, dash_end, active_dash_min_val, use_dash_date, use_dash_val)
else:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            display_text = msg.get("display", msg.get("content", ""))
            clean_text = display_text.replace("`", "") if isinstance(display_text, str) else display_text
            
            if msg.get("is_alert"):
                st.error(clean_text)
            elif msg.get("is_warning"):
                st.warning(clean_text)
            else:
                st.markdown(clean_text)
            
            if "data" in msg and msg["data"]:
                render_visuals(msg["data"], msg.get("chart_config", {}))
                
                with st.expander("📜 View Generated SQL"):
                    st.code(msg.get("sql", "N/A"), language="sql")
                
                with st.expander("🗄️ View Raw Data"):
                    df = pd.DataFrame(msg["data"])
                    st.dataframe(df)
                    
                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                    with dl_col1:
                        st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode('utf-8'), file_name='export.csv', mime='text/csv', key=f"dl_csv_{i}")
                    with dl_col2:
                        st.download_button("⬇️ Download JSON", data=json.dumps(msg["data"], indent=2, default=str), file_name='export.json', mime='application/json', key=f"dl_json_{i}")
                    with dl_col3:
                        st.download_button("⬇️ Download TXT", data=df.to_string(index=False).encode('utf-8'), file_name='export.txt', mime='text/plain', key=f"dl_txt_{i}")
                        
                if "latency" in msg:
                    with st.expander("⏱️ Execution Traces"):
                        st.markdown("##### 🚀 Performance Metrics")
                        t1, t2, t3 = st.columns(3)
                        with t1:
                            st.metric("Pipeline Latency", f"{msg.get('latency', 0)}s")
                        with t2:
                            st.metric("Est. Tokens (approx)", msg.get('tokens', 0))
                        with t3:
                            st.metric("Agents Invoked", msg.get('agents_invoked', 4))
                        st.caption("Detailed step-by-step LLM tracing and observability are continuously logged to the LangFuse dashboard.")

    user_input = st.chat_input("Ask a question about hospital operations...")
    if user_input:
        st.session_state.prompt_trigger = user_input

    if st.session_state.prompt_trigger:
        base_prompt = st.session_state.prompt_trigger
        st.session_state.prompt_trigger = None 
        
        forbidden_pattern = re.compile(r'\b(drop|delete|truncate|update|insert|alter|remove)\b', re.IGNORECASE)
        
        if forbidden_pattern.search(base_prompt):
            st.session_state.messages.append({"role": "user", "content": base_prompt, "display": base_prompt})
            alert_msg = "🚨 **Security Alert:** Destructive database commands (like DROP, DELETE, or UPDATE) are strictly prohibited.\n\n*Did you mean to ask to **view**, **list**, or **count** the records instead?*"
            st.session_state.messages.append({"role": "assistant", "content": alert_msg, "display": alert_msg, "is_alert": True})
            st.rerun()
            
        elif len(base_prompt.strip()) < 5:
            st.session_state.messages.append({"role": "user", "content": base_prompt, "display": base_prompt})
            typo_msg = "🤔 That looks like a typo or mistake. \n\n*Did you mean to ask about **Revenue**, **Patients**, or **Doctor Appointments**? Please try asking a full question!*"
            st.session_state.messages.append({"role": "assistant", "content": typo_msg, "display": typo_msg, "is_warning": True})
            st.rerun()
            
        else:
            active_filters = []
            if active_chat_dept != "All Departments": active_filters.append(f"Department = '{active_chat_dept}'")
            if use_chat_date: active_filters.append(f"Date Range = {chat_start} to {chat_end}")
            if use_chat_val: active_filters.append(f"Minimum Monetary Value = ${active_chat_min_val}")
            
            if active_filters:
                filter_str = ", ".join(active_filters)
                actual_prompt = f"{base_prompt}\n\n[System Context: Please restrict this SQL query explicitly by applying these filters: {filter_str}]"
            else:
                actual_prompt = base_prompt
                
            st.session_state.messages.append({"role": "user", "content": actual_prompt, "display": base_prompt})
            with st.chat_message("user"):
                st.markdown(base_prompt)

            with st.chat_message("assistant"):
                with st.status("Analyzing database...", expanded=True) as status:
                    st.write("🕵️‍♂️ Routing intent...")
                    st.write("⚙️ Generating SQL query...")
                    st.write("🔍 Executing on PostgreSQL...")
                    
                    start_time = time.time()
                    response = pipeline.process_query(actual_prompt, st.session_state.messages[:-1])
                    latency = round(time.time() - start_time, 2)
                    
                    st.write("📊 Interpreting results...")
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                is_pipeline_error = response.get("type") == "error"
                if is_pipeline_error:
                    clean_live_content = f"⚠️ {response['content']}\n\n*Did you mean to ask something like: 'Show me the revenue for last month' or 'How many patients are there?'*"
                    st.warning(clean_live_content)
                else:
                    clean_live_content = response["content"].replace("`", "") if isinstance(response["content"], str) else response["content"]
                    st.markdown(clean_live_content)
                
                est_tokens = int((len(actual_prompt) + len(response.get("sql", "")) + len(clean_live_content)) / 3.5)
                
                msg_data = {
                    "role": "assistant", 
                    "content": clean_live_content, 
                    "display": clean_live_content,
                    "data": response.get("raw_data"), 
                    "sql": response.get("sql"), 
                    "chart_config": response.get("chart_config"),
                    "latency": latency,
                    "tokens": est_tokens,
                    "agents_invoked": response.get("agents_invoked", 4),
                    "is_warning": is_pipeline_error
                }
                
                if response.get("type") == "data" and response.get("raw_data"):
                    render_visuals(response["raw_data"], response.get("chart_config", {}))
                    
                    with st.expander("📜 View Generated SQL"):
                        st.code(response.get("sql", "N/A"), language="sql")
                    
                    with st.expander("🗄️ View Raw Data"):
                        df = pd.DataFrame(response["raw_data"])
                        st.dataframe(df)
                        
                        dl_col1, dl_col2, dl_col3 = st.columns(3)
                        with dl_col1:
                            st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode('utf-8'), file_name='export.csv', mime='text/csv', key="dl_csv_live")
                        with dl_col2:
                            st.download_button("⬇️ Download JSON", data=json.dumps(response["raw_data"], indent=2, default=str), file_name='export.json', mime='application/json', key="dl_json_live")
                        with dl_col3:
                            st.download_button("⬇️ Download TXT", data=df.to_string(index=False).encode('utf-8'), file_name='export.txt', mime='text/plain', key="dl_txt_live")
                            
                    with st.expander("⏱️ Execution Traces"):
                        st.markdown("##### 🚀 Performance Metrics")
                        t1, t2, t3 = st.columns(3)
                        with t1:
                            st.metric("Pipeline Latency", f"{latency}s")
                        with t2:
                            st.metric("Est. Tokens (approx)", est_tokens)
                        with t3:
                            st.metric("Agents Invoked", response.get("agents_invoked", 4))
                        st.caption("Detailed step-by-step LLM tracing and observability are continuously logged to the LangFuse dashboard.")
                
                st.session_state.messages.append(msg_data)
                st.rerun()