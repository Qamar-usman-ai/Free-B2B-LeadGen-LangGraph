import streamlit as st
import json
import time
import pandas as pd
from io import BytesIO
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# DuckDuckGo امپورٹ کی سیکیورٹی
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

# 1. لینگ گراف اسٹیٹ
class ResearchState(TypedDict):
    items_list: List[str]
    current_index: int
    search_method: str
    all_data: List[dict]

st.set_page_config(page_title="AI Lead & Brand Researcher", page_icon="🚀", layout="wide")

st.title("🚀 AI Lead & Brand Researcher")
st.write("کمپنیوں یا برانڈز کی لسٹ داخل کریں اور ان کا بزنس ڈیٹا خود بخود حاصل کریں۔")

# --- سائیڈ بار ---
st.sidebar.header("🔑 API Keys & Settings")
groq_key = st.sidebar.text_input("Groq API Key درج کریں:", type="password")
search_option = st.sidebar.selectbox("سرچ کا طریقہ منتخب کریں:", ["DuckDuckGo (100% Free)", "Tavily Search API"])

tavily_key = ""
if search_option == "Tavily Search API":
    tavily_key = st.sidebar.text_input("Tavily API Key درج کریں:", type="password")

model_choice = st.sidebar.selectbox("Llama ماڈل منتخب کریں:", ["llama-3.3-70b-specdec", "llama3-8b-8192"])

# --- مین ان پٹ ---
st.subheader("📝 اپنے برانڈز یا کمپنیوں کے نام لکھیں")
user_input = st.text_area("ہر لائن پر ایک نام لکھیں:", height=150)
input_list = [name.strip() for name in user_input.split("\n") if name.strip()]

if st.button("تحقیق شروع کریں (Start Research)"):
    if not groq_key:
        st.error("❌ برائے مہربانی پہلے سائیڈ بار میں Groq API Key درج کریں۔")
    elif search_option == "Tavily Search API" and not tavily_key:
        st.error("❌ آپ نے Tavily منتخب کیا ہے، برائے مہربانی Tavily API Key درج کریں۔")
    elif search_option == "DuckDuckGo (100% Free)" and not DDG_AVAILABLE:
        st.error("❌ سسٹم میں 'duckduckgo-search' انسٹال نہیں ہے۔")
    elif not input_list:
        st.error("❌ برائے مہربانی کم از کم ایک نام ضرور لکھیں۔")
    else:
        import os
        os.environ["GROQ_API_KEY"] = groq_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        try:
            llm = ChatGroq(model=model_choice, temperature=0.0) # Temperature 0.0 تاکہ ماڈل لکیر کا فقیر رہے اور JSON خراب نہ کرے

            def research_node(state: ResearchState):
                idx = state["current_index"]
                current_item = state["items_list"][idx]
                
                status_text.text(f"🔍 [{idx + 1}/{len(state['items_list'])}] پر تحقیق جاری ہے: {current_item}")
                progress_bar.progress((idx + 1) / len(state['items_list']))
                
                query = f"{current_item} company brand founder CEO contact email website"
                search_results = ""

                # سرچ انجن رن کرنا
                try:
                    if state["search_method"] == "Tavily Search API":
                        from langchain_community.tools.tavily_search import TavilySearchResults
                        search_tool = TavilySearchResults(max_results=3)
                        res = search_tool.invoke({"query": query})
                        search_results = str(res)
                    else:
                        with DDGS() as ddgs:
                            results = [r for r in ddgs.text(query, max_results=3)]
                            search_results = str(results)
                except Exception as e:
                    search_results = f"Search engine failed: {str(e)}"

                # پرامپٹ کو انتہائی سخت (Strict) کر دیا گیا ہے
                prompt = f"""
                You are a data extraction assistant. Analyze the text data provided and extract the details for the entity '{current_item}'.
                
                Text Data: {search_results}
                
                You must return ONLY a JSON object. No conversational text, no explanations, no markdown formatting, no backticks.
                
                Desired JSON Structure:
                {{
                    "Name": "{current_item}",
                    "Founder_or_CEO": "Extract CEO name or write Not Found",
                    "Category": "Extract business category or write Not Found",
                    "Parent_Company": "Extract parent company or write Independent",
                    "Business_Email": "Extract email address or write Not Found",
                    "Website": "Extract official URL or write Not Found"
                }}
                """
                
                try:
                    response = llm.invoke(prompt)
                    clean_content = response.content.strip()
                    
                    # 🛠️ ایڈوانسڈ جے سن کلیننگ (JSON Cleaning): اگر ماڈل پھر بھی markdown بلاکس بنا دے
                    if "```json" in clean_content:
                        clean_content = clean_content.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_content:
                        clean_content = clean_content.split("```")[1].strip()
                        
                    item_data = json.loads(clean_content)
                except Exception as parse_error:
                    # اگر جے سن پارس نہ ہو سکے تو سرچ کے کچے ڈیٹا میں سے کچھ نہ کچھ نکالنے کی کوشش کریں
                    item_data = {
                        "Name": current_item,
                        "Founder_or_CEO": "Parsing Error",
                        "Category": "Check API Keys",
                        "Parent_Company": "Error",
                        "Business_Email": "Not Found",
                        "Website": "Failed to Parse Model Output"
                    }

                updated_data = state["all_data"] + [item_data]
                time.sleep(2) # ریٹ لمٹ سیفٹی
                
                return {
                    "all_data": updated_data,
                    "current_index": idx + 1
                }

            def router_node(state: ResearchState):
                if state["current_index"] < len(state["items_list"]):
                    return "continue"
                return "end"

            # گراف بلڈ کرنا
            workflow = StateGraph(ResearchState)
            workflow.add_node("research", research_node)
            workflow.set_entry_point("research")
            workflow.add_conditional_edges("research", router_node, {"continue": "research", "end": END})
            app = workflow.compile()

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            initial_inputs = {
                "items_list": input_list,
                "current_index": 0,
                "search_method": search_option,
                "all_data": []
            }
            
            final_output = app.invoke(initial_inputs)
            status_text.success("✅ تحقیق مکمل ہو گئی!")
            
            df = pd.DataFrame(final_output["all_data"])
            st.subheader("📊 حاصل کردہ نتائج (Results)")
            st.dataframe(df, use_container_width=True)
            
            # Excel ڈاؤن لوڈ
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Leads_Data')
            processed_data = output.getvalue()
            
            st.download_button(
                label="📥 ایکسل فارمیٹ میں ڈیٹا ڈاؤن لوڈ کریں (.xlsx)",
                data=processed_data,
                file_name="ai_generated_leads.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as global_err:
            st.error(f"💥 خرابی آئی: {str(global_err)}")
