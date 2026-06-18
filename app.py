import streamlit as st
import json
import time
import pandas as pd
from io import BytesIO
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. لینگ گراف اسٹیٹ کی تعریف
class ResearchState(TypedDict):
    items_list: List[str]
    current_index: int
    search_method: str
    all_data: List[dict]

# پیج کی بنیادی سیٹنگز
st.set_page_config(page_title="AI Lead & Brand Researcher", page_icon="🚀", layout="wide")

st.title("🚀 AI Lead & Brand Researcher")
st.write("کمپنیوں یا برانڈز کی لسٹ اپ لوڈ کریں اور ان کا بزنس ڈیٹا (CEO، ای میل، ویب سائٹ) خود بخود نکالیں۔")

# --- سائیڈ بار: API Keys اور سیٹنگز ---
st.sidebar.header("🔑 API Keys & Settings")

groq_key = st.sidebar.text_input("Groq API Key درج کریں:", type="password")
search_option = st.sidebar.selectbox("سرچ کا طریقہ منتخب کریں:", ["DuckDuckGo (100% Free)", "Tavily Search API"])

tavily_key = ""
if search_option == "Tavily Search API":
    tavily_key = st.sidebar.text_input("Tavily API Key درج کریں:", type="password")

# ماڈل سلیکشن
model_choice = st.sidebar.selectbox("Llama ماڈل منتخب کریں:", ["llama-3.3-70b-specdec", "llama3-8b-8192"])

# --- مین ان پٹ ایریا ---
st.subheader("📝 اپنے برانڈز یا کمپنیوں کے نام لکھیں")
user_input = st.text_area("ہر لائن پر ایک نام لکھیں (مثال کے طور پر Khaadi، Systems Limited):", height=150)

# ناموں کو لسٹ میں تبدیل کرنا
input_list = [name.strip() for name in user_input.split("\n") if name.strip()]

# رن بٹن
if st.button("تحقیق شروع کریں (Start Research)"):
    # ویلیڈیشنز (Checks)
    if not groq_key:
        st.error("❌ برائے مہربانی پہلے سائیڈ بار میں Groq API Key درج کریں۔")
    elif search_option == "Tavily Search API" and not tavily_key:
        st.error("❌ آپ نے Tavily منتخب کیا ہے، برائے مہربانی Tavily API Key درج کریں۔")
    elif not input_list:
        st.error("❌ برائے مہربانی کم از کم ایک کمپنی یا برانڈ کا نام لکھیں۔")
    else:
        # انوائرمنٹ ویری ایبلز سیٹ کریں
        import os
        os.environ["GROQ_API_KEY"] = groq_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        try:
            # ٹولز اور LLM لوڈ کریں
            llm = ChatGroq(model=model_choice, temperature=0.1)
            
            if search_option == "Tavily Search API":
                search_tool = TavilySearchResults(max_results=3)
            else:
                search_tool = DuckDuckGoSearchResults(max_results=3)

            # --- لینگ گراف نوڈز فنکشنز ---
            def research_node(state: ResearchState):
                idx = state["current_index"]
                current_item = state["items_list"][idx]
                
                # پروگریس اپ ڈیٹ کریں
                status_text.text(f"🔄 [{idx + 1}/{len(state['items_list'])}] پر تحقیق جاری ہے: {current_item}")
                progress_bar.progress((idx + 1) / len(state['items_list']))
                
                # سرچ کوئری
                query = f"{current_item} brand founder CEO owner parent company business email contact info"
                
                try:
                    search_results = search_tool.invoke({"query": query})
                except Exception as e:
                    search_results = f"Search failed due to rate limits or error: {str(e)}"
                
                # Llama 3.3 پرامپٹ
                prompt = f"""
                Analyze the following search data for '{current_item}' and extract:
                1. Founder or current CEO.
                2. Category (e.g., Clothing, Electronics, IT, Software).
                3. Parent Company (or Independent).
                4. Business Email or contact info.
                5. Official Website.

                Data: {search_results}
                
                Respond ONLY with a valid JSON object matching this structure. Do not include markdown formatting or backticks.
                {{
                    "Name": "{current_item}",
                    "Founder_or_CEO": "Name or Not Found",
                    "Category": "Category or Not Found",
                    "Parent_Company": "Company Name or Independent",
                    "Business_Email": "Email or Not Found",
                    "Website": "URL or Not Found"
                }}
                """
                
                try:
                    response = llm.invoke(prompt)
                    clean_content = response.content.strip()
                    # اگر ماڈل نے غلطی سے مارک ڈاؤن کوڈ بلاک ایڈ کر دیا ہو تو اسے صاف کریں
                    if clean_content.startswith("```"):
                        clean_content = clean_content.replace("
```json", "").replace("```", "").strip()
                    item_data = json.loads(clean_content)
                except Exception as e:
                    item_data = {
                        "Name": current_item,
                        "Founder_or_CEO": "Not Found",
                        "Category": "Not Found",
                        "Parent_Company": "Not Found",
                        "Business_Email": "Not Found",
                        "Website": "Not Found"
                    }

                updated_data = state["all_data"] + [item_data]
                time.sleep(2)  # ریٹ لیمیٹ سے بچنے کے لیے سمال ڈیلے
                
                return {
                    "all_data": updated_data,
                    "current_index": idx + 1
                }

            def router_node(state: ResearchState):
                if state["current_index"] < len(state["items_list"]):
                    return "continue"
                return "end"

            # --- گراف بلڈ کرنا ---
            workflow = StateGraph(ResearchState)
            workflow.add_node("research", research_node)
            workflow.set_entry_point("research")
            
            workflow.add_conditional_edges(
                "research",
                router_node,
                {
                    "continue": "research",
                    "end": END
                }
            )
            
            app = workflow.compile()

            # اسکرین پر پروگریس دکھانے کے ایلیمنٹس
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # گراف رن کریں
            initial_inputs = {
                "items_list": input_list,
                "current_index": 0,
                "search_method": search_option,
                "all_data": []
            }
            
            final_output = app.invoke(initial_inputs)
            
            status_text.success("✅ تمام ڈیٹا کامیابی سے پروسیس ہو گیا ہے!")
            
            # ڈیٹا کو پنڈاس ڈیٹا فریم میں لائیں
            df = pd.DataFrame(final_output["all_data"])
            
            # اسکرین پر ڈیٹا ٹیبل دکھائیں
            st.subheader("📊 حاصل کردہ نتائج (Results)")
            st.dataframe(df, use_container_width=True)
            
            # --- ایکسل ڈاؤن لوڈ بٹن (Excel Download Button) ---
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
            st.error(f"💥 سسٹم رن کرنے میں کوئی مسئلہ آیا ہے: {str(global_err)}")
