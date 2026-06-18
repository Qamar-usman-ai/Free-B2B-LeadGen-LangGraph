import streamlit as st
import json
import time
import pandas as pd
from io import BytesIO
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# بالکل محفوظ طریقے سے DuckDuckGo امپورٹ کرنے کا طریقہ
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

# 1. لینگ گراف اسٹیٹ (Graph State)
class ResearchState(TypedDict):
    items_list: List[str]
    current_index: int
    search_method: str
    all_data: List[dict]

# اسٹریم لٹ پیج سیٹنگز
st.set_page_config(page_title="AI Lead & Brand Researcher", page_icon="🚀", layout="wide")

st.title("🚀 AI Lead & Brand Researcher")
st.write("کمپنیوں یا برانڈز کی لسٹ داخل کریں اور ان کا بزنس ڈیٹا (CEO، ای میل، ویب سائٹ) خود بخود حاصل کریں۔")

# --- سائیڈ بار سیٹنگز ---
st.sidebar.header("🔑 API Keys & Settings")

groq_key = st.sidebar.text_input("Groq API Key درج کریں:", type="password")
search_option = st.sidebar.selectbox("سرچ کا طریقہ منتخب کریں:", ["DuckDuckGo (100% Free)", "Tavily Search API"])

tavily_key = ""
if search_option == "Tavily Search API":
    tavily_key = st.sidebar.text_input("Tavily API Key درج کریں:", type="password")

model_choice = st.sidebar.selectbox("Llama ماڈل منتخب کریں:", ["llama-3.3-70b-specdec", "llama3-8b-8192"])

# --- مین ان پٹ ایریا ---
st.subheader("📝 اپنے برانڈز یا کمپنیوں کے نام لکھیں")
user_input = st.text_area("ہر لائن پر ایک نام لکھیں (جیسے Khaadi، Systems Limited وغیرہ):", height=150)

# ان پٹ کو کلین لسٹ میں تبدیل کریں
input_list = [name.strip() for name in user_input.split("\n") if name.strip()]

# رن بٹن
if st.button("تحقیق شروع کریں (Start Research)"):
    # بنیادی چیکس (Validation)
    if not groq_key:
        st.error("❌ برائے مہربانی پہلے سائیڈ بار میں Groq API Key درج کریں۔")
    elif search_option == "Tavily Search API" and not tavily_key:
        st.error("❌ آپ نے Tavily منتخب کیا ہے، برائے مہربانی Tavily API Key درج کریں۔")
    elif search_option == "DuckDuckGo (100% Free)" and not DDG_AVAILABLE:
        st.error("❌ سسٹم میں 'duckduckgo-search' پیکیج انسٹال نہیں ہے۔ برائے مہربانی ٹرمینل میں 'pip install duckduckgo-search' چلائیں۔")
    elif not input_list:
        st.error("❌ برائے مہربانی ٹیکسٹ ایریا میں کم از کم ایک کمپنی یا برانڈ کا نام ضرور لکھیں۔")
    else:
        # انوائرمنٹ سیٹ اپ
        import os
        os.environ["GROQ_API_KEY"] = groq_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        try:
            # LLM لوڈ کریں
            llm = ChatGroq(model=model_choice, temperature=0.1)

            # --- لینگ گراف نوڈ فنکشن ---
            def research_node(state: ResearchState):
                idx = state["current_index"]
                current_item = state["items_list"][idx]
                
                # اسکرین پر پروگریس دکھائیں
                status_text.text(f"🔍 [{idx + 1}/{len(state['items_list'])}] پر تحقیق جاری ہے: {current_item}")
                progress_bar.progress((idx + 1) / len(state['items_list']))
                
                query = f"{current_item} brand founder CEO owner parent company business email contact info"
                search_results = ""

                # سرچ انجن لاجک
                try:
                    if state["search_method"] == "Tavily Search API":
                        from langchain_community.tools.tavily_search import TavilySearchResults
                        search_tool = TavilySearchResults(max_results=3)
                        res = search_tool.invoke({"query": query})
                        search_results = str(res)
                    else:
                        # خالص اور ایرر فری ڈک ڈک گو سرچ
                        with DDGS() as ddgs:
                            results = [r for r in ddgs.text(query, max_results=3)]
                            search_results = str(results)
                except Exception as e:
                    search_results = f"Search failed or rate limited: {str(e)}"
                
                # کچی معلومات کو صاف کرنے کے لیے پرامپٹ
                prompt = f"""
                Analyze the following text and extract details for '{current_item}':
                1. Founder or current CEO.
                2. Category (e.g., Clothing, Electronics, IT).
                3. Parent Company (or Independent).
                4. Business Email or contact info.
                5. Official Website.

                Text Data: {search_results}
                
                Respond ONLY with a valid JSON object matching this structure. Do not include markdown code blocks or backticks.
                {{
                    "Name": "{current_item}",
                    "Founder_or_CEO": "Name or Not Found",
                    "Category": "Category or Not Found",
                    "Parent_Company": "Company Name or Independent",
                    "Business_Email": "Email or Not Found",
                    "Website": "URL or Not Found"
                }}
                """
                
                # AI سے جواب حاصل کریں اور جے سن (JSON) پارس کریں
                try:
                    response = llm.invoke(prompt)
                    clean_content = response.content.strip()
                    
                    # اگر ماڈل غلطی سے مارک ڈاؤن کوڈ بلاکس لگا دے تو صفائی کریں
                    if clean_content.startswith("```"):
                        clean_content = clean_content.replace("```json", "").replace("```", "").strip()
                        
                    item_data = json.loads(clean_content)
                except Exception:
                    # اگر کوئی بھی مسئلہ آئے تو خالی کالمز بنائیں تاکہ ایپ رکے نہیں
                    item_data = {
                        "Name": current_item,
                        "Founder_or_CEO": "Not Found",
                        "Category": "Not Found",
                        "Parent_Company": "Not Found",
                        "Business_Email": "Not Found",
                        "Website": "Not Found"
                    }

                updated_data = state["all_data"] + [item_data]
                time.sleep(2)  # ریٹ لیمیٹ سے بچنے کے لیے سیکیورٹی ڈیلے
                
                return {
                    "all_data": updated_data,
                    "current_index": idx + 1
                }

            # راؤٹر فنکشن (کنڈیشنل ایج)
            def router_node(state: ResearchState):
                if state["current_index"] < len(state["items_list"]):
                    return "continue"
                return "end"

            # --- گراف بنانا ---
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

            # پروگریس بار ہینڈلرز
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # ان پٹ ڈیٹا گراف میں بھیجیں
            initial_inputs = {
                "items_list": input_list,
                "current_index": 0,
                "search_method": search_option,
                "all_data": []
            }
            
            # گراف رن کریں
            final_output = app.invoke(initial_inputs)
            
            status_text.success("✅ تمام برانڈز/کمپنیوں کی ریسرچ مکمل ہو گئی ہے!")
            
            # ڈیٹا کو ایکسل اور اسکرین پر لوڈ کرنا
            df = pd.DataFrame(final_output["all_data"])
            st.subheader("📊 حاصل کردہ نتائج (Results)")
            st.dataframe(df, use_container_width=True)
            
            # ایکسل جنریشن (.xlsx)
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
            st.error(f"💥 سسٹم چلانے کے دوران خرابی آئی: {str(global_err)}")
