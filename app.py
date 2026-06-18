import streamlit as st
import time
import pandas as pd
from io import BytesIO
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# ڈک ڈک گو سرچ ہینڈلر
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

# 🔴 1. پیڈانٹک اسکیما (Pydantic Schema for Guaranteed Structured Data)
# یہ ماڈل کو مجبور کرتا ہے کہ وہ ڈیٹا لازمی نکالے
class CompanyDetails(BaseModel):
    Founder_or_CEO: str = Field(description="The name of the Founder, CEO, or Managing Director. Write 'Not Found' if missing.")
    Category: str = Field(description="Business domain or industry category like Clothing, Electronics, IT, FMCG. Write 'Not Found' if missing.")
    Parent_Company: str = Field(description="Name of the parent organization. If independent, write 'Independent'.")
    Business_Email: str = Field(description="The corporate or official business/support email address. Write 'Not Found' if missing.")
    Website: str = Field(description="The official website URL. Write 'Not Found' if missing.")

# لینگ گراف اسٹیٹ
class ResearchState(TypedDict):
    items_list: List[str]
    current_index: int
    search_method: str
    all_data: List[dict]

st.set_page_config(page_title="AI Lead & Brand Researcher Pro", page_icon="🚀", layout="wide")

st.title("🚀 AI Lead & Brand Researcher Pro")
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
            # 🔴 ایڈوانسڈ ماڈل بائنڈنگ (Structured Output Integration)
            base_llm = ChatGroq(model=model_choice, temperature=0.0)
            structured_llm = base_llm.with_structured_output(CompanyDetails)

            def research_node(state: ResearchState):
                idx = state["current_index"]
                current_item = state["items_list"][idx]
                
                status_text.text(f"🔍 [{idx + 1}/{len(state['items_list'])}] پر انٹرنیٹ ریسرچ جاری ہے: {current_item}")
                progress_bar.progress((idx + 1) / len(state['items_list']))
                
                # سرچ کوئری کو زیادہ مخصوص کر دیا گیا ہے تاکہ فالتو کچرا ٹیکسٹ نہ آئے
                query = f"{current_item} official website founder CEO email contact"
                search_results = ""

                # سرچ کرنا
                try:
                    if state["search_method"] == "Tavily Search API":
                        from langchain_community.tools.tavily_search import TavilySearchResults
                        search_tool = TavilySearchResults(max_results=2)
                        res = search_tool.invoke({"query": query})
                        search_results = str(res)
                    else:
                        with DDGS() as ddgs:
                            results = [r for r in ddgs.text(query, max_results=2)]
                            # صرف اہم ٹیکسٹ رکھنے کے لیے باڈی اسٹرکشن
                            search_results = " ".join([f"{r.get('title','')}: {r.get('body','')}" for r in results])
                except Exception as e:
                    search_results = f"Search token/limit issue: {str(e)}"

                # پرامپٹ کو بالکل کلین کر دیا گیا ہے
                prompt = f"""You are a data intelligence agent. Extract accurate facts about '{current_item}' from the following search web text.
                
                Search Web Text:
                {search_results}
                """
                
                try:
                    # 🔴 اب ماڈل فنکشن کالنگ کے ذریعے ڈائریکٹ ہماری کلاس کے سٹرکچر میں ڈیٹا بھرے گا
                    extracted_obj = structured_llm.invoke(prompt)
                    
                    item_data = {
                        "Name": current_item,
                        "Founder_or_CEO": extracted_obj.Founder_or_CEO,
                        "Category": extracted_obj.Category,
                        "Parent_Company": extracted_obj.Parent_Company,
                        "Business_Email": extracted_obj.Business_Email,
                        "Website": extracted_obj.Website
                    }
                except Exception as llm_error:
                    # اگر ریٹ لیمیٹ کا پکا بلاک آ جائے تو صارف کو وارننگ دیں
                    item_data = {
                        "Name": current_item,
                        "Founder_or_CEO": "Rate Limit / Blocked",
                        "Category": "Groq Key Overloaded",
                        "Parent_Company": "Independent",
                        "Business_Email": "Not Found",
                        "Website": "Not Found"
                    }

                updated_data = state["all_data"] + [item_data]
                
                # ⏱️ اہم سیکیورٹی وقفہ: تاکہ Groq فری ٹیر بلاک نہ کرے
                time.sleep(3) 
                
                return {
                    "all_data": updated_data,
                    "current_index": idx + 1
                }

            def router_node(state: ResearchState):
                if state["current_index"] < len(state["items_list"]):
                    return "continue"
                return "end"

            # گراف بنانا
            workflow = StateGraph(ResearchState)
            workflow.add_node("research", research_node)
            workflow.set_entry_point("research")
            workflow.add_conditional_edges("research", router_node, {"continue": "research", "end": END})
            app = workflow.compile()

            progress_bar = st.progress(0)
            st_error_box = st.empty()
            status_text = st.empty()
            
            initial_inputs = {
                "items_list": input_list,
                "current_index": 0,
                "search_method": search_option,
                "all_data": []
            }
            
            final_output = app.invoke(initial_inputs)
            status_text.success("✅ فائنل رپورٹ تیار ہے!")
            
            df = pd.DataFrame(final_output["all_data"])
            st.subheader("📊 حاصل کردہ نتائج (Results)")
            st.dataframe(df, use_container_width=True)
            
            # Excel شیٹ بنانا
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
            st.error(f"💥 چلانے کے دوران خرابی آئی: {str(global_err)}")
