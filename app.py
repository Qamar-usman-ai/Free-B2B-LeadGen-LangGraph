import streamlit as st
import json
import time
import pandas as pd
from io import BytesIO
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
# 🔴 فکس کے لیے اہم لائبریریز امپورٹ کی گئی ہیں
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

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
            # ماڈل لوڈ کریں
            llm = ChatGroq(model=model_choice, temperature=0.0)
            
            # 🔴 آفیشل جے سن پارسر کا سیٹ اپ
            parser = JsonOutputParser()

            def research_node(state: ResearchState):
                idx = state["current_index"]
                current_item = state["items_list"][idx]
                
                status_text.text(f"🔍 [{idx + 1}/{len(state['items_list'])}] پر تحقیق جاری ہے: {current_item}")
                progress_bar.progress((idx + 1) / len(state['items_list']))
                
                query = f"{current_item} company brand founder CEO contact email website"
                search_results = ""

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

                # 🔴 پرامپٹ ٹیمپلیٹ کے ساتھ فارمیٹ انسٹرکشنز کا اضافہ
                template = """You are an expert data scraper. Analyze the text data provided below and extract the business details for the entity '{current_item}'.
                
                Text Data: {search_results}
                
                {format_instructions}
                
                Make sure you extract actual data from the text. If any value is completely missing, write "Not Found". Do not make up fake data.
                """
                
                # پارسر خود بخود ماڈل کو بتائے گا کہ جے سن کا سٹرکچر کیا ہونا چاہیے
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["current_item", "search_results"],
                    partial_variables={"format_instructions": parser.get_format_instructions()},
                )
                
                # چین (Chain) بنانا
                chain = prompt | llm | parser

                try:
                    # اب ماڈل ڈائریکٹ درست پائتھون ڈکشنری واپس کرے گا، پارسنگ کی ضرورت نہیں پڑے گی
                    item_data = chain.invoke({"current_item": current_item, "search_results": search_results})
                    
                    # یہ یقینی بنانا کہ کالمز کے نام وہی رہیں جو ہمیں چاہئیں
                    final_mapped_data = {
                        "Name": item_data.get("Name", current_item) or item_data.get("Name", current_item),
                        "Founder_or_CEO": item_data.get("Founder_or_CEO", "Not Found") or item_data.get("founder_or_ceo", "Not Found"),
                        "Category": item_data.get("Category", "Not Found") or item_data.get("category", "Not Found"),
                        "Parent_Company": item_data.get("Parent_Company", "Independent") or item_data.get("parent_company", "Independent"),
                        "Business_Email": item_data.get("Business_Email", "Not Found") or item_data.get("business_email", "Not Found"),
                        "Website": item_data.get("Website", "Not Found") or item_data.get("website", "Not Found")
                    }
                except Exception as e:
                    # اگر ماڈل پھر بھی فیل ہو تو کچے سرچ ڈیٹا کو محفوظ رکھیں
                    final_mapped_data = {
                        "Name": current_item,
                        "Founder_or_CEO": "LLM Refused to answer",
                        "Category": "Data Error",
                        "Parent_Company": "Independent",
                        "Business_Email": "Not Found",
                        "Website": "Not Found"
                    }

                updated_data = state["all_data"] + [final_mapped_data]
                time.sleep(2) 
                
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
