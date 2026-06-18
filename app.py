import streamlit as st
import time
import pandas as pd
from io import BytesIO
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# Safe import for DuckDuckGo Search Engine
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

# 1. Define the Pydantic Schema for Guaranteed Structured Output
class CompanyDetails(BaseModel):
    Founder_or_CEO: str = Field(description="The full name of the Founder, current CEO, or Managing Director. Return 'Not Found' if missing.")
    Category: str = Field(description="The business domain or industry sector (e.g., Clothing, Electronics, IT Services, FMCG). Return 'Not Found' if missing.")
    Parent_Company: str = Field(description="Name of the parent organization. If independent or standalone, strictly return 'Independent'.")
    Business_Email: str = Field(description="The official corporate contact, support, or sales email address. Return 'Not Found' if missing.")
    Website: str = Field(description="The valid official website URL. Return 'Not Found' if missing.")

# 2. Define the LangGraph State
class ResearchState(TypedDict):
    items_list: List[str]
    current_index: int
    search_method: str
    all_data: List[dict]

# Application Configuration
st.set_page_config(page_title="AI Lead & Brand Researcher Pro", page_icon="🚀", layout="wide")

st.title("🚀 AI Lead & Brand Researcher Pro")
st.write("Automate business intelligence gathering. Input a list of brands/companies to extract key decision-makers, verified emails, and corporate details using Google Gemini and LangGraph.")

# --- Sidebar Configuration ---
st.sidebar.header("🔑 Credentials & Settings")
gemini_key = st.sidebar.text_input("Google Gemini API Key:", type="password", help="Get a free API key from Google AI Studio")
search_option = st.sidebar.selectbox("Select Search Engine:", ["DuckDuckGo (100% Free)", "Tavily Search API"])

tavily_key = ""
if search_option == "Tavily Search API":
    tavily_key = st.sidebar.text_input("Tavily API Key:", type="password")

# Documentation Helper Link
st.sidebar.markdown("---")
if st.sidebar.button("💡 How to get a free Gemini API Key?"):
    st.sidebar.info(
        "1. Go to [Google AI Studio](https://aistudio.google.com/)\n"
        "2. Sign in with your Google account.\n"
        "3. Click on **'Get API Key'**.\n"
        "4. Generate and copy your key into the sidebar field."
    )

# --- Main Input Interface ---
st.subheader("📝 Target Entities")
user_input = st.text_area("Enter company or brand names (One entity per line):", height=150, placeholder="Example:\nKhaadi\nSystems Limited\nSamsung")
input_list = [name.strip() for name in user_input.split("\n") if name.strip()]

# Execution Pipeline
if st.button("Start Research Pipeline"):
    if not gemini_key:
        st.error("❌ Authentication Error: Please provide a valid Google Gemini API Key in the sidebar.")
    elif search_option == "Tavily Search API" and not tavily_key:
        st.error("❌ Authentication Error: Tavily Search is selected but the API key is missing.")
    elif search_option == "DuckDuckGo (100% Free)" and not DDG_AVAILABLE:
        st.error("❌ Dependency Error: 'duckduckgo-search' package is missing in your environment.")
    elif not input_list:
        st.error("❌ Input Error: The target entity list is empty. Please enter at least one brand name.")
    else:
        # Bind environment variables safely
        import os
        os.environ["GOOGLE_API_KEY"] = gemini_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        try:
            # Initialize Gemini Model with Structured Output Binding
            base_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
            structured_llm = base_llm.with_structured_output(CompanyDetails)

            # --- LangGraph Core Workflow Nodes ---
            def research_node(state: ResearchState):
                idx = state["current_index"]
                current_item = state["items_list"][idx]
                
                # Dynamic Dashboard Progress Reporting
                status_text.text(f"🔍 Executing Web Intelligence Search [{idx + 1}/{len(state['items_list'])}]: {current_item}")
                progress_bar.progress((idx + 1) / len(state['items_list']))
                
                query = f"{current_item} corporate headquarters official website founder CEO business email contact address"
                search_results = ""

                # Execute Search Queries Safely
                try:
                    if state["search_method"] == "Tavily Search API":
                        from langchain_community.tools.tavily_search import TavilySearchResults
                        search_tool = TavilySearchResults(max_results=2)
                        res = search_tool.invoke({"query": query})
                        search_results = str(res)
                    else:
                        with DDGS() as ddgs:
                            results = [r for r in ddgs.text(query, max_results=2)]
                            search_results = " ".join([f"{r.get('title','')}: {r.get('body','')}" for r in results])
                except Exception as e:
                    search_results = f"Search Execution Context Failed: {str(e)}"

                # Core Prompts for Data Context Matching
                prompt = f"""You are an elite data mining agent. Analyze the provided raw web context and extract verified business metrics for the entity '{current_item}'.
                
                Raw Web Context:
                {search_results}
                """
                
                try:
                    # Invoke Structured Object Retrieval
                    extracted_obj = structured_llm.invoke(prompt)
                    
                    item_data = {
                        "Name": current_item,
                        "Founder_or_CEO": extracted_obj.Founder_or_CEO,
                        "Category": extracted_obj.Category,
                        "Parent_Company": extracted_obj.Parent_Company,
                        "Business_Email": extracted_obj.Business_Email,
                        "Website": extracted_obj.Website
                    }
                except Exception:
                    # Robust Fallback Matrix to prevent pipeline halting
                    item_data = {
                        "Name": current_item,
                        "Founder_or_CEO": "Not Found",
                        "Category": "Extraction Error",
                        "Parent_Company": "Independent",
                        "Business_Email": "Not Found",
                        "Website": "Not Found"
                    }

                updated_data = state["all_data"] + [item_data]
                
                # Rate Limiting Guardrail to safely respect Gemini Free Tier (15 RPM)
                time.sleep(4) 
                
                return {
                    "all_data": updated_data,
                    "current_index": idx + 1
                }

            def router_node(state: ResearchState):
                if state["current_index"] < len(state["items_list"]):
                    return "continue"
                return "end"

            # --- State Graph Compilation ---
            workflow = StateGraph(ResearchState)
            workflow.add_node("research", research_node)
            workflow.set_entry_point("research")
            workflow.add_conditional_edges("research", router_node, {"continue": "research", "end": END})
            app = workflow.compile()

            # Dynamic Progress Displays
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            initial_inputs = {
                "items_list": input_list,
                "current_index": 0,
                "search_method": search_option,
                "all_data": []
            }
            
            # Execute Agent Pipeline
            final_output = app.invoke(initial_inputs)
            status_text.success("✅ Business Intelligence Pipeline Executed Successfully!")
            
            # Data Frame Processing
            df = pd.DataFrame(final_output["all_data"])
            st.subheader("📊 Structured Research Metrics")
            st.dataframe(df, use_container_width=True)
            
            # Excel Generation Stream
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Leads_Data')
            processed_data = output.getvalue()
            
            st.download_button(
                label="📥 Export Dataset to Excel (.xlsx)",
                data=processed_data,
                file_name="ai_generated_leads_pro.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as global_err:
            st.error(f"💥 Critical Pipeline System Failure: {str(global_err)}")
