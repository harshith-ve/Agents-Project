import pickle
import os
import bm25s
from typing import Dict, List, TypedDict, Optional
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents import create_tool_calling_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF (pip install pymupdf)
import re
import torch 
from langchain_core.runnables.config import RunnableConfig
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import SystemMessage
config = RunnableConfig(recursion_limit=50)

# Define API key for Gemini models
os.environ["GEMINI_API_KEY"] = "AIzaSyDIc_WwIPtqT55NZqnxOOBWsdE4hD-Symw"

# Define the state structure
class State(TypedDict):
    query: str
    args: Optional[str]
    template: Optional[str]
    keywords: Optional[List[str]]
    papers: Optional[List[str]]
    similar_papers: Optional[List[str]]
    ranked_papers: Optional[List[str]]
    human_feedback: Optional[str]
    summaries: Optional[List[dict]]
    data_summary: Optional[str]
    review: Optional[str]
    next_step: Optional[str]
    feedback: Optional[str]
    previous_node: Optional[str]
    previous_responses: Optional[str]
    previous_tool_calls: Optional[str]

# ---------------------------
# Data Loader Function
# ---------------------------
file_path1 = 'dataset/AP_TABS.pkl'
index_path = 'dataset/bm25_index.pkl'
EMBEDDING_DIM = 768
model = SentenceTransformer("all-mpnet-base-v2")
index = faiss.read_index("/home/harshit.harshit/Faiss_AP_FT/faiss.index")
paper_ids = np.load("/home/harshit.harshit/Faiss_AP_FT/paper_ids.npy")
index.nprobe = 50  # Adjust based on your index characteristics

# ---------------------------
# Define Tools 
# ---------------------------

def suggest_keywords_function(query: str, template: str, existing_keywords: List[str] = None) -> List[str]:
    """
    Suggests keywords based on the query and template.
    
    Args:
        query: The user's query for literature review
        template: The template with questions about the topic
        existing_keywords: Optional list of existing keywords
        
    Returns:
        List of suggested keywords
    """
    print(f"\n[FUNCTION] Executing suggest_keywords()")
    if existing_keywords is None:
        existing_keywords = []
    
    prompt = f"""
Based on the following query and template, suggest 5-10 relevant keywords to locate academic papers.

Query: {query}
Template: {template}
Return your answer as a comma-separated list.
Just provide the keywords, no additional text.
Example: <keyword1>, <keyword2>, <keyword3>
Here keyword1, keyword2, keyword3 are place holders where actual keywords should be present.
No fullstop at the end.
"""
    if existing_keywords:
        prompt += f"""List of available key words: {','.join(existing_keywords)}"""
    
    print(f"\n[LLM CALL] suggest_keywords() - Calling Gemini model for keyword suggestions")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
    response = llm.invoke(prompt)
    suggested_keywords = response.content.strip()
    print(f"[LLM RESULT] suggest_keywords() - Got keywords: {suggested_keywords}")
    
    new_keywords = [kw.strip() for kw in suggested_keywords.split(",") if kw.strip()]
    combined_keywords = existing_keywords.copy() if existing_keywords else []
    
    for key in new_keywords:
        if key not in combined_keywords:
            combined_keywords.append(key)
    
    print("\n=== Suggested Keywords ===")
    print(combined_keywords)
    print("==========================\n")
    
    return combined_keywords

@tool
def search_papers(keywords: List[str]) -> List[str]:
    """
    Searches for papers using the provided keywords and generates summaries.

    Args:
        keywords: List of keywords to use for searching papers.

    Returns:
        List of relevant papers with summaries.
    """
    if not keywords:
        print("No keywords provided for search.")
        return []

    # 1. FAISS Search for Relevant Papers
    query_text = ' '.join(keywords)
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    _, I = index.search(query_embedding, k=15)
    matched_paper_ids = [paper_ids[i] for i in I[0]]

    papers_with_summaries = []
    for paper_id in matched_paper_ids:
        full_text = extract_text_from_arxiv(paper_id)
        if not full_text:
            continue

        prompt = f"""Generate a 500-word summary of this paper focusing on:
- Core research problem
- Methodology used
- Key findings
- Significance of results

Paper text: {full_text[:15000]}"""  # Removed extra indentation

        # Get proper response content
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
        response = llm.invoke(prompt)
        summary = response.content.strip()[:3000]  # Changed .text to .content

        papers_with_summaries.append(f"{paper_id} {summary}")

    print(f"\n=== Found {len(papers_with_summaries)} Papers ===")
    if papers_with_summaries:
        print("Sample:", papers_with_summaries[0][:150] + "...")

    return papers_with_summaries

@tool
def find_similar_papers(paper_name: str) -> List[str]:
    """
    Finds papers similar to the provided paper.
    
    Args:
        paper_name: The name of the paper to find similar papers to
        
    Returns:
        List of papers similar to the provided paper
    """

    query_text = paper_name
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    _, I = index.search(query_embedding, k=7)
    matched_paper_ids = [paper_ids[i] for i in I[0]]

    papers_with_summaries = []
    for paper_id in matched_paper_ids:
        full_text = extract_text_from_arxiv(paper_id)
        if not full_text:
            continue

        prompt = f"""Generate a 500-word summary of this paper focusing on:
- Core research problem
- Methodology used
- Key findings
- Significance of results

Paper text: {full_text[:15000]}"""  # Removed extra indentation

        # Get proper response content
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
        response = llm.invoke(prompt)
        summary = response.content.strip()[:3000]  # Changed .text to .content

        papers_with_summaries.append(f"{paper_id} {summary}")

    print(f"\n=== Found {len(papers_with_summaries)} Papers ===")
    if papers_with_summaries:
        print("Sample:", papers_with_summaries[0][:150] + "...")

    return papers_with_summaries

@tool
def rank_papers(papers: List[str], query: str) -> List[str]:
    """
    Ranks papers based on their relevance to the query.
    
    Args:
        papers: List of papers to rank
        query: The query to rank papers against
        
    Returns:
        List of papers ranked by relevance
    """
    print(f"\n[FUNCTION] Executing rank_papers() with {len(papers)} papers")
    
    if not papers:
        return []
    
    papers_to_rank = papers[:12]
    paper_details = []
    for i, paper in enumerate(papers_to_rank):
        paper_details.append(f"{i+1}. {paper}")
    
    prompt = f"""
Query: {query}

Rank the following papers from most to least relevant:
{chr(10).join(paper_details)}

Return your answer as a comma-separated list of indexes of the paper.
Example : <index1>, <index2>, <index3>
Here index1, index2, index3 are place holders where the actual index numbers should be present.
Start indices with 0.
No additional text should be generated.
"""
    print(f"\n[LLM CALL] rank_papers() - Calling Gemini model for paper ranking")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
    response = llm.invoke(prompt)
    ranking_text = response.content.strip()
    print(f"[LLM RESULT] rank_papers() - Got ranking: {ranking_text}")
    
    try:
        import re
        indices = [int(x) - 1 for x in re.findall(r'\d+', ranking_text)]
        valid_indices = [i for i in indices if 0 <= i < len(papers_to_rank)]

        ranked_papers = [papers_to_rank[i] for i in valid_indices]
        remaining_papers = [p for idx, p in enumerate(papers_to_rank) if idx not in valid_indices]
        
        ranked_papers.extend(remaining_papers)
        print(f"\n=== Ranked Papers ===\nRanked {len(ranked_papers)} papers\n====================\n")
        return ranked_papers
    except Exception as e:
        print("Error parsing ranking:", e)
        return papers_to_rank

def fix_arxiv_id(arxiv_id):
    """
    Adjusts an arXiv identifier by inserting a slash if it's missing.
    For example, converts "quant-ph0311017" to "quant-ph/0311017".
    """
    if "/" in arxiv_id:
        return arxiv_id
    match = re.match(r"([a-zA-Z-]+)(\d+)", arxiv_id)
    if match:
        prefix, digits = match.groups()
        return f"{prefix}/{digits}"
    # Return the original if pattern matching fails.
    return arxiv_id

def extract_text_from_arxiv(arxiv_id):
    """
    Fetches and extracts text from an arXiv paper without saving the PDF.
    
    The function adjusts the arXiv id format if necessary, downloads the PDF,
    and extracts text from each page.
    """
    corrected_id = fix_arxiv_id(arxiv_id)
    pdf_url = f"https://arxiv.org/pdf/{corrected_id}.pdf"
    response = requests.get(pdf_url, stream=True)
    
    if response.status_code == 200:
        pdf_document = fitz.open(stream=response.content, filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf_document)
        return text
    else:
        print(f"❌ Failed to fetch: {arxiv_id}")
        return None

@tool
def summarize_papers(papers: List[str]) -> List[Dict]:
    """
    Summarizes the provided papers.
    
    Args:
        papers: List of papers to summarize
        
    Returns:
        List of dictionaries containing paper ID and summary
    """
    os.environ["GEMINI_API_KEY"] = "AIzaSyCxr8E4Cad5rLpHqtwqQ0UzCfQAYd13nvs"
    print(f"\n[FUNCTION] Executing summarize_papers() with {len(papers)} papers")
    
    if not papers:
        print("No papers provided for summarization.")
        return []
    
    papers_to_summarize = papers[:15]
    summaries = []
    
    for i, item in enumerate(papers_to_summarize):
        print(f"\n[PROCESSING] Summarizing paper {i+1}/{len(papers_to_summarize)}: {item}")
        title = item.split(" ", 1)[0]
        full_text = extract_text_from_arxiv(title)
        if not full_text:
            print(f"[WARNING] Could not extract text for paper: {title}")
            continue
            
        prompt = f"""Summarize the following research paper with a focus on thoroughly discussing its results:
**Full Text:** 
{full_text}

Your summary should include:
1. **Key Findings**: Highlight the main results presented in the paper.
2. **Implications**: Discuss the significance of these findings and their impact on the field.
3. **Limitations**: Mention any limitations or challenges identified by the authors.
4. **Comparison to Previous Work**: Explain how these results compare to prior research in the same domain.
5. **Future Directions**: Suggest potential avenues for further research based on the paper's conclusions.

Make sure your summary is concise, well-structured, and written in an academic tone suitable for researchers and professionals.
"""
        print(f"\n[LLM CALL] summarize_papers() - Calling Gemini model for paper {title}")
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
        response = llm.invoke(prompt)
        summary_text = response.content.strip()
        print(f"[LLM RESULT] summarize_papers() - Generated summary of {len(summary_text)} characters")
        
        summaries.append({'paper': title, 'summary': summary_text})
    
    print(f"\n=== Paper Summaries ===\nGenerated summaries for {len(summaries)} papers.\n=======================\n")
    return summaries

# ---------------------------
# Node: Template Generator
# ---------------------------
def template_generator(state: State) -> State:
    current_state = dict(state)
    query = state.get('query', '')
    prompt = f"""
Based on the following query, generate a set of questions that need to be answered 
to create a literature review on the topic.:

Query: {query}

Generate 7 to 10 specific and complete questions.
Just generate the questions, no need to answer them.
Format: One question per line.
Only questions are required, no additional text.
"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
    response = llm.invoke(prompt)
    template = response.content.strip()
    
    print("\n=== Template Generated ===")
    print(template)
    print("==========================\n")
    
    return {**current_state, "template": template, "next_step": "data_curator"}

# ---------------------------
# Node: Suggest Keywords
# ---------------------------
def suggest_keywords_node(state: State) -> State:
    """
    Node function to suggest keywords based on query and template.
    """
    current_state = dict(state)
    query = state.get('query', '')
    template = state.get('template', '')
    existing_keywords = state.get('keywords', [])
    
    # Use the existing suggest_keywords_function to get keywords
    keywords = suggest_keywords_function(query, template, existing_keywords)
    
    return {**current_state, "keywords": keywords, "next_step": "data_curator"}

# ---------------------------
# Node: Human Feedback on Keywords
# ---------------------------
def human_feedback_on_keywords(state: State) -> State:
    """
    Node function to get human feedback on suggested keywords.
    """
    current_state = dict(state)
    keywords = state.get('keywords', [])
    
    print("\n=== Human Feedback on Keywords ===")
    print("Please review the suggested keywords:")
    for i, kw in enumerate(keywords, 1):
        print(f"{i}. {kw}")
    
    # Ask if user wants to edit keywords
    edit_response = input("Would you like to edit these keywords? (yes/no): ").strip().lower()
    
    if edit_response in ['y', 'yes']:
        # Show editing options
        print("\nEditing options:")
        print("1. Add new keywords (format: +keyword1,keyword2,...)")
        print("2. Remove keywords (format: -1,3,5 to remove keywords at positions 1, 3, and 5)")
        print("3. Replace all (format: =keyword1,keyword2,...)")
        
        edit_command = input("Enter your edit command: ").strip()
        
        if edit_command.startswith('+'):
            # Add new keywords
            new_keywords = [kw.strip() for kw in edit_command[1:].split(',') if kw.strip()]
            keywords.extend(new_keywords)
            print(f"Added {len(new_keywords)} keywords")
            
        elif edit_command.startswith('-'):
            # Remove keywords by position
            try:
                positions = [int(pos.strip()) for pos in edit_command[1:].split(',') if pos.strip()]
                positions.sort(reverse=True)  # Sort in reverse to remove from end first
                for pos in positions:
                    if 1 <= pos <= len(keywords):
                        removed = keywords.pop(pos-1)
                        print(f"Removed: {removed}")
                    else:
                        print(f"Position {pos} is out of range")
            except ValueError:
                print("Invalid position format. Expected numbers separated by commas.")
                
        elif edit_command.startswith('='):
            # Replace all keywords
            new_keywords = [kw.strip() for kw in edit_command[1:].split(',') if kw.strip()]
            keywords = new_keywords
            print(f"Replaced with {len(keywords)} new keywords")
    else :
        print("No changes made to keywords.")
    
    print("\n=== Final Keywords ===")
    for i, kw in enumerate(keywords, 1):
        print(f"{i}. {kw}")
    print("=====================\n")
    
    return {**current_state, "keywords": keywords, "next_step": "data_curator"}

# ---------------------------
# Node: Data Curator (ReactAgent with tools)
# ---------------------------
def data_curator(state: State) -> State:
    os.environ["GEMINI_API_KEY"] = "AIzaSyAAlAIuwTm7eJ9BpzPnZzsd1KzpFLuSUXA"
    current_state = dict(state)
    # Create a list of tools
    tools = [
        search_papers,
        find_similar_papers,
        rank_papers,
        summarize_papers
    ]

    # Get previous response and tool calls from state or initialize if not present
    previous_responses = state.get('previous_responses', [])
    previous_tool_calls = state.get('previous_tool_calls', [])
    
    # Define the system message for the data curator
    system_message = """
You are the Data Curator in a literature review generation system.

IMPORTANT INSTRUCTIONS:
1. You need to decide which tool to call based on the current state.
2. ALWAYS USE THE TOOL DIRECTLY - DO NOT just describe what tool you would use.
3. Follow this workflow sequence:
   - Search for papers (search_papers) using the keywords
   - When you have papers, rank them (rank_papers) by relevance to the query
   - Finally, summarize the top ranked papers (summarize_papers)
5. YOU MUST MAKE A TOOL CALL OR RESPOND WITH "COMPLETE" IN YOUR RESPONSE.

NEVER call summarize_papers before rank_papers. Always ensure papers are properly ranked first.

Available tools:
1. search_papers: Retrieve papers using keywords
2. rank_papers: Rank papers by relevance to the query
3. summarize_papers: Summarize the ranked papers
4. find_similar_papers: Find papers similar to a specific paper

If you think the current information is sufficient for the literature review, respond with "COMPLETE" with no additional text.
"""

    # Create a ChatPromptTemplate with the system message and a placeholder for user input
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
    
    # Create the message with current state information
    message = f"""Current state:
Query: {state.get('query', '')}
Template: {state.get('template', '')}
Keywords: {state.get('keywords', [])}
Papers: {(state.get('papers', []))}
Ranked Papers: {len(state.get('ranked_papers', []))} papers ranked 
Summaries: {len(state.get('summaries', []))} paper summaries
Feedback from data_summarizer: {state.get('feedback', '')}

Based on this state, which ONE tool should I call next?
"""
    if previous_responses:
        message += f"\nPrevious responses: {previous_responses}"
    if previous_tool_calls:
        message += f"\nPrevious tool calls: {previous_tool_calls}"
    # Create the agent using llm.bind_tools approach
    llm_with_tools = llm.bind_tools(tools)
    
    # Call LLM directly with the system prompt and user message
    response = llm_with_tools.invoke([
        SystemMessage(content=system_message),
        {"role": "human", "content": message}
    ])
    
    print("\n=== LLM RESPONSE ===")
    print(f"Response content: {response.content}")
    print(f"Tool calls detected: {getattr(response, 'tool_calls', [])}")
    print("====================\n")
    
    updated_state = current_state.copy()
    updated_state["previous_responses"] = [response.content]
    updated_state["previous_tool_calls"] = getattr(response, "tool_calls", [])
    # Process tool calls from the response - only expecting one call
    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        # Process only the first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        
        print(f"\n=== Executing tool: {tool_name} ===")
        print(f"Tool arguments: {tool_args}")
        print()
        
        # Execute the tool call
        if tool_name == "search_papers":
            output = search_papers.invoke({"keywords": state.get('keywords', [])})
            updated_state["papers"] = output
        elif tool_name == "find_similar_papers" and "paper_name" in tool_args:
            output = find_similar_papers.invoke({"paper_name": tool_args["paper_name"]})
            existing_similar = current_state.get("similar_papers", [])
            updated_state["similar_papers"] = existing_similar + output
        elif tool_name == "rank_papers":
            output = rank_papers.invoke({
                "papers": state.get('papers', []),
                "query": state.get('query', '')
            })
            updated_state["ranked_papers"] = output
        elif tool_name == "summarize_papers":
            # Check if we have ranked papers to use, otherwise use regular papers
            papers_to_summarize = state.get('ranked_papers', [])
            if not papers_to_summarize and state.get('papers', []):
                # If we have papers but they're not ranked, rank them first
                print("\n=== Papers not ranked yet, ranking papers first ===\n")
                ranked_papers = rank_papers.invoke({
                    "papers": state.get('papers', []),
                    "query": state.get('query', '')
                })
                updated_state["ranked_papers"] = ranked_papers
                papers_to_summarize = ranked_papers
                
            output = summarize_papers.invoke({"papers": papers_to_summarize[:15]})  # Limit to top 15 papers
            existing_summaries = current_state.get("summaries", [])
            updated_state["summaries"] = existing_summaries + output
    elif "COMPLETE" in response.content:
        print("\n=== Data Collection Complete (LLM signaled completion) ===\n")
        updated_state["next_step"] = "data_summarizer"
        return updated_state
    updated_state["feedback"] = ""
    updated_state["next_step"] = "data_curator"
    
    return updated_state

# ---------------------------
# Node: Data Summarizer Node
# ---------------------------
def data_summarizer(state: State) -> State:
    os.environ["GEMINI_API_KEY"] = "AIzaSyBFjWXxeeMLFKtlueKC7fMYJE8W9NeuDlo"
    current_state = dict(state)
    summaries = state.get('summaries', [])
    template = state.get('template',"")
    query = state.get('query', '')
    feedback = state.get('feedback', '')
    
    if not summaries:
        return {**current_state, "feedback": "No relevant paper Summaries found.", "next_step": "data_curator"}
    
    summary_texts = []
    for i, item in enumerate(summaries):
        # Properly access dictionary fields in each summary item
        title = item['paper']
        summ_text = item['summary']
        summary_texts.append(f"Paper {i+1}: {title}\nSummary: {summ_text}")
    
    prompt = f"""
Query = {query}
Template= {template}
Summaries:{chr(10).join(summary_texts)}

{'' if not feedback else f'IMPORTANT FEEDBACK FROM REVIEW EVALUATION: {feedback}Based on this feedback, please improve your summary to address the issues mentioned.'}

For the given query, Template and the Summaries, Answer 3 to 5 questions that can be answerd using the given summaries.
Give answers in the paragraph format. Try to generate long answers.
If you think the given summaries are not sufficient to answer the questions, then return "NO SUFFICIENT INFORMATION" in the answer with no additional text.
"""
    # Using langchain ChatGoogleGenerativeAI instead of direct genai
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
    response = llm.invoke(prompt)
    data_summary = response.content.strip()
    
    if "NO SUFFICIENT INFORMATION" in data_summary:
        return {**current_state, "previous_node": "data_summarizer", "feedback": "Insufficeient information. Try to fetch more relavant papers", "next_step": "data_curator"}
    print("\n=== Data Summary Generated ===")
    print(data_summary)
    print("==============================\n")
    return {**current_state, "data_summary": data_summary, "next_step": "review_generator"}

# ---------------------------
# Node: Review Generator Node
# ---------------------------
def review_generator(state: State) -> State:
    current_state = dict(state)
    data_summary = state.get('data_summary', '')
    query = state.get('query', '')
    if not data_summary:
        return {**current_state, "feedback": "Insufficient data to generate review.", "next_step": END}
    
    prompt = f"""
Generate a detailed and well-structured 'Related Work' section for the query: "{query}".

Data Provided:
{data_summary}
"""
    # Using langchain ChatGoogleGenerativeAI instead of direct genai
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
    response = llm.invoke(prompt)
    review = response.content.strip()
    
    print("\n=== Final Review Generated ===")
    print(review)
    print("=============================\n")
    
    return {**current_state, "review": review, "next_step": "review_evaluator"}

# ---------------------------
# Review Evaluator Router
# ---------------------------
def review_evaluator(state: State) -> str:
    """
    Evaluates the generated review and determines if it needs to be regenerated or if it's ready.
    
    Args:
        state: The current state including the generated review
        
    Returns:
        String indicating where to route next: "data_summarizer" for regeneration or "END" for completion
    """
    review = state.get('review', '')
    query = state.get('query', '')
    
    if not review:
        print("\n=== No Review Generated, Routing to data_summarizer ===\n")
        return "data_summarizer"
    
    prompt = f"""
You are a critical evaluator of literature reviews. Evaluate the following literature review 
on "{query}" and determine if it meets academic standards.

Review to evaluate:
{review}

Evaluate the review on the following criteria:
1. Comprehensiveness: Does it cover the main aspects of the topic?
2. Structure: Is it well-structured with clear sections?
3. Critical analysis: Does it analyze and not just describe the literature?
4. Coherence: Is the review cohesive and logical?
5. Academic style: Is it written in proper academic language?

If the review meets all criteria adequately (doesn't need to be perfect), respond ONLY with "ACCEPTABLE".
If the review has significant issues on any criteria, respond ONLY with "NEEDS IMPROVEMENT".
"""
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
    response = llm.invoke(prompt)
    evaluation = response.content.strip()
    
    print("\n=== Review Evaluation ===")
    print(f"Evaluation result: {evaluation}")
    print("========================\n")
    
    # Determine where to route based on evaluation
    if "ACCEPTABLE" in evaluation:
        return "END"
    else:
        # Add feedback to the state for data_summarizer to consider
        state["feedback"] = f"Review needs improvement: {evaluation}"
        return "data_summarizer"

# ---------------------------
# Graph Construction & Edges
# ---------------------------
# Create the leaderboard generation benchmark using langgraph
workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("template_generator", template_generator)
workflow.add_node("suggest_keywords", suggest_keywords_node)
workflow.add_node("human_feedback_on_keywords", human_feedback_on_keywords)
workflow.add_node("data_curator", data_curator)
workflow.add_node("data_summarizer", data_summarizer)
workflow.add_node("review_generator", review_generator)

# Set the entry point to the graph
workflow.set_entry_point("template_generator")

# Define the conditional routing from data_curator based on the next_step field
workflow.add_conditional_edges(
    "data_curator",
    lambda state: state["next_step"],
    {
        "data_curator": "data_curator",
        "data_summarizer": "data_summarizer",
        END: END
    }
)

# Add conditional edge from data_summarizer based on next_step
workflow.add_conditional_edges(
    "data_summarizer",
    lambda state: state["next_step"],
    {
        "data_curator": "data_curator",
        "review_generator": "review_generator"
    }
)

# Add edge from review_generator to END or data_summarizer based on evaluation
workflow.add_conditional_edges(
    "review_generator",
    review_evaluator,  # Using review_evaluator as a router function
    {
        "data_summarizer": "data_summarizer",
        "END": END
    }
)

# Add edge from template_generator to suggest_keywords
workflow.add_edge("template_generator", "suggest_keywords")

# Add edge from suggest_keywords to human_feedback_on_keywords
workflow.add_edge("suggest_keywords", "human_feedback_on_keywords")

# Add edge from human_feedback_on_keywords to data_curator
workflow.add_edge("human_feedback_on_keywords", "data_curator")

# Compile the graph to make it executable
app = workflow.compile()

# Example usage
def run_leaderboard_benchmark(query):
    """Run the leaderboard generation benchmark with the given query."""
    result = app.invoke({"query": query}, config)
    return result

# Example benchmark execution
if __name__ == "__main__":
    benchmark_result = run_leaderboard_benchmark("Generate a literature review on question answering datasets on research papers")
