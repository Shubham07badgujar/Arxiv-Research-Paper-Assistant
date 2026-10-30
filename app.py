import os
# Keep transformers on PyTorch; an installed TensorFlow with Keras 3 breaks its import
os.environ.setdefault("USE_TF", "0")

import streamlit as st
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
from langchain_core.documents import Document
import pandas as pd
from langchain_community.vectorstores import FAISS
import re
import fitz  # PyMuPDF for PDF text extraction
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
ATOM = "{http://www.w3.org/2005/Atom}"

# Neo4j database connection function
def connect_to_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

def store_papers_in_neo4j(driver, papers):
    with driver.session() as session:
        for paper in papers:
            session.run(
                """
                MERGE (p:Paper {link: $link})
                SET p.title = $title, p.summary = $summary, p.year = $year
                """,
                title=paper["title"],
                summary=paper["summary"],
                link=paper["link"],
                year=paper["year"]
            )

def clean_text(text):
    # ArXiv Atom fields contain hard line breaks and indentation
    return " ".join(text.split())

def fetch_arxiv_papers(topic, start_year):
    end_year = datetime.now().year
    params = {
        "search_query": f'all:"{topic}" AND submittedDate:[{start_year}01010000 TO {end_year}12312359]',
        "start": 0,
        "max_results": 50,
    }
    try:
        response = requests.get("https://export.arxiv.org/api/query", params=params, timeout=30)
    except requests.RequestException as e:
        return f"Error fetching data from ArXiv API: {e}"

    if response.status_code != 200:
        return f"Error fetching data from ArXiv API. Status Code: {response.status_code}"

    root = ET.fromstring(response.content)
    papers = []
    for entry in root.findall(f'{ATOM}entry'):
        papers.append({
            "title": clean_text(entry.find(f'{ATOM}title').text),
            "summary": clean_text(entry.find(f'{ATOM}summary').text),
            "link": entry.find(f'{ATOM}id').text,
            "year": int(entry.find(f'{ATOM}published').text.split('-')[0])
        })

    return papers

def download_arxiv_paper(link):
    match = re.search(r'arxiv.org/abs/([0-9.]+)', link)
    if not match:
        return None
    paper_id = match.group(1)
    pdf_link = f"https://arxiv.org/pdf/{paper_id}.pdf"

    # Define the local directory to save PDFs
    download_dir = "downloaded_papers"
    os.makedirs(download_dir, exist_ok=True)  # Create the directory if it doesn't exist
    pdf_path = os.path.join(download_dir, f"{paper_id}.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    try:
        response = requests.get(pdf_link, timeout=60)
    except requests.RequestException:
        return None
    if response.status_code == 200:
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        return pdf_path
    else:
        return None

def extract_text_from_pdf(pdf_path):
    # Extract text from the PDF file
    full_text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text()
    return full_text


# Streamlit reruns the whole script on every interaction, so load models once
@st.cache_resource
def load_sbert_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def load_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name=GROQ_MODEL)

def query_relevant_papers(driver, user_query):
    with driver.session() as session:
        result = session.run("MATCH (p:Paper) RETURN p.title AS title, p.summary AS summary, p.link AS link")
        papers = [{"title": r["title"], "summary": r["summary"], "link": r["link"]} for r in result]

    if not papers:
        return None

    sbert_model = load_sbert_model()
    user_query_embedding = sbert_model.encode(user_query, convert_to_tensor=True)
    paper_embeddings = sbert_model.encode([p["summary"] for p in papers], convert_to_tensor=True)
    scores = util.cos_sim(user_query_embedding, paper_embeddings)[0]
    for paper, score in zip(papers, scores):
        paper["similarity"] = score.item()

    return sorted(papers, key=lambda x: x["similarity"], reverse=True)


prompt=ChatPromptTemplate.from_template(
    """
    You are a research paper assistant.
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question:{input}

    """
)

def create_vector_embedding(input_text: str, source: str):
    # Rebuild the index whenever the question is about a different paper
    if st.session_state.get("vectors_source") != source:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents([Document(page_content=input_text)])
        st.session_state.vectors = FAISS.from_documents(docs, load_embeddings())
        st.session_state.vectors_source = source
    return st.session_state.vectors



def generate_future_directions(papers: List[Dict]) -> str:
    """Generate future research directions based on paper summaries."""
    summaries = "\n".join([f"Title: {p['title']}\nSummary: {p['summary']}\n" for p in papers[:5]])

    future_prompt = f"""Based on these recent research papers:
    {summaries}

    Please analyze the current trends and suggest 3-5 promising future research directions.
    For each direction, explain:
    1. The motivation
    2. Potential impact
    3. Technical challenges to overcome"""
    try:
        response = load_llm().invoke(future_prompt)
        return response.content
    except Exception as e:
        print(f"Error generating future directions: {e}")
        return "Error generating future research directions."


def extract_topic(query):
    match = re.search(r'related to (.*?)(?: in the last|$)', query, re.IGNORECASE)
    return (match.group(1) if match else query).strip()


def main():
    st.title("Academic Research Paper Assistant Application")

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not groq_api_key:
        st.error("GROQ_API_KEY is not set. Add it to your .env file.")
        return

    first_query = st.text_input("Enter your first query (e.g., 'show me all the papers related to machine learning in the last 5 year'):")
    if first_query:
        topic = extract_topic(first_query)
        current_year = datetime.now().year
        start_year = current_year - 5

        if st.button("Fetch and Store Papers"):
            with st.spinner(f"Fetching papers on '{topic}'..."):
                papers = fetch_arxiv_papers(topic, start_year)

                if isinstance(papers, str):
                    st.error(papers)
                elif not papers:
                    st.info("No papers found for the given topic and year range.")
                else:
                    st.header("Fetched Papers")

                    # Create a DataFrame from the papers list
                    papers_df = pd.DataFrame(papers)
                    papers_df['year'] = papers_df['year'].astype(int)

                    # Display the DataFrame in a Streamlit dataframe
                    st.dataframe(papers_df[['title', 'year']])

                    with st.spinner("Connecting to Neo4j and storing data..."):
                        driver = connect_to_neo4j(neo4j_uri, neo4j_user, neo4j_password)
                        try:
                            store_papers_in_neo4j(driver, papers)
                            st.success("Papers successfully stored in the Neo4j database.")
                        except Exception as e:
                            st.error(f"An error occurred while storing data in Neo4j: {e}")
                        finally:
                            driver.close()
    else:
        st.warning("Please enter your first query.")


    st.header("User Query and Generate Answer")
    user_query = st.text_input("Enter your query:")

    if st.button("Generate Answer"):
        if user_query:
            with st.spinner("Querying the database and generating answer..."):
                driver = connect_to_neo4j(neo4j_uri, neo4j_user, neo4j_password)
                try:
                    relevant_papers = query_relevant_papers(driver, user_query)
                    if relevant_papers:
                        most_relevant_paper = relevant_papers[0]
                        st.write(f"**Most Relevant Paper**: {most_relevant_paper['title']}")
                        st.write(f"**Summary**: {most_relevant_paper['summary']}")
                        st.write(f"**Link**: [View Paper]({most_relevant_paper['link']})")

                        pdf_path = download_arxiv_paper(most_relevant_paper["link"])
                        if pdf_path:
                            st.write("**Downloaded Paper PDF**:", pdf_path)
                            context = extract_text_from_pdf(pdf_path)
                            vectors = create_vector_embedding(context, pdf_path)

                            document_chain=create_stuff_documents_chain(load_llm(),prompt)
                            retriever=vectors.as_retriever()
                            retrieval_chain=create_retrieval_chain(retriever,document_chain)
                            response=retrieval_chain.invoke({'input':user_query})
                            st.write("**Answer to Your Query:**")
                            st.write(response['answer'])
                        else:
                            st.error("Could not download the paper PDF.")

                    else:
                        st.info("No relevant papers found for the query.")
                except Exception as e:
                    st.error(f"An error occurred while querying data: {e}")
                finally:
                    driver.close()

    if st.button("Generate Future Directions"):
        if user_query:
            with st.spinner("Generating future research directions..."):
                driver = connect_to_neo4j(neo4j_uri, neo4j_user, neo4j_password)
                try:
                    most_relevant_papers = query_relevant_papers(driver, user_query)
                    if most_relevant_papers:
                        future_directions = generate_future_directions(most_relevant_papers)
                        st.write("**Future Research Directions:**")
                        st.write(future_directions)
                    else:
                        st.info("No relevant papers found for the query.")
                except Exception as e:
                    st.error(f"An error occurred while generating future directions: {e}")
                finally:
                    driver.close()

if __name__ == "__main__":
    main()
