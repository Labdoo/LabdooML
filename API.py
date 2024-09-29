from fastapi import FastAPI, Query
from langchain_community.vectorstores import Chroma
import os
from langchain_openai import OpenAIEmbeddings
from Secret import OAIKey
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
import pickle
from langdetect import detect
import sqlite3
from difflib import SequenceMatcher
import faiss
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load environment variables and initialize models
os.environ["OPENAI_API_KEY"] = OAIKey
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory_ = os.path.join(current_dir, "Data", "vector_db")
persistent_directory = os.path.join(persistent_directory_, 'chroma_labdoo')

# Load the databases and retrievers once at startup
dbDocs = Chroma(collection_name="Documents", persist_directory=persistent_directory, embedding_function=embeddings)
dbQA = Chroma(collection_name="QADocs", persist_directory=persistent_directory, embedding_function=embeddings)
docRetriever = dbDocs.as_retriever(search_kwargs={'k': 1})
docDirectory = os.path.join(persistent_directory_, 'docs_labdoo')
with open((os.path.join(docDirectory, 'retriever.pkl')), "rb") as file:
	data = pickle.load(file)
store = InMemoryByteStore()
store.mset(list(data.items()))
QARetriever = MultiVectorRetriever(
	vectorstore=dbQA,
	byte_store=store,
	id_key="doc_id",
	search_kwargs={'k': 2}
)

SIMILARITY_THRESHOLD = 0.9

FAISS_INDEX_FILE = 'faiss_index.index'
embedding_dimension = 1536
index = None


# Initialize GPT model
GPT = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=500, timeout=45, max_retries=0)

# DEtects if language is in English
def is_english(text):
	try:
		language = detect(text)
		return language == 'en'  # Check if the detected language is English ('en' stands for English)
	except:
		return False 

# If the language is not in english it will translate the query just to perform similarity check
def RouteLanguage(query):
	if len(query) > 40:
		query2 = query[:40]
	else:
		query2 = query
	if is_english(query2):
		return query
	else:
		messages = [
			('system', "Translate to the best of your ability the user query to english"),
			('human', "User question: {query}")
		]
		template = ChatPromptTemplate.from_messages(messages)
		prompt = template.invoke({'query': query})
		response = GPT.invoke(prompt)
		return response.content


# Cache with semantic similarity NOT TESTED (THE SYSTEM IS UNDER TESTING)
def create_connection(db_file='chatbot_cache.db'):
	conn = sqlite3.connect(db_file)
	return conn

def get_openai_embedding(query):
	response = embeddings.embed_query(query)
	return np.array(response)

def load_faiss_index():
	global index
	if os.path.exists(FAISS_INDEX_FILE):
		index = faiss.read_index(FAISS_INDEX_FILE)
		print("Faiss index loaded from disk.")
	else:
		base_index = faiss.IndexFlatL2(embedding_dimension)
		index = faiss.IndexIDMap2(base_index) 
		print("New Faiss index created.")

def save_faiss_index():
	global index
	faiss.write_index(index, FAISS_INDEX_FILE)
	print("Faiss index saved to disk.")

def CacheAnswer(query, answer):
	global index
	conn = create_connection()
	cursor = conn.cursor()
	try:
		cursor.execute("INSERT INTO cache (query, answer) VALUES (?, ?)", (query, answer))
		conn.commit()
		id = cursor.lastrowid
		vector = get_openai_embedding(query)
		index.add(np.array([vector]), np.array([id]))
		save_faiss_index()
	except Exception as e:
		print(e)
		pass
	conn.close()

def CheckCache(query):
	global index
	conn = create_connection()
	cursor = conn.cursor()
	query_embedding = get_openai_embedding(query)
	D, I = index.search(np.array([query_embedding]), k=12)  
	for sqlite_row_id, distance in zip(I[0], D[0]):
			cursor.execute("SELECT query, answer FROM cache WHERE id = ?", (sqlite_row_id,))
			result = cursor.fetchone()
			if result:
				cached_query, cached_answer = result
				similarity = SequenceMatcher(None, query, cached_query).ratio()
				if similarity >= SIMILARITY_THRESHOLD:
					conn.close()
					return cached_answer
	conn.close()
	return None

load_faiss_index()


# Cache without semnatic similarity
"""def create_connection(db_file='chatbot_cache.db'):
	conn = sqlite3.connect(db_file)
	return conn

def CacheAnswer(query, answer):
	conn = create_connection()
	cursor = conn.cursor()
	cursor.execute("SELECT query FROM cache")
	rows = cursor.fetchall()
	for row in rows:
		existing_query = row[0]
		similarity = SequenceMatcher(None, query, existing_query).ratio()
		if similarity >= SIMILARITY_THRESHOLD:
			conn.close()
			return
	try:
		cursor.execute("INSERT INTO cache (query, answer) VALUES (?, ?)", (query, answer))
		conn.commit()
	except sqlite3.IntegrityError:
		pass
	finally:
		conn.close()

def CheckCache(query):
	conn = create_connection()
	cursor = conn.cursor()
	cursor.execute("SELECT query, answer FROM cache")
	rows = cursor.fetchall()
	for row in rows:
		existing_query, answer = row
		similarity = SequenceMatcher(None, query, existing_query).ratio()
		if similarity >= SIMILARITY_THRESHOLD:
			conn.close()
			return answer
	conn.close()
	return None	"""

# Retrieves relevant docs
def GetContext(query, docRetriv, QARetriv):
	context = set()
	context.add(docRetriv.invoke(query)[0].page_content)
	context.add(QARetriv.invoke(query)[0].page_content)
	return "\n\n".join(context)

# Gets the retrieved docs and answers the users question
def AnswerQuestion(query, docRetriv, QARetriv, GPT, messages=None, cache=True):
	if cache:
		response = CheckCache(query)
		if response != None: return response
	contextQuery = RouteLanguage(query)
	context = GetContext(contextQuery, docRetriv, QARetriv)
	print(context)
	if not messages:
		messages = [
			('system', "Act as a Q&A chatbot who must respond to the user query using information provided by RAG. JUST use the information provided to answer the query. Respond in a concise and clear way in 1-2 paragraphs but making sure to completely answer the query considering the user is a new labdoo user. Use clear and simple language. If the user asks you for detailed explanation, you should response in full detail"),
			('human', "RAG context: \n {context}\n\n User question: {query}")
		]
	template = ChatPromptTemplate.from_messages(messages)
	prompt = template.invoke({'context': context, 'query': query})
	response = GPT.invoke(prompt).content
	if cache:
		CacheAnswer(query, response)
	return response

# API endpoint
@app.get("/ask")
def ask_question(query: str = Query(..., description="The question you want to ask")):
	answer = AnswerQuestion(query, docRetriever, QARetriever, GPT)
	return {"question": query, "answer": answer}


# To run the FastAPI server:
# Use the following command in your terminal:
# uvicorn filename:app --reload
