from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
import os
import sys
from langchain_openai import OpenAIEmbeddings
from Secret import OAIKey
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
import ast
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage._lc_store import create_kv_docstore
import uuid
from langchain.storage import LocalFileStore
import pickle
from langchain.storage import InMemoryByteStore
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from langchain_core.output_parsers import StrOutputParser
import umap
from sklearn.mixture import GaussianMixture
import json
import sqlite3

def CreateEmbeddings(embedder: OpenAIEmbeddings, docs: list):
	return embedder.embed_documents(docs)

def InitializeGPT(temp, Mtokens):
	GPT = ChatOpenAI(model="gpt-4o-mini",     
		temperature=temp,
		max_tokens=Mtokens,
		timeout=45,
		max_retries=0,)
	return GPT
os.environ["OPENAI_API_KEY"] = OAIKey
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def GetDB(persistentDir, embedder, collection: str):
	db = Chroma(
	 	collection_name= collection,
		persist_directory=persistentDir,
		embedding_function=embedder,
	)
	return db
def InitialiseEmbeddingsModel():
	embedder = OpenAIEmbeddings(model="text-embedding-3-small")
	return embedder
def CreateRetrieverFromDB(db, texts=2):
	return db.as_retriever(search_kwargs={"k": texts})
def BasicInfoRetrieve(path, texts, minScore, query):
	embeds = InitialiseEmbeddingsModel()
	retriever = CreateRetrieverFromDB(GetDB(path, embeds), texts, minScore)
	print(query)
	return retriever.invoke(query)

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory_ = os.path.join(current_dir, "Data", "vector_db")
persistent_directory = os.path.join(persistent_directory_, 'chroma_labdoo')

# Create Vector Storage
loader = TextLoader('labdooDocsComplex.txt', encoding="utf-8")
documents = loader.load()
embeddings = OpenAIEmbeddings(
	model="text-embedding-3-small"
)
# Function to create and persist vector store
def create_vector_store(docs, persistent_directory, collection):
	if not os.path.exists(persistent_directory):
		db = Chroma.from_documents(
			collection_name=collection, documents=docs, embedding=embeddings, persist_directory=persistent_directory)
# Splits the document into chunks of text by paragraph
def SplitDocs(documents):      
	char_splitter = CharacterTextSplitter(
		chunk_size=2000, separator="\n\n")
	char_docs = char_splitter.split_documents(documents)
	return char_docs
# Summarises the docs and prepares the data to be RAG ready
def PrepareDocs(docs: list[Document], GPT):
	messages = [
	('system', "Act as an expert at preparing data for RAG. Your input will consist of a document and your output must consist of that document but RAG preared. To accomplish this, you must: \n 1- Correct typos anf format inconsistencies \n 2- Reduce noise and eliminate redundant text that doesn't provide useful information (if any) \n 3- Summarise long chunks of information in FULL DETAIL, without loosing any important information but try to make the text shorter \n 4- Remove extra spaces and thigs that could confuse the embeddings model \n 5- Remove text that doesn't provide meanigful info for the model like (Thank you for your support). Your output should consist of JUST this corrected document in plain text format"),
	('human', "Document: {document}")
	]
	template = ChatPromptTemplate.from_messages(messages)
	newDocs = []
	for doc in docs:
		doc_ = doc.page_content
		prompt = template.invoke({'document': doc_})
		response = GPT.invoke(prompt).content
		doc.page_content = response
		newDocs.append(doc)
	return newDocs
# Assign hypothetical questions as keys for the embeddings
def GetMultiVectorKeysAsQuestions(docs: list[Document], GPT, messages=None, questions=3):
	if not messages:
		messages = [
		('system', "Generate {questions} hypothetical questions that the below document could be used to answer, in other words, generate questions that can be answered with the document, make these questions with RAG in mind. Output JUST these questions as it was a python list BUT in raw text"),
		('human', "Document: {document}")
		]
	template = ChatPromptTemplate.from_messages(messages)
	keys=[]
	for doc in docs:
		prompt = template.invoke({'document': doc.page_content, 'questions': questions})
		responses = GPT.invoke(prompt).content
		print(responses)
		responses = ast.literal_eval(responses)
		keys.append([Document(page_content=response, metadata=doc.metadata) for response in responses])
		print("doc summarised")
	return keys
def normalize_encoding(text):
	return text.encode('utf-8', errors='ignore').decode('utf-8')
def AddKeyedDocsToVDB(docs: list[Document], keys: list[list[Document]], vectorDB: Chroma, doc_persistent_directory, docstore=None):
	print(doc_persistent_directory)
	store = InMemoryByteStore()
	id_key = "doc_id"
	retriever = MultiVectorRetriever(
		vectorstore=vectorDB,
		byte_store=store,
		id_key=id_key,
	)
	if docstore:
		retriever = LoadQARetriever(docstore, None, vectorDB, 1)
	sub_docs = []
	doc_ids = [str(uuid.uuid4()) for _ in docs]
	for i, doc in enumerate(docs):
		_sub_docs = []
		_id = doc_ids[i]
		for _doc in keys[i]:
			_doc.metadata[id_key] = _id
			_sub_docs.append(_doc)
		sub_docs.extend(_sub_docs)
	retriever.vectorstore.add_documents(sub_docs)
	retriever.docstore.mset(list(zip(doc_ids, docs)))
	with open((os.path.join(doc_persistent_directory, 'retriever.pkl')), 'wb') as f:
		pickle.dump(retriever.byte_store.store, f, pickle.HIGHEST_PROTOCOL)

 
# Answering question
def LoadQARetriever(path, embedder, vdb, texts):
	with open((os.path.join(path, 'retriever.pkl')), "rb") as file:
		data = pickle.load(file)
	store = InMemoryByteStore()
	store.mset(list(data.items()))
	retriever = MultiVectorRetriever(
		vectorstore=vdb,
		byte_store=store,
		id_key="doc_id",
		search_kwargs={'k': texts+1}
	)
	return retriever
def GetContext(query, docRetriv, QARetriv):
	context = set()
	context.add(docRetriv.invoke(query)[0].page_content)
	context.add(QARetriv.invoke(query)[0].page_content)
	return "\n\n".join(context)
def AnswerQuestion(query, docRetriv, QARetriv, GPT, messages=None):
	context = GetContext(query, docRetriv, QARetriv)
	if not messages:
		messages = [
		('system', "Act as a Q&A chatbot who must respond the user query using information provided by RAG. JUST use the information provided to answer the query, if info is not relevant to the query then you are allowed to use your own knowledge. Respond in a concise and clear way using only 2-3 sentences unless the question requires a longer explanation."),
		('human', "RAG context: \n {context}\n\n User question: {query}")
		]
	template = ChatPromptTemplate.from_messages(messages)
	prompt = template.invoke({'context': context, 'query': query})
	response = GPT.invoke(prompt)
	return response

# RAPTOR data
def SplitDocsRAPTOR(documents):      
	char_splitter = CharacterTextSplitter(
		chunk_size=5000, separator="\n\n")
	char_docs = char_splitter.split_documents(documents)
	for doc in char_docs:
		print(len(doc.page_content))
	return char_docs
def PrepareDocsRAPTOR(docs: list[Document], GPT):
	messages = [
	('system', "Act as an expert summariser. Your input will consist of raw data and your output must consist of a RAG friendly very detailed summary, remember this summary is to be used FOR RAG not for human reading. To accomplish this, you must: \n 1- Correct typos and format inconsistencies \n 2- Reduce noise and eliminate redundant text that doesn't provide useful information or raw text that is not RAG frienly, like references to images as the RAG will have no images (if any) \n 3- Summarise the text keeping all important information but in a much more synthesized way \n 4- Remove extra spaces and things that could confuse the embeddings model \n 5- Remove text that doesn't provide meanigful info for the model like (Thank you for your support). Your output should consist of JUST this RAG friendly summary in plain text format, don't add filler sentences"),
	('human', "Raw data: {document}")
	]
	template = ChatPromptTemplate.from_messages(messages)
	newDocs = []
	for doc in docs:
		doc_ = doc.page_content
		prompt = template.invoke({'document': doc_})
		response = GPT.invoke(prompt).content
		doc.page_content = response
		newDocs.append(doc)
	return newDocs
# Actual RAPTOR
# Gaussian embedding clustering
RANDOM_SEED = 224
def global_cluster_embeddings(
	embeddings: np.ndarray,
	dim: int,
	n_neighbors: Optional[int] = None,
	metric: str = "cosine",
) -> np.ndarray:
	if n_neighbors is None:
		n_neighbors = int((len(embeddings) - 1) ** 0.5)
	return umap.UMAP(
		n_neighbors=n_neighbors, n_components=dim, metric=metric
	).fit_transform(embeddings)
def local_cluster_embeddings(
	embeddings: np.ndarray, dim: int, num_neighbors: int = 4, metric: str = "cosine"
) -> np.ndarray:
	return umap.UMAP(
		n_neighbors=num_neighbors, n_components=dim, metric=metric
	).fit_transform(embeddings)
def get_optimal_clusters(
	embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
	max_clusters = min(max_clusters, len(embeddings))
	n_clusters = np.arange(1, max_clusters)
	bics = []
	for n in n_clusters:
		gm = GaussianMixture(n_components=n, random_state=random_state)
		gm.fit(embeddings)
		bics.append(gm.bic(embeddings))
	return n_clusters[np.argmin(bics)]
def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
	n_clusters = get_optimal_clusters(embeddings)
	gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
	gm.fit(embeddings)
	probs = gm.predict_proba(embeddings)
	labels = [np.where(prob > threshold)[0] for prob in probs]
	return labels, n_clusters
def perform_clustering(
	embeddings: np.ndarray,
	dim: int,
	threshold: float,
) -> List[np.ndarray]:
	if len(embeddings) <= dim + 1:
		return [np.array([0]) for _ in range(len(embeddings))]
	reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
	global_clusters, n_global_clusters = GMM_cluster(
		reduced_embeddings_global, threshold
	)
	all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
	total_clusters = 0
	for i in range(n_global_clusters):
		global_cluster_embeddings_ = embeddings[
			np.array([i in gc for gc in global_clusters])
		]
		if len(global_cluster_embeddings_) == 0:
			continue
		if len(global_cluster_embeddings_) <= dim + 1:
			local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
			n_local_clusters = 1
		else:
			reduced_embeddings_local = local_cluster_embeddings(
				global_cluster_embeddings_, dim
			)
			local_clusters, n_local_clusters = GMM_cluster(
				reduced_embeddings_local, threshold
			)
		for j in range(n_local_clusters):
			local_cluster_embeddings_ = global_cluster_embeddings_[
				np.array([j in lc for lc in local_clusters])
			]
			indices = np.where(
				(embeddings == local_cluster_embeddings_[:, None]).all(-1)
			)[1]
			for idx in indices:
				all_local_clusters[idx] = np.append(
					all_local_clusters[idx], j + total_clusters
				)
		total_clusters += n_local_clusters
	return all_local_clusters
def RAPTOR(docs: list[Document], GPT, messages=None, recursivity=3) -> list[Document]:
	docs_texts = [d.page_content for d in docs]
	summaries = RecursiveRAPTOR(docs_texts, GPT, messages, 1, recursivity=recursivity)
	texts = docs_texts.copy()
	for level in sorted(summaries.keys()):
		summariesCluster = summaries[level][1]["summaries"].tolist()
		texts.extend(summariesCluster)
	finalDocs = []
	i = 0
	for text in texts:
		if i < len(docs):
			doc = Document(page_content=text, metadata=docs[i])
		else:
			doc = Document(page_content=text)
		i+=1
		finalDocs.append(doc)
	return finalDocs
def FormatDfText(df: pd.DataFrame, column):
	unique_txt = df[column].tolist()
	final_text = "--- --- \n --- --- ".join(unique_txt)
	return final_text
def RecursiveRAPTOR(docs: list[str], GPT, messages=None, level=1, recursivity=2):
	results = {}
	dfClusters, dfSummaries = RAPTORStep(docs, GPT, 0, messages)
	results[level] = (dfClusters, dfSummaries)
	NClusters = dfSummaries['cluster'].nunique()
	if level < recursivity and NClusters > 1:
		new_texts = dfSummaries["summaries"].tolist()
		next_level_results = RecursiveRAPTOR(
			new_texts, GPT, messages, level+1, recursivity
		)
		results.update(next_level_results)
	return results
def RAPTORStep(docs, GPT, level, messages=None):
	embeddings = np.array(CreateEmbeddings(InitialiseEmbeddingsModel(), docs))
	cluster_labels = perform_clustering(
	embeddings, 10, 0.1
	)  # Perform clustering on the embeddings
	df = pd.DataFrame()  # Initialize a DataFrame to store the results
	df["text"] = docs  # Store original texts
	df["embd"] = list(embeddings)  # Store embeddings as a list in the DataFrame
	df["cluster"] = cluster_labels  # Store cluster labels
	dfExpansion = []
	for index, row in df.iterrows():
		for cluster in row["cluster"]:
			dfExpansion.append(
				{"text": row["text"], "embd": row["embd"], "cluster": cluster}
			)
	ExpandedDf = pd.DataFrame(dfExpansion)
	all_clusters = ExpandedDf["cluster"].unique()
	if not messages:
		messages = [
		('system', "Act as an expert summariser. Summarise the following document in full detaill without missing important information, to be stored for RAG. Repsond JUST with the summary. Make sure th summary includes all the relevant information"),
		('human', "Document: {document}")
		]
	prompt = ChatPromptTemplate.from_messages(messages)
	chain = prompt | GPT | StrOutputParser()
	summaries=[]
	for cluster in all_clusters:
		df_cluster = ExpandedDf[ExpandedDf["cluster"] == cluster]
		formatted_txt = FormatDfText(df_cluster, "text")
		summaries.append(chain.invoke({"document": formatted_txt}))
	df_summary = pd.DataFrame(
		{
			"summaries": summaries,
			"level": [level] * len(summaries),
			"cluster": list(all_clusters),
		}
	)
	return df, df_summary	
def SaveDocs(docs: list[Document]):
	docs_as_dicts = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
	with open('langchain_docs.json', 'w') as f:
		json.dump(docs_as_dicts, f)
def LoadDocs():
	with open('langchain_docs.json', 'r') as f:
		docs = json.load(f)
	docs = [Document(page_content=doc['content'], metadata={'source': 'ComplexLabdoo'}) for doc in docs]
	return docs
def AddToVDB(vectorDb: Chroma, docs):
	vectorDb.add_documents(docs)
def InitialiseCache():
	conn = sqlite3.connect('chatbot_cache.db')
	cursor = conn.cursor()
	cursor.execute('''
		CREATE TABLE IF NOT EXISTS cache (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			query TEXT UNIQUE,
			answer TEXT
		)
	''')
	cursor.execute('CREATE INDEX IF NOT EXISTS idx_query ON cache (query)')
	conn.commit()
	conn.close()

InitialiseCache()

# Creating the VectorDB
docDirectory = os.path.join(persistent_directory_, 'docs_labdoo')
if not os.path.exists(persistent_directory):
	docs = SplitDocs(documents)
	docs = PrepareDocs(docs, InitializeGPT(0.1, 2000))
	create_vector_store(docs, persistent_directory, "Documents")
	docs = GetDB(persistent_directory, embeddings, "Documents").similarity_search("eddovillage", 2000)
	keys = GetMultiVectorKeysAsQuestions(docs, InitializeGPT(0.2, 300), questions=4)
	AddKeyedDocsToVDB(docs, keys, GetDB(persistent_directory, embeddings, "QADocs"), docDirectory)
	docs = LoadDocs()
	AddToVDB(GetDB(persistent_directory, embeddings, "Documents"), docs)
	keys = GetMultiVectorKeysAsQuestions(docs, InitializeGPT(0.2, 300), questions=4)
	AddKeyedDocsToVDB(docs,keys,GetDB(persistent_directory, embeddings, "QADocs"), docDirectory, docDirectory)
# Retrieval (we would retrieve 1 doc using the doc embeddings and the other doc using the QA embeddings)
dbDocs = GetDB(persistent_directory, embeddings, "Documents")
dbQA = GetDB(persistent_directory, embeddings, "QADocs")
docRetriever = dbDocs.as_retriever(search_kwargs={'k': 1})
QARetriever = LoadQARetriever(docDirectory, embeddings, dbQA, 1)
print(QARetriever)
while True:
	query = input("Test Question: ")
	print(AnswerQuestion(query, docRetriever, QARetriever, InitializeGPT(0, 500)))

	








