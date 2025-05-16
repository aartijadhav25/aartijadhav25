import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
product_name = []
Prices = []
Description = []
Ratings = []
for i in range(2,5):
    url = "https://www.flipkart.com/search?q=laptops&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=off&as=off&page="+str(i)

r = requests.get(url)

soup = BeautifulSoup(r. text, "lxml")
box = soup.find("div",class_ = "DOjaWF gdgoEp")

names = box.find_all("div", class_ = "KzDlHZ")
# print(names)

for i in names:
    name = i.text
    product_name.append(name)
    print(product_name)

prices = box.find_all("div", class_ = "hl05eU")
for i in prices:
    name = i.text
    Prices.append(name)

#print(Prices)

desc = box.find_all("ul",class_ = "G4BRas")

for i in desc:
    name = i.text
    Description.append(name)

#print(Description)

ratings = box.find_all("div", class_ = "XQDdHH")

for i in ratings:
    name = i.text
    Ratings.append(name)

#print(len(Ratings))
min_len=96
product_name = product_name[:min_len]
Prices = Prices[:min_len]
Description = Description[:min_len]
Ratings = Ratings[:min_len]

df = pd.DataFrame({"Product Name":product_name,"Price":Prices,"Description":Description,"Ratings":Ratings})
print(df)
data=df[['Product Name','Price','Description']]
data
df.head()
df.tail()
df.columns
df=df[["Product Name","Price","Description"]]
df.to_csv('D:\E-commerce.csv')
df=pd.read_csv('D:\E-commerce.csv')
product_list=[]
#itrate over the rows of the dataframe
for index, row in data.iterrows():
  object={
      'name':row['Product Name'],
      'price':row['Price'],
      'description':row['Description']
}

#append the obj to the product list
  product_list.append(object)
  print(product_list)
  product_list[0]
  !pip install langchain
pip install langchain_core
from langchain_core.documents import Document
docs=[]

for object in product_list:
    print(object['description'])
    metadata={'product_name':object['name']}
    page_content=object['description']
    doc=Document(page_content=page_content,metadata=metadata)
    docs.append(doc)
    len(docs) # length of document
    docs[0]
    GROQ_API = "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"
ASTRA_DB_API_ENDPOINT = "https://5e558417-3845-4dfb-a2b7-1f4880b383ae-us-east-2.apps.astra.datastax.com"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:iZjkJsbvJmrpjjbjofFpRSKy:e74fc5611130c5178ddc3fd0114f84ca8bb159a570031fa58f65b450a8654c06"
ASTRA_DB_KEYSPACE = 'default_keyspace'
HF_TOKEN = "hf_WsHRqvdxzPAasrrCPKTXDdgeXrrulfiJm
import requests
import time

class HuggingFaceInferenceAPIEmbeddings:
    def _init_(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self._api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def embed_documents(self, texts):
        response = requests.post(
            self._api_url,
            headers=self._headers,
            json={"inputs": texts},
        )
        print("Response Status Code:", response.status_code)

        if response.status_code == 503:
            print("Model is loading. Retrying in 10 seconds...")
            time.sleep(10)
            return self.embed_documents(texts)

        if response.status_code == 200:
            embeddings = response.json()
            return embeddings

        print("Response Text:", response.text)  # For debugging other errors
        raise Exception(f"Error: {response.status_code} - {response.text}")

    def embed_query(self, text):
        """
        Embed a single query (text) and return the embedding vector.
        This method is required by AstraDBVectorStore.
        """
        result = self.embed_documents([text])  # This will call embed_documents
        if isinstance(result, list):
            return result[0]  # Return the first embedding (vector) for the query
        else:
            raise ValueError("Unexpected result format in embed_query method.")
        from langchain_astradb import AstraDBVectorStore
        from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Initialize AstraDBVectorStore
vstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="flipkart",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_KEYSPACE
)
insert_ids = vstore.add_documents(docs)
pip install langchain-groq
from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
model = ChatGroq(groq_api_key = GROQ_API,model="mistral",tempature=0.5)
retriever_prompt = ("Given a chat hi"
"story and the latest user question which might reference context in the chat history,"
                    "formulate a standalone question which can be understood witout the chat history."
                    "Do NOT answer the queston, just reformulate if it needed and otherwise return it as is.")
retriver = vstore.as_retriever(search_kwargs={"k":3})
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
pip install --upgrade langchain
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",retriever_prompt),
        MessagesPlaceholder(variable_name= "chat_histroy"),
        ("human","{input}")
    ]
)
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama
model = ChatOllama(model="mistral")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain.vectorstores import Chroma
pip install chromadb
retriever = Chroma(persist_directory="db", embedding_function=embedding).as_retriever()
pip install -U langchain
history_aware_retriever  = create_history_aware_retriever(
    llm=model,
    retriever=retriever,
    prompt=contextualize_q_prompt
)
PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    QUESTION: {input}

    YOUR ANSWER:

    """
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
qa_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", PRODUCT_BOT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
        ]
  )
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
chat_history = []
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
store ={}
def get_session_history(session_id: str)-> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id]= ChatMessageHistory()
  return store[session_id]
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
chain_with_memory.invoke(
       {"input": "can you tell me the best bluetooth buds?"},
       config={
        "configurable": {"session_id": "dhruv"}
    } ,# constructs a key "abc123" in `store`.
)
["answer"]