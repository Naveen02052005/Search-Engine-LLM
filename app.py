import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
load_dotenv()
import wikipedia

wikipedia.set_user_agent(
    "SearchEngineLLM/1.0 (educational project)"
)

## Arxiv and wikipedia tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)


api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchResults(name="Search")

st.title("🔎 Langchain - Chat with search")


## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key: ",type="password")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


if prompt:=st.chat_input(placeholder="What is machine learning?"):
    if not api_key:
        st.warning("Please enter your Groq API Key.")
    else:
        api_key = api_key.strip()
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").write(prompt)


        llm = ChatGroq(groq_api_key=api_key,model="openai/gpt-oss-120b",streaming=True)
        tools = [search,arxiv,wiki]

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system","You are a helpful assistant who can search the web, Arxiva and Wikipedia."),
                ("human","{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        agent = create_tool_calling_agent(llm,tools,prompt_template)
        search_agent = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response = search_agent.invoke({"input":prompt},config={"callbacks":[st_cb]})
            output = response["output"]
            st.write(output)
            st.session_state.messages.append({'role':'assistant','content':response})
