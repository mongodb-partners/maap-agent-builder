import json
import uuid
from time import sleep
from typing import List, Literal, Optional

import certifi
import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pymongo import MongoClient

from agent_builder.utils.logger import logger


def create_long_term_memory_agent(
    model,
    connection_str: str,
    namespace: str,
    tools: Optional[List] = None,
    checkpointer=InMemorySaver(),
    name: str = "long_term_memory_agent",
):
    """
    Create a long-term memory agent with vector store capabilities.

    Args:
        model: The language model to use
        connection_str: MongoDB connection string
        namespace: MongoDB namespace (database.collection)
        tools: Optional list of additional tools
        checkpointer: Optional checkpointer for saving state
        name: Name of the agent for logging purposes
    """

    # logger = logging.getLogger(name)
    logger.info(f"Creating long-term memory agent: {name}")

    # validate all inputs and throw error if any are missing
    if not model:
        raise ValueError("Language model is required.")
    if not connection_str:
        raise ValueError("MongoDB connection string is required.")
    if not namespace:
        raise ValueError("Namespace (database.collection) is required.")
    logger.info(
        f"Agent {name}: Initializing with model, connection string, and namespace"
    )

    # Initialize embeddings and vector store
    logger.info(f"Agent {name}: Initializing embeddings and vector store")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    client = MongoClient(connection_str, tlsCAFile=certifi.where())
    recall_vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=connection_str,
        namespace=namespace,
        embedding=embeddings,
        embedding_field="embedding",
        index_name="recall_memory_index",
        text_field="text",
    )
    embedding_dim = len(embeddings.embed_query("this is a test"))
    logger.debug(f"Agent {name}: Embedding dimension: {embedding_dim}")

    def get_user_id(config: RunnableConfig) -> str:
        user_id = config["configurable"].get("user_id")
        if user_id is None:
            logger.error(f"Agent {name}: User ID not provided in configuration")
            raise ValueError("User ID needs to be provided to save a memory.")
        return user_id

    @tool
    def save_recall_memory(memory: str, config: RunnableConfig) -> str:
        """Save memory to vectorstore for later semantic retrieval."""
        user_id = get_user_id(config)
        logger.info(f"Agent {name}: Saving recall memory for user {user_id}")
        document = Document(
            page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
        )
        recall_vector_store.add_documents([document])

        # Ensure the recall memory index exists and is queryable
        indexes = list(recall_vector_store.collection.list_search_indexes())
        if not any(x["name"] == "recall_memory_index" for x in indexes):
            logger.info(f"Agent {name}: Creating recall memory index...")
            recall_vector_store.create_vector_search_index(
                dimensions=embedding_dim,
                filters=[{"type": "filter", "path": "user_id"}],
                update=True,
            )
            # Wait until the index is queryable
            while not any(
                x["name"] == "recall_memory_index" and x.get("queryable") is True
                for x in recall_vector_store.collection.list_search_indexes()
            ):
                logger.info(
                    f"Agent {name}: Waiting for recall memory index to be queryable..."
                )
                sleep(10)
            logger.info(f"Agent {name}: Recall memory index is now queryable.")
        return memory

    @tool
    def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
        """Search for relevant memories."""
        user_id = get_user_id(config)
        logger.info(f"Agent {name}: Searching recall memories for user {user_id}")
        try:
            documents = recall_vector_store.similarity_search(
                query, k=3, pre_filter={"user_id": user_id}
            )
            logger.debug(f"Agent {name}: Found {len(documents)} relevant memories")
            return [document.page_content for document in documents]
        except Exception as e:
            logger.error(f"Agent {name}: Error during memory search: {e}")
            return ["No relevant memories found."]

    # Compose all tools
    all_tools = (tools or []) + [save_recall_memory, search_recall_memories]

    class State(MessagesState):
        recall_memories: List[str]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant with advanced long-term memory"
                " capabilities. Powered by a stateless LLM, you must rely on"
                " external memory to store information between conversations."
                " Utilize the available memory tools to store and retrieve"
                " important details that will help you better attend to the user's"
                " needs and understand their context.\n\n"
                "Memory Usage Guidelines:\n"
                "1. Actively use memory tools (save_core_memory, save_recall_memory)"
                " to build a comprehensive understanding of the user.\n"
                "2. Make informed suppositions and extrapolations based on stored"
                " memories.\n"
                "3. Regularly reflect on past interactions to identify patterns and"
                " preferences.\n"
                "4. Update your mental model of the user with each new piece of"
                " information.\n"
                "5. Cross-reference new information with existing memories for"
                " consistency.\n"
                "6. Prioritize storing emotional context and personal values"
                " alongside facts.\n"
                "7. Use memory to anticipate needs and tailor responses to the"
                " user's style.\n"
                "8. Recognize and acknowledge changes in the user's situation or"
                " perspectives over time.\n"
                "9. Leverage memories to provide personalized examples and"
                " analogies.\n"
                "10. Recall past challenges or successes to inform current"
                " problem-solving.\n\n"
                "## Recall Memories\n"
                "Recall memories are contextually retrieved based on the current"
                " conversation:\n{recall_memories}\n\n"
                "## Instructions\n"
                "Engage with the user naturally, as a trusted colleague or friend."
                " There's no need to explicitly mention your memory capabilities."
                " Instead, seamlessly incorporate your understanding of the user"
                " into your responses. Be attentive to subtle cues and underlying"
                " emotions. Adapt your communication style to match the user's"
                " preferences and current emotional state. Use tools to persist"
                " information you want to retain in the next conversation. If you"
                " do call tools, all text preceding the tool call is an internal"
                " message. Respond AFTER calling the tool, once you have"
                " confirmation that the tool completed successfully.\n\n",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    def agent(state: State) -> State:
        model_with_tools = model.bind_tools(all_tools)
        recall_str = (
            "<recall_memory>\n"
            + "\n".join(state["recall_memories"])
            + "\n</recall_memory>"
        )
        bound = prompt | model_with_tools
        prediction = bound.invoke(
            {
                "messages": state["messages"],
                "recall_memories": recall_str,
            }
        )
        return {
            "messages": [prediction],
        }

    def load_memories(state: State, config: RunnableConfig) -> State:
        convo_str = get_buffer_string(state["messages"])
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(convo_str)
        if len(tokens) > 2048:
            tokens = tokens[:2048]
        convo_str = encoding.decode(tokens)
        recall_memories = search_recall_memories.invoke(convo_str, config)
        return {
            "recall_memories": recall_memories,
        }

    def route_tools(state: State):
        msg = state["messages"][-1]
        if getattr(msg, "tool_calls", None):
            return "tools"
        return END

    # Build the graph
    builder = StateGraph(State)
    builder.add_node(load_memories)
    builder.add_node(agent)
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "agent")
    builder.add_conditional_edges("agent", route_tools, ["tools", END])
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)
