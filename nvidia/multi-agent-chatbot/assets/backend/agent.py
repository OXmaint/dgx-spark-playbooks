#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""ChatAgent implementation for LLM-powered conversational AI with tool calling."""

import asyncio
import contextlib
import json
import re
from typing import AsyncIterator, List, Dict, Any, TypedDict, Optional, Callable, Awaitable

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage, ToolMessage, ToolCall
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from client import MCPClient
from logger import logger
from prompts import Prompts
from postgres_storage import PostgreSQLConversationStorage
from utils import convert_langgraph_messages_to_openai


memory = MemorySaver()
SENTINEL = object()
StreamCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class State(TypedDict, total=False):
    iterations: int
    messages: List[AnyMessage]
    chat_id: Optional[str]
    image_data: Optional[str]
    annotated_image: Optional[str]


class ChatAgent:
    """Main conversational agent with tool calling and agent delegation capabilities.
    
    This agent orchestrates conversation flow using a LangGraph state machine that can:
    - Generate responses using LLMs
    - Execute tool calls (including MCP tools)
    - Handle image processing
    - Manage conversation history via Redis
    """

    def __init__(self, vector_store, config_manager, postgres_storage: PostgreSQLConversationStorage):
        """Initialize the chat agent.
        
        Args:
            vector_store: VectorStore instance for document retrieval
            config_manager: ConfigManager for reading configuration
            postgres_storage: PostgreSQL storage for conversation persistence
        """
        self.vector_store = vector_store
        self.config_manager = config_manager
        self.conversation_store = postgres_storage
        self.current_model = None
        
        self.current_model = None
        self.max_iterations = 3
        
        self.mcp_client = None
        self.openai_tools = None
        self.tools_by_name = None
        self.system_prompt = None
        
        self.graph = self._build_graph()
        self.stream_callback = None
        self.last_state = None

    @classmethod
    async def create(cls, vector_store, config_manager, postgres_storage: PostgreSQLConversationStorage):
        """
        Asynchronously creates and initializes a ChatAgent instance.
        
        This factory method ensures that all async setup, like loading tools,
        is completed before the agent is ready to be used.
        """
        agent = cls(vector_store, config_manager, postgres_storage)
        await agent.init_tools()
        
        available_tools = list(agent.tools_by_name.values()) if agent.tools_by_name else []
        template_vars = {
            "tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in available_tools]) if available_tools else "No tools available",
        }
        agent.system_prompt = Prompts.get_template("supervisor_agent").render(template_vars)
        
        logger.debug(f"Agent initialized with {len(available_tools)} tools.")
        agent.set_current_model(config_manager.get_selected_model())
        return agent

    async def init_tools(self) -> None:
        """Initialize MCP client and tools with retry logic.
        
        Sets up the MCP client, retrieves available tools, converts them to OpenAI format,
        and initializes specialized agents like the coding agent.
        """
        self.mcp_client = await MCPClient().init()
        
        base_delay, max_retries = 0.1, 10
        mcp_tools = []
        
        for attempt in range(max_retries):
            try:
                mcp_tools = await self.mcp_client.get_tools()
                break
            except Exception as e:
                logger.warning(f"MCP tools initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"MCP servers not ready after {max_retries} attempts, continuing without MCP tools")
                    mcp_tools = []
                    break
                wait_time = base_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
                logger.info(f"MCP servers not ready, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
        
        self.tools_by_name = {tool.name: tool for tool in mcp_tools}
        logger.debug(f"Loaded {len(mcp_tools)} MCP tools: {list(self.tools_by_name.keys())}")
        
        if mcp_tools:
            mcp_tools_openai = [convert_to_openai_tool(tool) for tool in mcp_tools]
            logger.debug(f"MCP tools converted to OpenAI format: {mcp_tools_openai}")
            
            self.openai_tools = [
                {"type": "function", "function": tool['function']} 
                for tool in mcp_tools_openai
            ]
            logger.debug(f"Final OpenAI tools format: {self.openai_tools}")
        else:
            self.openai_tools = []
            logger.warning("No MCP tools available - agent will run with limited functionality")

    def set_current_model(self, model_name: str) -> None:
        """Set the current model for completions.
        
        Args:
            model_name: Name of the model to use
            
        Raises:
            ValueError: If the model is not available
        """
        available_models = self.config_manager.get_available_models()

        try:
            if model_name in available_models:
                self.current_model = model_name
                logger.info(f"Switched to model: {model_name}")
                
                   #Use Ollama for llama3-oxmaint model
                   if model_name == "llama-oxmaint":
                   self.model_client = AsyncOpenAI(
                    base_url="http://host.docker.internal:11434/v1,
                    api_key="api_key"
                    #self.model_client = AsyncOpenAI(
                    #base_url=f"http://{self.current_model}:8000/v1",
                    #api_key="api_key"
                )
            else:
                raise ValueError(f"Model {model_name} is not available. Available models: {available_models}")
        except Exception as e:
            logger.error(f"Error setting current model: {e}")
            raise ValueError(f"Model {model_name} is not available. Available models: {available_models}")

    def should_continue(self, state: State) -> str:
        """Determine whether to continue the tool calling loop.
        
        Args:
            state: Current graph state
            
        Returns:
            "end" if no more tool calls or max iterations reached, "continue" otherwise
        """
        messages = state.get("messages", [])
        if not messages:
            return "end"
            
        last_message = messages[-1]
        iterations = state.get("iterations", 0)
        has_tool_calls = bool(last_message.tool_calls) if hasattr(last_message, 'tool_calls') else False

        logger.debug({
            "message": "GRAPH: should_continue decision",
            "chat_id": state.get("chat_id"),
            "iterations": iterations,
            "max_iterations": self.max_iterations,
            "has_tool_calls": has_tool_calls,
            "tool_calls_count": len(last_message.tool_calls) if has_tool_calls else 0
        })

        if iterations >= self.max_iterations:
            logger.debug({
                "message": "GRAPH: should_continue → END (max iterations reached)",
                "chat_id": state.get("chat_id"),
                "final_message_preview": str(last_message)[:100] + "..." if len(str(last_message)) > 100 else str(last_message)
            })
            return "end"

        if not has_tool_calls:
            logger.debug({"message": "GRAPH: should_continue → END (no tool calls)", "chat_id": state.get("chat_id")})
            return "end"

        logger.debug({"message": "GRAPH: should_continue → CONTINUE (has tool calls)", "chat_id": state.get("chat_id")})
        return "continue"

    async def tool_node(self, state: State) -> Dict[str, Any]:
        """Execute tools from the last AI message's tool calls.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with tool results and incremented iteration count
        """
        logger.debug({
            "message": "GRAPH: ENTERING NODE - action/tool_node",
            "chat_id": state.get("chat_id"),
            "iterations": state.get("iterations", 0)
        })
        await self.stream_callback({'type': 'node_start', 'data': 'tool_node'})
        
        outputs = []
        messages = state.get("messages", [])
        last_message = messages[-1]
        for i, tool_call in enumerate(last_message.tool_calls):
            logger.debug(f'Executing tool {i+1}/{len(last_message.tool_calls)}: {tool_call["name"]} with args: {tool_call["args"]}')
            await self.stream_callback({'type': 'tool_start', 'data': tool_call["name"]})
            
            try:
                if tool_call["name"] == "explain_image":
                    tool_args = tool_call["args"].copy()
                    # If image data is in state (current upload), use it
                    if state.get("image_data") and not tool_args.get("image"):
                        tool_args["image"] = state["image_data"]
                    # If image_id is provided, fetch from database
                    elif tool_args.get("image_id") and not tool_args.get("image"):
                        image_id = tool_args.get("image_id")
                        stored_image = await self.conversation_store.get_image(image_id)
                        if stored_image:
                            tool_args["image"] = stored_image
                            logger.info(f"[EXPLAIN_IMAGE] Retrieved image from database: {image_id}")
                        else:
                            logger.warning(f"[EXPLAIN_IMAGE] Image not found in database: {image_id}")
                    logger.info(f'Executing tool {tool_call["name"]} with args: {tool_args}')
                    tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_args)
                    state["process_image_used"] = True
                elif tool_call["name"] == "annotate_image":
                    tool_args = tool_call["args"].copy()
                    # If image data is in state (current upload), use it
                    if state.get("image_data") and not tool_args.get("image"):
                        tool_args["image"] = state["image_data"]
                    # If image_id is provided, fetch from database
                    elif tool_args.get("image_id") and not tool_args.get("image"):
                        image_id = tool_args.get("image_id")
                        stored_image = await self.conversation_store.get_image(image_id)
                        if stored_image:
                            tool_args["image"] = stored_image
                            logger.info(f"[ANNOTATE_IMAGE] Retrieved image from database: {image_id}")
                        else:
                            logger.warning(f"[ANNOTATE_IMAGE] Image not found in database: {image_id}")
                    logger.info(f'Executing tool {tool_call["name"]} with args: {tool_args}')
                    tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_args)

                    # Log the tool result for debugging
                    logger.info(f"[ANNOTATE_IMAGE] Tool returned result of length: {len(tool_result) if tool_result else 0}")
                    logger.info(f"[ANNOTATE_IMAGE] Result starts with: {tool_result[:50] if tool_result else 'None'}...")
                    logger.info(f"[ANNOTATE_IMAGE] Result is data URL: {tool_result.startswith('data:image/') if tool_result else False}")

                    # Store the annotated image in state for frontend display
                    state["annotated_image"] = tool_result
                    logger.info(f"[ANNOTATE_IMAGE] Stored in state['annotated_image'], length: {len(state.get('annotated_image', ''))}")

                    # Return a short confirmation to LLM instead of the full base64 image
                    num_boxes = len(tool_args.get("bounding_boxes", []))
                    tags = tool_args.get("tags", [])
                    tag_info = f" with labels: {', '.join(tags)}" if tags else ""
                    tool_result = f"Successfully annotated image with {num_boxes} bounding box(es){tag_info}. The annotated image will be automatically displayed to the user. Do NOT include any image data, base64 strings, or markdown image syntax in your response - just describe what was annotated."
                else:
                    tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])

                if "code" in tool_call["name"]:
                    content = str(tool_result)
                elif isinstance(tool_result, str):
                    content = tool_result
                else:
                    content = json.dumps(tool_result)
            except Exception as e:
                logger.error(f'Error executing tool {tool_call["name"]}: {str(e)}', exc_info=True)
                content = f"Error executing tool '{tool_call['name']}': {str(e)}"
            
            await self.stream_callback({'type': 'tool_end', 'data': tool_call["name"]})

            outputs.append(
                ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        state["iterations"] = state.get("iterations", 0) + 1

        logger.debug({
            "message": "GRAPH: EXITING NODE - action/tool_node",
            "chat_id": state.get("chat_id"),
            "iterations": state.get("iterations"),
            "tools_executed": len(outputs),
            "next_step": "→ returning to generate"
        })
        await self.stream_callback({'type': 'node_end', 'data': 'tool_node'})

        # Build return dict with all state updates
        result = {
            "messages": messages + outputs,
            "iterations": state.get("iterations", 0) + 1
        }

        # Include annotated_image if it was set
        if state.get("annotated_image"):
            result["annotated_image"] = state["annotated_image"]
            logger.info(f"[TOOL_NODE] Returning annotated_image in result, length: {len(state['annotated_image'])}")

        return result

    async def generate(self, state: State) -> Dict[str, Any]:
        """Generate AI response using the current model.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with new AI message
        """
        messages = convert_langgraph_messages_to_openai(state.get("messages", []))
        logger.debug({
            "message": "GRAPH: ENTERING NODE - generate",
            "chat_id": state.get("chat_id"),
            "iterations": state.get("iterations", 0),
            "current_model": self.current_model,
            "message_count": len(state.get("messages", []))
        })
        await self.stream_callback({'type': 'node_start', 'data': 'generate'})

        supports_tools = self.current_model in {"gpt-oss-20b", "gpt-oss-120b"}
        has_tools = supports_tools and self.openai_tools and len(self.openai_tools) > 0
        
        logger.debug({
            "message": "Tool calling debug info",
            "chat_id": state.get("chat_id"),
            "current_model": self.current_model,
            "supports_tools": supports_tools,
            "openai_tools_count": len(self.openai_tools) if self.openai_tools else 0,
            "openai_tools": self.openai_tools,
            "has_tools": has_tools
        })
        
        tool_params = {}
        if has_tools:
            tool_params = {
                "tools": self.openai_tools,
                "tool_choice": "auto"
            }
        
        stream = await self.model_client.chat.completions.create(
            model=self.current_model,
            messages=messages,
            temperature=0,
            top_p=1,
            stream=True,
            stream_options={"include_usage": True},
            **tool_params
        )

        llm_output_buffer, tool_calls_buffer, usage_info = await self._stream_response(stream, self.stream_callback)

        # Log token usage
        if usage_info:
            logger.info({
                "message": "TOKEN_USAGE",
                "chat_id": state.get("chat_id"),
                "model": self.current_model,
                "prompt_tokens": usage_info.get("prompt_tokens", 0),
                "completion_tokens": usage_info.get("completion_tokens", 0),
                "total_tokens": usage_info.get("total_tokens", 0)
            })
        tool_calls = self._format_tool_calls(tool_calls_buffer)
        raw_output = "".join(llm_output_buffer)
        
        logger.debug({
            "message": "Tool call generation results",
            "chat_id": state.get("chat_id"),
            "tool_calls_buffer": tool_calls_buffer,
            "formatted_tool_calls": tool_calls,
            "tool_calls_count": len(tool_calls),
            "raw_output_length": len(raw_output),
            "raw_output": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output
        })

        response = AIMessage(
            content=raw_output,
            **({"tool_calls": tool_calls} if tool_calls else {})
        )

        logger.debug({
            "message": "GRAPH: EXITING NODE - generate",
            "chat_id": state.get("chat_id"),
            "iterations": state.get("iterations", 0),
            "response_length": len(response.content) if response.content else 0,
            "tool_calls_generated": len(tool_calls),
            "tool_calls_names": [tc["name"] for tc in tool_calls] if tool_calls else [],
            "next_step": "→ should_continue decision"
        })
        await self.stream_callback({'type': 'node_end', 'data': 'generate'})
        return {"messages": state.get("messages", []) + [response]}

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for conversation flow.
        
        Returns:
            Compiled StateGraph with nodes and conditional edges
        """
        workflow = StateGraph(State)

        workflow.add_node("generate", self.generate)
        workflow.add_node("action", self.tool_node)
        workflow.add_edge(START, "generate")
        workflow.add_conditional_edges(
            "generate",
            self.should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "generate")

        return workflow.compile(checkpointer=memory)

    def _format_tool_calls(self, tool_calls_buffer: Dict[int, Dict[str, str]]) -> List[ToolCall]:
        """Parse streamed tool call buffer into ToolCall objects.
        
        Args:
            tool_calls_buffer: Buffer of streamed tool call data
            
        Returns:
            List of formatted ToolCall objects
        """
        if not tool_calls_buffer:
            return []

        tool_calls = []
        for i in sorted(tool_calls_buffer):
            item = tool_calls_buffer[i]
            try:
                parsed_args = json.loads(item["arguments"] or "{}")
            except json.JSONDecodeError:
                parsed_args = {}
                
            tool_calls.append(
                ToolCall(
                    name=item["name"],
                    args=parsed_args,
                    id=item["id"] or f"call_{i}",
                )
            )
        return tool_calls

    async def _stream_response(self, stream, stream_callback: StreamCallback) -> tuple[List[str], Dict[int, Dict[str, str]], Dict]:
        """Process streaming LLM response and extract content and tool calls.

        Args:
            stream: Async stream from LLM
            stream_callback: Callback for streaming events

        Returns:
            Tuple of (content_buffer, tool_calls_buffer, usage_info)
        """
        llm_output_buffer = []
        tool_calls_buffer = {}
        usage_info = {}
        saw_tool_finish = False

        async for chunk in stream:
            # Capture usage info from the final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = {
                    "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(chunk.usage, "completion_tokens", 0),
                    "total_tokens": getattr(chunk.usage, "total_tokens", 0)
                }

            for choice in getattr(chunk, "choices", []) or []:
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue

                content = getattr(delta, "content", None)
                if content:
                    await stream_callback({"type": "token", "data": content})
                    llm_output_buffer.append(content)
                for tc in getattr(delta, "tool_calls", []) or []:
                    idx = getattr(tc, "index", None)
                    if idx is None:
                        idx = 0 if not tool_calls_buffer else max(tool_calls_buffer) + 1
                    entry = tool_calls_buffer.setdefault(idx, {"id": None, "name": None, "arguments": ""})

                    if getattr(tc, "id", None):
                        entry["id"] = tc.id

                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            entry["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            entry["arguments"] += fn.arguments

                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "tool_calls":
                    saw_tool_finish = True
                    break

            if saw_tool_finish:
                break

        return llm_output_buffer, tool_calls_buffer, usage_info

    async def query(self, query_text: str, chat_id: str, image_data: str = None) -> AsyncIterator[Dict[str, Any]]:
        """Process user query and stream response tokens.
        
        Args:
            query_text: User's input text
            chat_id: Unique chat identifier
            
        Yields:
            Streaming events and tokens
        """
        logger.debug({
            "message": "GRAPH: STARTING EXECUTION",
            "chat_id": chat_id,
            "query": query_text[:100] + "..." if len(query_text) > 100 else query_text,
            "graph_flow": "START → generate → should_continue → action → generate → END"
        })

        config = {"configurable": {"thread_id": chat_id}}

        try:
            existing_messages = await self.conversation_store.get_messages(chat_id)

            # Build system prompt with image context awareness
            base_system_prompt = self.system_prompt

            # Get available images for this chat to make model context-aware
            available_images = await self.conversation_store.list_images_for_chat(chat_id)
            image_catalog_context = ""

            if available_images:
                image_list = []
                for img in available_images:
                    img_info = f"  - ID: {img['image_id']}"
                    if img.get('filename'):
                        img_info += f", Filename: {img['filename']}"
                    if img.get('description'):
                        img_info += f", Description: {img['description']}"
                    image_list.append(img_info)

                image_catalog_context = "\n\nAVAILABLE IMAGES IN THIS CHAT:\n"
                image_catalog_context += "You have access to the following images that were previously uploaded in this conversation:\n"
                image_catalog_context += "\n".join(image_list)
                image_catalog_context += "\n\nIMPORTANT - How to reference images:"
                image_catalog_context += "\n- When the user refers to an image by filename (e.g., 'the cat image', 'circuit_board.jpg'), find the matching image above and use its ID"
                image_catalog_context += "\n- When using image tools (explain_image, annotate_image), pass the image_id parameter with the correct ID from the list above"
                image_catalog_context += "\n- If the user's description matches a filename or description, use that image's ID"
                image_catalog_context += "\n- Example: If user says 'annotate the cat in my_cat.jpg' and you see 'ID: abc-123, Filename: my_cat.jpg', use image_id='abc-123'"

            if image_data:
                image_context = "\n\nCURRENT IMAGE: The user has uploaded a NEW image with their current message. You MUST use the explain_image tool to analyze it."
                system_prompt_with_context = base_system_prompt + image_catalog_context + image_context
                messages_to_process = [SystemMessage(content=system_prompt_with_context)]
            else:
                system_prompt_with_context = base_system_prompt + image_catalog_context
                messages_to_process = [SystemMessage(content=system_prompt_with_context)]

            if existing_messages:
                # Filter out SystemMessages and ensure ToolMessages have valid preceding AIMessage with tool_calls
                # Also strip image attachments from messages to reduce context size
                filtered_messages = []
                last_ai_had_tool_calls = False

                for msg in existing_messages:
                    if isinstance(msg, SystemMessage):
                        continue
                    elif isinstance(msg, AIMessage):
                        # Check if message has image attachments and strip them
                        if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs.get('attachments'):
                            attachments = msg.additional_kwargs.get('attachments', [])
                            image_attachments = [a for a in attachments if a.get('type') == 'image']
                            if image_attachments:
                                # Create a copy without the full image data, just note that an image was attached
                                content = msg.content
                                if content:
                                    content += "\n\n[This message included an annotated image that was displayed to the user]"
                                else:
                                    content = "[This message included an annotated image that was displayed to the user]"
                                # Create new AIMessage without attachments
                                cleaned_msg = AIMessage(
                                    content=content,
                                    **({"tool_calls": msg.tool_calls} if hasattr(msg, 'tool_calls') and msg.tool_calls else {})
                                )
                                filtered_messages.append(cleaned_msg)
                            else:
                                filtered_messages.append(msg)
                        else:
                            filtered_messages.append(msg)
                        last_ai_had_tool_calls = bool(hasattr(msg, 'tool_calls') and msg.tool_calls)
                    elif isinstance(msg, ToolMessage):
                        # Only include ToolMessage if preceding AIMessage had tool_calls
                        if last_ai_had_tool_calls:
                            filtered_messages.append(msg)
                        # Keep flag true for multiple ToolMessages from same AIMessage
                    else:
                        # HumanMessage or other types
                        filtered_messages.append(msg)
                        last_ai_had_tool_calls = False

                messages_to_process.extend(filtered_messages)

            messages_to_process.append(HumanMessage(content=query_text))

            config_obj = self.config_manager.read_config()

            initial_state = {
                "iterations": 0,
                "chat_id": chat_id,
                "messages": messages_to_process,
                "image_data": image_data if image_data else None,
                "process_image_used": False,
                "annotated_image": None
            }
            

            model_name = self.config_manager.get_selected_model()
            if self.current_model != model_name:
                self.set_current_model(model_name)

            logger.debug({
                "message": "GRAPH: LAUNCHING EXECUTION",
                "chat_id": chat_id,
                "initial_state": {
                    "iterations": initial_state["iterations"],
                    "message_count": len(initial_state["messages"]),
                }
            })

            self.last_state = None
            token_q: asyncio.Queue[Any] = asyncio.Queue()
            self.stream_callback = lambda event: self._queue_writer(event, token_q)
            runner = asyncio.create_task(self._run_graph(initial_state, config, chat_id, token_q))

            try:
                while True:
                    item = await token_q.get()
                    if item is SENTINEL:
                        break
                    yield item
            except Exception as stream_error:
                logger.error({"message": "Error in streaming", "error": str(stream_error)}, exc_info=True)
            finally:
                with contextlib.suppress(asyncio.CancelledError):
                    await runner

                logger.debug({
                    "message": "GRAPH: EXECUTION COMPLETED",
                    "chat_id": chat_id,
                    "final_iterations": self.last_state.get("iterations", 0) if self.last_state else 0
                })

        except Exception as e:
            logger.error({"message": "GRAPH: EXECUTION FAILED", "error": str(e), "chat_id": chat_id}, exc_info=True)
            yield {"type": "error", "data": f"Error performing query: {str(e)}"}


    async def _queue_writer(self, event: Dict[str, Any], token_q: asyncio.Queue) -> None:
        """Write events to the streaming queue.
        
        Args:
            event: Event data to queue
            token_q: Queue for streaming events
        """
        await token_q.put(event)

    async def _run_graph(self, initial_state: Dict[str, Any], config: Dict[str, Any], chat_id: str, token_q: asyncio.Queue) -> None:
        """Run the graph execution in background task.
        
        Args:
            initial_state: Starting state for graph
            config: LangGraph configuration
            chat_id: Chat identifier
            token_q: Queue for streaming events
        """
        try:
            async for final_state in self.graph.astream(
                initial_state,
                config=config,
                stream_mode="values",
                stream_writer=lambda event: self._queue_writer(event, token_q)
            ):
                self.last_state = final_state
        finally:
            try:
                if self.last_state and self.last_state.get("messages"):
                    final_msg = self.last_state["messages"][-1]
                    try:
                        logger.debug(f'Saving messages to conversation store for chat: {chat_id}')

                        # Store annotated image if present and attach reference to message
                        annotated_image_id = None
                        if self.last_state.get("annotated_image"):
                            import uuid
                            annotated_image_id = f"annotated-{uuid.uuid4()}"
                            await self.conversation_store.store_image_with_metadata(
                                image_id=annotated_image_id,
                                image_base64=self.last_state["annotated_image"],
                                chat_id=chat_id,
                                filename=f"annotated_image_{annotated_image_id[:8]}.png",
                                content_type="image/png",
                                persistent=True
                            )
                            logger.info(f"[ANNOTATE_IMAGE] Stored annotated image to database: {annotated_image_id}")

                            # Attach the image reference to the last AIMessage
                            messages = self.last_state["messages"]
                            if messages and hasattr(messages[-1], 'additional_kwargs'):
                                # Store image ID as attachment for persistence
                                if not messages[-1].additional_kwargs:
                                    messages[-1].additional_kwargs = {}
                                messages[-1].additional_kwargs["attachments"] = [{
                                    "type": "image",
                                    "image_id": annotated_image_id
                                }]

                        await self.conversation_store.save_messages(chat_id, self.last_state["messages"])
                    except Exception as save_err:
                        logger.warning({"message": "Failed to persist conversation", "chat_id": chat_id, "error": str(save_err)})

                    content = getattr(final_msg, "content", None)
                    logger.info(f"[FINAL_RESPONSE] Content length: {len(content) if content else 0}")
                    logger.info(f"[FINAL_RESPONSE] Has annotated_image in state: {bool(self.last_state.get('annotated_image'))}")

                    if content:
                        # Check if we have an annotated image to send separately
                        if self.last_state.get("annotated_image"):
                            annotated_img = self.last_state["annotated_image"]
                            logger.info(f"[FINAL_RESPONSE] Annotated image length: {len(annotated_img)}")
                            logger.info(f"[FINAL_RESPONSE] Annotated image starts with: {annotated_img[:50]}...")

                            # Strip any base64 image data from the text response
                            # Remove markdown image syntax with base64 data
                            cleaned_content = re.sub(
                                r'!\[([^\]]*)\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)',
                                '',
                                content
                            )
                            # Also remove any raw base64 data URLs
                            cleaned_content = re.sub(
                                r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
                                '',
                                cleaned_content
                            )
                            # Clean up any extra whitespace
                            cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content).strip()

                            logger.info(f"[FINAL_RESPONSE] Cleaned content: {cleaned_content}")

                            response_data = {
                                "type": "token",
                                "data": {
                                    "type": "final_response",
                                    "text": cleaned_content,
                                    "image": annotated_img
                                }
                            }
                            logger.info(f"[FINAL_RESPONSE] Sending response_data with nested type: final_response, text length: {len(cleaned_content)}, image length: {len(annotated_img)}")
                            await token_q.put(response_data)
                        else:
                            await token_q.put(content)
            finally:
                await token_q.put(SENTINEL)
