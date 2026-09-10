#!/usr/bin/env python3
"""
Async LLM Handler for local LLMs (LM Studio, Ollama, etc.)
Handles communication with locally running language models via OpenAI-compatible APIs.
Supports callbacks for response handling and streaming.
Enhanced with conversation history management.
"""
import aiohttp
import asyncio
import json
import os
from typing import Optional, Dict, Any, Callable, Union, List
from dataclasses import dataclass
from datetime import datetime
import re


COURSE_API_TOKEN = "I-promise-I-will-watch-Twin-Peaks"


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    user_message: str
    assistant_response: str
    timestamp: datetime


@dataclass
class LLMResponse:
    """Response container for LLM interactions."""
    content: Optional[str] = None
    error: Optional[str] = None
    success: bool = False
    raw_response: Optional[Dict[str, Any]] = None


class AsyncLLMHandler:
    """
    Async handler for local LLM communication via OpenAI-compatible APIs.
    Works with LM Studio, Ollama, and other local LLM servers.
    Supports callbacks for response handling and streaming.
    Enhanced with conversation history management.
    """
    
    def __init__(self, 
                 base_url: str = "http://127.0.0.1:11434/v1",
                 model_name: str = "nvidia/nemotron-3-nano-4b",#"nemotron-3-nano:4b",#"qwen3-vl:4b",
                 max_history_turns: int = 10,
                 timeout: int = 30,
                 api_token: Optional[str] = None):
        """
        Initialize the async LLM handler.
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.max_history_turns = max_history_turns
        self.timeout = timeout
        self.api_token = (
            api_token
            or os.getenv("AFFECTALM_API_TOKEN")
            or COURSE_API_TOKEN
        )
        self.chat_endpoint = f"{self.base_url}/chat/completions"
        
        # Store conversation history in memory
        self.conversation_history: List[ConversationTurn] = []
        
        # Instructions for the LLM
        self.instructions = (
            "You are a friendly, funny and social robot that loves having conversations! "
            "Keep your responses short and friendly. Use a conversational tone and be engaging. "
            'Default to replies under ~2 sentences unless the user asks for detail.'
            "Your task is to be a barista. You will take peoples' orders and provide good customer service"
            "The menu consists of Tea and Filter coffee (12 crowns), "
            "Cafe Latte (25 crowns), Cappucino (22 crowns)"
            "A standard transaction is as follows: Greet the customer, they order a beverage, they have to pay, give them their order"
            "Do not use any emojis"
            "You are the barista so you are the one in charge of making sure that they pay and provide the drinks"
        )
        
        print(f"AsyncLLMHandler initialized for {self.base_url}")
        print(f"Model: {self.model_name}")
        print(f"Max history turns: {self.max_history_turns}")
        print(f"Instructions: {self.instructions[:100]}...")

    def _request_headers(self) -> Dict[str, str]:
        """Build request headers without storing credentials in source code."""
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = self.api_token
        return headers
    
    def set_instructions(self, instructions: str):
        """Update the LLM instructions."""
        self.instructions = instructions
        print(f"Instructions updated: {instructions[:100]}...")
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        print("🗑️ Conversation history cleared")
    
    def get_history_summary(self) -> str:
        """Get a summary of the conversation history."""
        if not self.conversation_history:
            return "No conversation history"
        
        summary = f"Conversation history ({len(self.conversation_history)} turns):\n"
        for i, turn in enumerate(self.conversation_history[-5:], 1):  # Show last 5 turns
            timestamp = turn.timestamp.strftime("%H:%M")
            summary += f"{i}. [{timestamp}] User: {turn.user_message[:50]}...\n"
            summary += f"   Assistant: {turn.assistant_response[:50]}...\n"
        return summary

    # (kept for debugging reference, but not used any more)
    def _build_prompt_with_history(self, user_text: str) -> str:
        prompt_parts = []
        prompt_parts.append(f"INSTRUCTIONS: {self.instructions}")
        if self.conversation_history:
            prompt_parts.append("CONVERSATION HISTORY:")
            for turn in self.conversation_history:
                prompt_parts.append(f"Human: {turn.user_message}")
                prompt_parts.append(f"Robot: {turn.assistant_response}")
        prompt_parts.append(f"Human: {user_text}")
        prompt_parts.append("Output only a short answer to the last human response")
        return "\n".join(prompt_parts)

    # Build messages using chat roles.
    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        """
        Build a role-aware messages array for /chat/completions.
        """
        messages: List[Dict[str, str]] = []

        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})

        for turn in self.conversation_history[-self.max_history_turns:]:
            if turn.user_message:
                messages.append({"role": "user", "content": turn.user_message})
            if turn.assistant_response:
                messages.append({"role": "assistant", "content": turn.assistant_response})

        messages.append({"role": "user", "content": user_text})
        return messages

    # Remove leaked banners and role prefixes.
    def _sanitize_assistant(self, text: str) -> str:
        t = (text or "").strip()
        t = re.sub(r"(?i)^\s*(assistant|robot)\s*:\s*", "", t)
        t = re.sub(r"(?i)^\s*(instructions|system|conversation history)\s*:.*?\n", "", t)
        t = re.split(r"\n(?:Human|User|Robot|Assistant)\s*:\s*", t)[0]
        return t.strip()
    
    def _add_to_history(self, user_message: str, assistant_response: str):
        """
        Add a conversation turn to the history.
        """
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.now()
        )
        
        self.conversation_history.append(turn)
        
        # Trim history if it exceeds max turns
        if len(self.conversation_history) > self.max_history_turns:
            removed = self.conversation_history.pop(0)
            print(f"📝 History trimmed (removed turn from {removed.timestamp.strftime('%H:%M')})")
        
        print(f"📝 Added to history (total: {len(self.conversation_history)} turns)")
    
    async def send_prompt(self, 
                         user_text: str, 
                         callback: Optional[Callable[[LLMResponse], None]] = None,
                         add_to_history: bool = True) -> Optional[str]:
        """
        Send a prompt to the LLM and return the response.
        """
        if not user_text or not user_text.strip():
            error_msg = "Warning: Empty user text provided"
            print(error_msg)
            return None
        
        try:
            # Build well-formed chat messages with roles
            messages = self._build_messages(user_text.strip())
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 600,
                "reasoning_effort": "none",
                "stream": False,
                "stop": ["\nUser:", "\nHuman:", "\nAssistant:", "\nRobot:", "User:", "Human:", "Assistant:", "Robot:"]
            }
            
            print(f"🤖 Sending to LLM: '{user_text}'")
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.chat_endpoint,
                    json=payload,
                    headers=self._request_headers()
                ) as response:
                    
                    response.raise_for_status()
                    
                    response_data = await response.json()
                    
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        assistant_message = response_data["choices"][0]["message"]["content"]
                        assistant_message = self._sanitize_assistant(assistant_message)
                        
                        print(f"✅ LLM Response: '{assistant_message}'")
                        
                        if add_to_history:
                            self._add_to_history(user_text.strip(), assistant_message)
                        
                        if callback:
                            callback(LLMResponse(
                                content=assistant_message,
                                success=True,
                                raw_response=response_data
                            ))
                        
                        return assistant_message
                    else:
                        error_msg = "No response content in LLM response"
                        print(f"❌ {error_msg}")
                        print(f"Response data: {response_data}")
                        
                        if callback:
                            callback(LLMResponse(
                                error=error_msg,
                                success=False,
                                raw_response=response_data
                            ))
                        return None
                        
        except aiohttp.ClientConnectorError:
            error_msg = f"Connection error: Could not connect to {self.base_url}"
            print(f"❌ {error_msg}")
            print("Make sure your local LLM server is running!")
            if callback:
                callback(LLMResponse(error=error_msg, success=False))
            return None
            
        except asyncio.TimeoutError:
            error_msg = f"Timeout: Request took longer than {self.timeout} seconds"
            print(f"❌ {error_msg}")
            if callback:
                callback(LLMResponse(error=error_msg, success=False))
            return None
            
        except aiohttp.ClientResponseError as e:
            error_msg = f"HTTP Error: {e}"
            print(f"❌ {error_msg}")
            if callback:
                callback(LLMResponse(error=error_msg, success=False))
            return None
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {e}"
            print(f"❌ {error_msg}")
            if callback:
                callback(LLMResponse(error=error_msg, success=False))
            return None
            
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(f"❌ {error_msg}")
            if callback:
                callback(LLMResponse(error=error_msg, success=False))
            return None
    
    async def send_prompt_streaming(self, 
                                   user_text: str,
                                   callback: Callable[[str], None],
                                   final_callback: Optional[Callable[[LLMResponse], None]] = None,
                                   add_to_history: bool = True) -> Optional[str]:
        """
        Send a prompt to the LLM with streaming response.
        """
        if not user_text or not user_text.strip():
            error_msg = "Warning: Empty user text provided"
            print(error_msg)
            if final_callback:
                final_callback(LLMResponse(error=error_msg, success=False))
            return None
        
        try:
            # Build well-formed chat messages with roles
            messages = self._build_messages(user_text.strip())
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 600,
                "reasoning_effort": "none",
                "stream": True,
                "stop": ["\nUser:", "\nHuman:", "\nAssistant:", "\nRobot:", "User:", "Human:", "Assistant:", "Robot:"]
            }
            
            print(f"🤖 Streaming from LLM: '{user_text[:50]}...'")
            
            complete_response = ""
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.chat_endpoint,
                    json=payload,
                    headers=self._request_headers()
                ) as response:
                    
                    response.raise_for_status()
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            data_str = line[6:]
                            
                            if data_str == '[DONE]':
                                break
                                
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        chunk = delta['content']
                                        complete_response += chunk
                                        callback(chunk)
                            except json.JSONDecodeError:
                                continue  # Skip invalid JSON lines
            
                    complete_response = self._sanitize_assistant(complete_response)
                    
                    print(f"✅ Complete streaming response received")
                    
                    if add_to_history and complete_response:
                        self._add_to_history(user_text.strip(), complete_response)
                    
                    if final_callback:
                        final_callback(LLMResponse(
                            content=complete_response,
                            success=True
                        ))
                    
                    return complete_response
                        
        except Exception as e:
            error_msg = f"Streaming error: {e}"
            print(f"❌ {error_msg}")
            if final_callback:
                final_callback(LLMResponse(error=error_msg, success=False))
            return None
    
    async def test_connection(self) -> bool:
        """
        Test the connection to the LLM server.
        """
        print(f"Testing connection to {self.base_url}...")
        
        try:
            test_response = await self.send_prompt("Hello, can you hear me?", add_to_history=False)
            if test_response:
                print("✅ LLM connection test successful!")
                return True
            else:
                print("❌ LLM connection test failed - no response")
                return False
                
        except Exception as e:
            print(f"❌ LLM connection test failed: {e}")
            return False
    
    async def get_models(self) -> Optional[list]:
        """
        Get list of available models from the server.
        """
        try:
            models_endpoint = f"{self.base_url}/models"
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(models_endpoint) as response:
                    response.raise_for_status()
                    
                    data = await response.json()
                    if "data" in data:
                        models = [model["id"] for model in data["data"]]
                        print(f"Available models: {models}")
                        return models
                    else:
                        print("No models data in response")
                        return None
                        
        except Exception as e:
            print(f"Error getting models: {e}")
            return None


# Example usage and testing
async def example_callback(response: LLMResponse):
    """Example callback function."""
    if response.success:
        print(f"📨 Callback received successful response: {response.content[:50]}...")
    else:
        print(f"📨 Callback received error: {response.error}")


async def example_streaming_callback(chunk: str):
    """Example streaming callback function."""
    print(f"📡 Stream chunk: {chunk}", end='', flush=True)


async def main():
    """Test the async LLM handler with conversation history."""
    print("Testing Enhanced Async LLM Handler with Conversation History...")
    
    # Test with LM Studio (default)
    print("\n=== Testing LM Studio (localhost:11434) ===")
    lm_studio = AsyncLLMHandler(
        base_url="https://affectalm.fam-f.dk/v1",
        max_history_turns=5
    )
    
    if await lm_studio.test_connection():
        print("\n--- Testing conversation with history ---")
        response1 = await lm_studio.send_prompt("Hi there! What's your name?")
        print(f"Response: {response1}")
    
    print("\ testing complete!")


if __name__ == "__main__":
    # Install required dependency:
    # pip install aiohttp
    
    print("Usage examples:")
    print("# Basic async usage with history:")
    print("llm = AsyncLLMHandler('http://localhost:11434/v1', max_history_turns=10)")
    print("response = await llm.send_prompt('Hello!')")
    print("response = await llm.send_prompt('Do you remember what I said?')")
    print("\n# Clear conversation history:")
    print("llm.clear_history()")
    print("\n# View history summary:")
    print("print(llm.get_history_summary())")
    
    # Run the test
    asyncio.run(main())
