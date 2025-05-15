# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import json
import pytest

from camel.utils.token_counting import OpenAITokenCounter
from camel.types import ModelType


@pytest.fixture
def token_counter():
    return OpenAITokenCounter(ModelType.GPT_4)


@pytest.fixture
def message_with_tool_calls():
    return {
        "role": "assistant",
        "content": "I'll help you with that.",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "San Francisco", "unit": "celsius"})
                }
            }
        ]
    }


@pytest.fixture
def message_with_multiple_tool_calls():
    return {
        "role": "assistant",
        "content": "I'll help you with that.",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "San Francisco", "unit": "celsius"})
                }
            },
            {
                "id": "call_456",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": json.dumps({"timezone": "PST"})
                }
            }
        ]
    }


@pytest.fixture
def message_without_tool_calls():
    return {
        "role": "assistant",
        "content": "I'll help you with that."
    }


@pytest.fixture
def message_tool_response():
    return {
        "role": "tool",
        "content": "The weather in San Francisco is 15°C",
        "tool_call_id": "call_123"
    }


@pytest.fixture
def conversation(message_with_tool_calls, message_tool_response):
    return [
        {"role": "user", "content": "What's the weather in San Francisco?"},
        message_with_tool_calls,
        message_tool_response,
        {"role": "assistant", "content": "The weather in San Francisco is 15°C."}
    ]


@pytest.fixture
def multi_request_conversation():
    """Simulate a conversation with multiple LLM requests"""
    return [
        # First request
        {"role": "user", "content": "What's the weather in San Francisco and New York?"},
        {
            "role": "assistant",
            "content": "I'll check that for you.",
            "tool_calls": [
                {
                    "id": "call_sf",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "San Francisco", "unit": "celsius"})
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": "The weather in San Francisco is 15°C",
            "tool_call_id": "call_sf"
        },
        # Second request (continuing the conversation)
        {
            "role": "assistant",
            "content": "The weather in San Francisco is 15°C. Let me check New York for you.",
            "tool_calls": [
                {
                    "id": "call_ny",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "unit": "celsius"})
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": "The weather in New York is 10°C",
            "tool_call_id": "call_ny"
        },
        {
            "role": "assistant",
            "content": "The weather in San Francisco is 15°C and in New York is 10°C."
        }
    ]


def test_token_counting_with_tool_calls(
    token_counter, 
    message_with_tool_calls, 
    message_with_multiple_tool_calls,
    message_without_tool_calls,
    message_tool_response,
    conversation
):
    # Test single message with tool calls
    tokens_with_tool_calls = token_counter.count_tokens_from_messages([message_with_tool_calls])
    tokens_without_tool_calls = token_counter.count_tokens_from_messages([message_without_tool_calls])
    
    # Ensure tool calls add to the token count
    assert tokens_with_tool_calls > tokens_without_tool_calls
    
    # Test message with multiple tool calls
    tokens_with_multiple_tool_calls = token_counter.count_tokens_from_messages([message_with_multiple_tool_calls])
    
    # Ensure multiple tool calls add more tokens than a single tool call
    assert tokens_with_multiple_tool_calls > tokens_with_tool_calls
    
    # Test tool response message
    tokens_tool_response = token_counter.count_tokens_from_messages([message_tool_response])
    
    # Ensure tool response is counted correctly
    assert tokens_tool_response > 0
    
    # Test conversation with tool calls
    tokens_conversation = token_counter.count_tokens_from_messages(conversation)
    
    # The conversation token count should be greater than zero
    assert tokens_conversation > 0


def test_token_counting_with_multi_request(token_counter, multi_request_conversation):
    """Test token counting with a conversation that spans multiple LLM requests"""
    
    # Count tokens for the entire conversation
    total_tokens = token_counter.count_tokens_from_messages(multi_request_conversation)
    assert total_tokens > 0
    
    # Count tokens for each "request" segment
    first_request = multi_request_conversation[:3]  # User query, assistant with tool call, tool response
    second_request = multi_request_conversation[3:]  # Assistant with tool call, tool response, final response
    
    first_request_tokens = token_counter.count_tokens_from_messages(first_request)
    second_request_tokens = token_counter.count_tokens_from_messages(second_request)
    
    # Each segment should have tokens
    assert first_request_tokens > 0
    assert second_request_tokens > 0
    
    # The sum of segments should be less than or equal to the total
    # (due to potential overlapping tokens in the OpenAI token counting algorithm)
    assert total_tokens <= first_request_tokens + second_request_tokens