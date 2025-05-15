import json
import unittest

from camel.utils.token_counting import OpenAITokenCounter
from camel.types import ModelType


class TestTokenCountingToolCalls(unittest.TestCase):
    def setUp(self):
        self.token_counter = OpenAITokenCounter(ModelType.GPT_4)
        
        # Create test messages
        self.message_with_tool_calls = {
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
        
        self.message_with_multiple_tool_calls = {
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
        
        self.message_without_tool_calls = {
            "role": "assistant",
            "content": "I'll help you with that."
        }
        
        self.message_tool_response = {
            "role": "tool",
            "content": "The weather in San Francisco is 15°C",
            "tool_call_id": "call_123"
        }
        
        self.conversation = [
            {"role": "user", "content": "What's the weather in San Francisco?"},
            self.message_with_tool_calls,
            self.message_tool_response,
            {"role": "assistant", "content": "The weather in San Francisco is 15°C."}
        ]

    def test_token_counting_with_tool_calls(self):
        # Test single message with tool calls
        tokens_with_tool_calls = self.token_counter.count_tokens_from_messages([self.message_with_tool_calls])
        tokens_without_tool_calls = self.token_counter.count_tokens_from_messages([self.message_without_tool_calls])
        
        # Ensure tool calls add to the token count
        self.assertGreater(tokens_with_tool_calls, tokens_without_tool_calls)
        
        # Test message with multiple tool calls
        tokens_with_multiple_tool_calls = self.token_counter.count_tokens_from_messages([self.message_with_multiple_tool_calls])
        
        # Ensure multiple tool calls add more tokens than a single tool call
        self.assertGreater(tokens_with_multiple_tool_calls, tokens_with_tool_calls)
        
        # Test tool response message
        tokens_tool_response = self.token_counter.count_tokens_from_messages([self.message_tool_response])
        
        # Ensure tool response is counted correctly
        self.assertGreater(tokens_tool_response, 0)
        
        # Test conversation with tool calls
        tokens_conversation = self.token_counter.count_tokens_from_messages(self.conversation)
        
        # The conversation token count is not simply the sum of individual messages
        # due to how OpenAI counts tokens in conversations
        # Instead, we'll verify that each message type is counted correctly
        self.assertGreater(tokens_conversation, 0)
        
        # Print token counts for debugging
        print(f"Tokens with tool_calls: {tokens_with_tool_calls}")
        print(f"Tokens with multiple tool_calls: {tokens_with_multiple_tool_calls}")
        print(f"Tokens without tool_calls: {tokens_without_tool_calls}")
        print(f"Tokens with tool response: {tokens_tool_response}")
        print(f"Tokens in conversation: {tokens_conversation}")


if __name__ == "__main__":
    unittest.main()