import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.openai_client import DeepSeekCompatibleChatOpenAI


class CustomProviderTests(unittest.TestCase):
    @patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
    def test_custom_provider_uses_supplied_base_url_and_api_key(self, mock_chat):
        client = create_llm_client(
            "custom",
            "relay-model",
            base_url="https://relay.example.com/v1",
            api_key="relay-key",
        )

        client.get_llm()

        call_kwargs = mock_chat.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "relay-model")
        self.assertEqual(call_kwargs["base_url"], "https://relay.example.com/v1")
        self.assertEqual(call_kwargs["api_key"], "relay-key")
        self.assertNotIn("use_responses_api", call_kwargs)

    def test_custom_provider_requires_explicit_auth(self):
        client = create_llm_client(
            "custom",
            "relay-model",
            base_url="https://relay.example.com/v1",
        )

        with patch.dict(
            "os.environ",
            {"CUSTOM_OPENAI_API_KEY": "", "CUSTOM_OPENAI_BASE_URL": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "requires llm_api_key"):
                client.get_llm()

    def test_custom_provider_ignores_unsupported_all_proxy_scheme(self):
        client = create_llm_client(
            "custom",
            "relay-model",
            base_url="https://relay.example.com/v1",
            api_key="relay-key",
        )

        with patch.dict(
            "os.environ",
            {
                "ALL_PROXY": "socks://127.0.0.1:7897/",
                "all_proxy": "socks://127.0.0.1:7897/",
                "HTTPS_PROXY": "http://127.0.0.1:7897/",
            },
            clear=False,
        ):
            client.get_llm()

        self.assertEqual(client.provider, "custom")

    @patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
    def test_custom_provider_can_ignore_proxy_environment(self, mock_chat):
        client = create_llm_client(
            "custom",
            "relay-model",
            base_url="https://relay.example.com/v1",
            api_key="relay-key",
            trust_env=False,
        )

        client.get_llm()

        call_kwargs = mock_chat.call_args.kwargs
        self.assertFalse(call_kwargs["http_client"].trust_env)
        self.assertFalse(call_kwargs["http_async_client"].trust_env)

    @patch("tradingagents.llm_clients.openai_client.DeepSeekCompatibleChatOpenAI")
    def test_custom_provider_uses_deepseek_compat_for_deepseek_base_url(
        self, mock_chat
    ):
        client = create_llm_client(
            "custom",
            "deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key="deepseek-key",
        )

        client.get_llm()

        call_kwargs = mock_chat.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(call_kwargs["base_url"], "https://api.deepseek.com")
        self.assertEqual(call_kwargs["api_key"], "deepseek-key")

    def test_deepseek_compat_preserves_response_reasoning_content(self):
        with _without_proxy_env():
            llm = DeepSeekCompatibleChatOpenAI(
                model="deepseek-v4-flash",
                api_key="deepseek-key",
                base_url="https://api.deepseek.com",
            )
        response = {
            "id": "chatcmpl-test",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "need stock data before indicators",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_stock_data",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                }
            ],
        }

        result = llm._create_chat_result(response)

        message = result.generations[0].message
        self.assertEqual(
            message.additional_kwargs["reasoning_content"],
            "need stock data before indicators",
        )
        self.assertEqual(message.tool_calls[0]["name"], "get_stock_data")

    def test_deepseek_compat_passes_reasoning_content_back_in_context(self):
        with _without_proxy_env():
            llm = DeepSeekCompatibleChatOpenAI(
                model="deepseek-v4-flash",
                api_key="deepseek-key",
                base_url="https://api.deepseek.com",
            )
        messages = [
            HumanMessage(content="Analyze AAPL."),
            AIMessage(
                content="",
                additional_kwargs={
                    "reasoning_content": "need stock data before indicators"
                },
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_stock_data",
                        "args": {},
                    }
                ],
            ),
            ToolMessage(content="date,open,close", tool_call_id="call_1"),
        ]

        payload = llm._get_request_payload(messages)

        assistant_message = payload["messages"][1]
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(
            assistant_message["reasoning_content"],
            "need stock data before indicators",
        )
        self.assertEqual(
            assistant_message["tool_calls"][0]["function"]["name"],
            "get_stock_data",
        )


def _without_proxy_env():
    return patch.dict(
        "os.environ",
        {
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "http_proxy": "",
            "https_proxy": "",
            "all_proxy": "",
        },
        clear=False,
    )


if __name__ == "__main__":
    unittest.main()
