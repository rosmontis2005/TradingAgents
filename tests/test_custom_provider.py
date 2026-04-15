import unittest
from unittest.mock import patch

from tradingagents.llm_clients.factory import create_llm_client


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


if __name__ == "__main__":
    unittest.main()
