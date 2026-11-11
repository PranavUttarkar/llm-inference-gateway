import unittest
from benchmarks.make_workload import extend_prompt


class WorkloadTests(unittest.TestCase):
    def test_counts_include_chat_template(self):
        class CharacterTokenizer:
            def encode(self, text, **kwargs):
                return list(text)
            def decode(self, tokens, **kwargs):
                return ''.join(tokens)
            def apply_chat_template(self, messages, **kwargs):
                return ['header'] * 5 + list(messages[0]['content'])
        text, count = extend_prompt(CharacterTokenizer(), 'testing', 64)
        self.assertEqual(count, 64)
        self.assertEqual(len(text), 59)

    def test_impossible_target_refused(self):
        class HugeTemplate:
            def encode(self, text, **kwargs):
                return list(text)
            def decode(self, tokens, **kwargs):
                return ''.join(tokens)
            def apply_chat_template(self, messages, **kwargs):
                return list(range(100))
        with self.assertRaises(ValueError):
            extend_prompt(HugeTemplate(), 'testing', 64)
