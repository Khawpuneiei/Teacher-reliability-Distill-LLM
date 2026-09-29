import unittest

from teacher_reliability.alignment import align_generated_tokens


class FakeOffsetTokenizer:
    all_special_ids = [99]

    def __init__(self, reencoded_ids, offsets):
        self.reencoded_ids = reencoded_ids
        self.offsets = offsets

    def decode(self, token_ids, skip_special_tokens, clean_up_tokenization_spaces):
        return "42"

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        return {
            "input_ids": self.reencoded_ids,
            "offset_mapping": self.offsets,
        }


class GeneratedTokenAlignmentTests(unittest.TestCase):
    def test_special_tokens_are_skipped_and_roundtrip_offsets_are_mapped_back(self):
        tokenizer = FakeOffsetTokenizer([10, 11], [(0, 1), (1, 2)])

        alignment = align_generated_tokens([10, 11, 99], tokenizer)

        self.assertEqual(alignment.text, "42")
        self.assertEqual(alignment.character_spans, [(0, 1), (1, 2), None])
        self.assertEqual(alignment.status, "aligned")

    def test_retokenization_mismatch_does_not_guess_character_offsets(self):
        tokenizer = FakeOffsetTokenizer([10], [(0, 2)])

        alignment = align_generated_tokens([10, 11], tokenizer)

        self.assertEqual(alignment.text, "42")
        self.assertIsNone(alignment.character_spans)
        self.assertEqual(alignment.status, "token_roundtrip_mismatch")

    def test_tokenizer_without_offset_mapping_marks_alignment_unavailable(self):
        class NoOffsetsTokenizer(FakeOffsetTokenizer):
            def __call__(self, text, add_special_tokens, return_offsets_mapping):
                raise NotImplementedError

        alignment = align_generated_tokens([10, 11], NoOffsetsTokenizer([10, 11], []))

        self.assertIsNone(alignment.character_spans)
        self.assertEqual(alignment.status, "offsets_unavailable")


if __name__ == "__main__":
    unittest.main()
