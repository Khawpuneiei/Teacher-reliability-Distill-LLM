import unittest

from teacher_reliability.hub_data import MATH_CONFIGS, DATASET_REPOSITORIES, load_examples
from teacher_reliability.selection import RunProfile, get_profile


class HubDataLoadingTests(unittest.TestCase):
    def test_loads_test_splits_at_resolved_commits_and_preserves_all_math_subjects(self):
        revision_calls = []
        load_calls = []
        revision_by_repo = {
            DATASET_REPOSITORIES["gsm8k"]: "1" * 40,
            DATASET_REPOSITORIES["math"]: "2" * 40,
            DATASET_REPOSITORIES["gsm_plus"]: "3" * 40,
        }

        def resolve_revision(repository):
            revision_calls.append(repository)
            return revision_by_repo[repository]

        fixtures = {
            (DATASET_REPOSITORIES["gsm8k"], "main"): [
                {"question": "How many?", "answer": "#### 4"}
            ],
            (DATASET_REPOSITORIES["gsm_plus"], None): [
                {
                    "question": "A perturbed problem?",
                    "solution": "The answer is 5. #### 5",
                    "answer": "5",
                    "perturbation_type": "numerical substitution",
                    "seed_question": "A source problem?",
                }
            ],
        }
        for subject in MATH_CONFIGS:
            fixtures[(DATASET_REPOSITORIES["math"], subject)] = [
                {"problem": f"A {subject} problem?", "solution": r"\boxed{6}"}
            ]

        def load_dataset(repository, config, split, revision):
            load_calls.append((repository, config, split, revision))
            return fixtures[(repository, config)]

        examples, revisions = load_examples(
            get_profile("smoke"),
            seed=11,
            revision_resolver=resolve_revision,
            dataset_loader=load_dataset,
        )

        self.assertEqual(set(revision_calls), set(DATASET_REPOSITORIES.values()))
        self.assertEqual(len(revision_calls), 3)
        self.assertEqual(len(load_calls), 2 + len(MATH_CONFIGS))
        self.assertTrue(all(call[2] == "test" for call in load_calls))
        self.assertTrue(all(call[3] == revisions[call[0]] for call in load_calls))
        self.assertEqual(
            {row.source_config for row in examples if row.dataset == "math"},
            set(MATH_CONFIGS),
        )
        self.assertEqual(
            {row.source_revision for row in examples if row.dataset == "gsm_plus"},
            {revisions[DATASET_REPOSITORIES["gsm_plus"]]},
        )

    def test_loader_does_not_accept_a_non_commit_revision(self):
        def invalid_revision(_repository):
            return "main"

        with self.assertRaises(ValueError):
            load_examples(
                get_profile("smoke"),
                seed=11,
                revision_resolver=invalid_revision,
                dataset_loader=lambda **_kwargs: [],
            )

    def test_loader_rejects_each_empty_requested_split(self):
        requests = [
            ("openai/gsm8k", "main"),
            *[("EleutherAI/hendrycks_math", subject) for subject in MATH_CONFIGS],
            ("qintongli/GSM-Plus", None),
        ]
        for empty_source in requests:
            with self.subTest(empty_source=empty_source):
                def load_dataset(repository, config, split, revision):
                    if (repository, config) == empty_source:
                        return iter(())
                    if repository == "EleutherAI/hendrycks_math":
                        return iter([{"problem": "How many?", "solution": r"\boxed{4}"}])
                    return iter([{"question": "How many?", "answer": "#### 4"}])

                with self.assertRaisesRegex(ValueError, "empty.*" + empty_source[0]):
                    load_examples(
                        RunProfile("pool", None, None, None, False, "test", 8, 1024), seed=11,
                        revision_resolver=lambda _repository: "1" * 40,
                        dataset_loader=load_dataset,
                    )


if __name__ == "__main__":
    unittest.main()
