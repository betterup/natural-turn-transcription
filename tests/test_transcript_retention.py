"""Regression checks for AWS parsing and the baseline/Backbiter/Cliffhanger export path.

Run from the repository root with: python -m unittest discover -s tests -v
The tokenizer-only checks do not require downloading a spaCy language model.
"""

from collections import Counter
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import spacy

from transcription import text
from transcription.transcript_config import Transcript, TranscriptConfig, baseline, backbiter, cliffhanger, natural_turn


def load_formatter():
    """Load the notebook's actual class without executing its data-processing cells."""
    notebook_path = Path(__file__).resolve().parents[1] / "json-to-natural.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "".join(cell["source"]).startswith("class TranscriptFormatter(")
    )
    namespace = {
        "json": json,
        "pd": pd,
        "Transcript": Transcript,
        "collapse_transcript_turns": text.collapse_transcript_turns,
    }
    exec(compile(source, str(notebook_path), "exec"), namespace)
    return namespace["TranscriptFormatter"]


def word(content, start, stop, confidence="0.99"):
    return {
        "type": "pronunciation",
        "start_time": str(start),
        "end_time": str(stop),
        "alternatives": [{"content": content, "confidence": confidence}],
    }


def punctuation(content):
    return {
        "type": "punctuation",
        "alternatives": [{"content": content, "confidence": "0.0"}],
    }


def utterance_frame(utterances):
    return pd.DataFrame(
        {
            "speaker": ["A" if i % 2 == 0 else "B" for i in range(len(utterances))],
            "start": [float(i) for i in range(len(utterances))],
            "stop": [i + 0.5 for i in range(len(utterances))],
            "utterance": utterances,
            "confidence": [[0.99] for _ in utterances],
        }
    ).rename_axis("turn_id")


def all_output_words(df):
    utterances = df.utterance.tolist()
    if "backchannel" in df:
        utterances += df.backchannel.dropna().tolist()
    return Counter(" ".join(utterances).split())


class TranscriptRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.formatter_class = load_formatter()
        cls.nlp = spacy.blank("en")

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        (self.root / "aws-2024-candor-raw-api").mkdir()
        (self.root / "candor_metadata_files").mkdir()
        (self.root / "output").mkdir()
        self.formatter = self.formatter_class("test", self.root, self.root / "output")
        metadata = {"speakers": [{"channel": "L", "user_id": "A"}, {"channel": "R", "user_id": "B"}]}
        Path(self.formatter.metadata_path).write_text(json.dumps(metadata))
        model_patch = patch.object(text, "load_spacy_model", return_value=self.nlp)
        model_patch.start()
        self.addCleanup(model_patch.stop)

    def parse(self, left, right):
        payload = {"results": {"channel_labels": {"channels": [{"items": left}, {"items": right}]}}}
        Path(self.formatter.stt_path).write_text(json.dumps(payload))
        return self.formatter.construct_conversation_transcript()

    def test_final_run_retains_text_punctuation_confidence_and_times(self):
        parsed = self.parse(
            [word("Hello", 0.1, 0.3), punctuation("."), word("Take", 2.0, 2.2), word("care", 2.3, 2.6, "0.72"), punctuation("!")],
            [word("Goodbye", 1.0, 1.5, "0.95"), punctuation(".")],
        )["transcript"]
        self.assertEqual([row["utterance"] for row in parsed], ["Hello.", "Goodbye.", "Take care!"])
        self.assertEqual([row["speaker"] for row in parsed], ["L", "R", "L"])
        self.assertEqual([row["turn_id"] for row in parsed], [0, 1, 2])
        self.assertEqual(parsed[-1]["confidence"], [0.99, 0.72, 0.0])
        self.assertEqual(float(parsed[-1]["start"]), 2.0)
        self.assertEqual(float(parsed[-1]["stop"]), 2.6)
        self.assertEqual(sum(len(row["confidence"]) for row in parsed), 7)

    def test_single_speaker_and_single_token_are_retained(self):
        for tokens, expected in [
            ([word("Hello", 0.0, 0.5)], "Hello"),
            ([word("Hello", 0.0, 0.5), word("there", 0.6, 0.9), punctuation(".")], "Hello there."),
        ]:
            with self.subTest(expected=expected):
                for left, right in [(tokens, []), ([], tokens)]:
                    parsed = self.parse(left, right)["transcript"]
                    self.assertEqual(len(parsed), 1)
                    self.assertEqual(parsed[0]["utterance"], expected)
                    self.assertEqual(len(parsed[0]["confidence"]), len(tokens))

    def test_empty_input_does_not_create_a_phantom_turn(self):
        self.assertEqual(self.parse([], [])["transcript"], [])

    def test_overlap_keeps_both_speakers_and_the_terminal_run(self):
        parsed = self.parse(
            [word("First", 0.0, 1.0), word("last", 1.2, 1.5), punctuation(".")],
            [word("Yeah", 0.5, 0.7), punctuation(".")],
        )["transcript"]
        self.assertEqual([row["utterance"] for row in parsed], ["First", "Yeah.", "last."])
        self.assertEqual(float(parsed[1]["start"]), 0.5)
        self.assertEqual(float(parsed[0]["stop"]), 1.0)

    def test_opening_cue_stays_primary_and_second_candidate_is_unchanged(self):
        frame = utterance_frame(["Yeah.", "Yep.", "Hello there."])
        result = text._unstack_backchannels(frame, backbiter)
        self.assertEqual(result.utterance.tolist(), frame.utterance.tolist())
        self.assertEqual(result.backchannel_count.tolist(), [0, 0, 0])
        self.assertEqual(all_output_words(result), all_output_words(frame))

    def test_single_opening_cue_is_retained(self):
        frame = utterance_frame(["Yeah."])
        result = text._unstack_backchannels(frame, backbiter)
        self.assertEqual(result.utterance.tolist(), ["Yeah."])
        self.assertEqual(result.backchannel_count.tolist(), [0])

    def test_consecutive_candidates_keep_historical_suppression_and_all_text(self):
        frame = utterance_frame(["Hello there.", "Yeah.", "Yep.", "Sure.", "Good day.", "Right."])
        result = text._unstack_backchannels(frame, backbiter)
        # The first candidate in the middle run is pivoted; the other candidates
        # remain main turns, as in the original pandas 2 implementation.
        self.assertEqual(result.utterance.tolist(), ["Hello there.", "Yep.", "Sure.", "Good day."])
        self.assertEqual(result.backchannel.dropna().tolist(), ["Yeah.", "Right."])
        self.assertEqual(result.backchannel_count.tolist(), [1, 0, 0, 1])
        self.assertEqual(all_output_words(result), all_output_words(frame))
        self.assertEqual(result.iloc[0].backchannel_speaker, "B")
        self.assertEqual(result.iloc[0].backchannel_start, 1.0)
        self.assertEqual(result.iloc[0].backchannel_stop, 1.5)

    def test_baseline_and_backbiter_end_to_end_and_save_without_edit_history(self):
        parsed = self.parse(
            [word("Hello", 0.0, 0.3), word("there", 0.4, 0.6), punctuation("."), word("Goodbye", 2.0, 2.5), punctuation(".")],
            [word("Yeah", 1.0, 1.3), punctuation(".")],
        )
        frame = self.formatter.json_to_baseline_df(parsed)
        for label, config in [("audiophile", baseline), ("backbiter", backbiter)]:
            with self.subTest(preset=label):
                self.assertFalse(config.collapse_short_turns)
                transcript = self.formatter.baseline_to_natural_turn(frame, config)
                self.assertIsNone(transcript.edit_history)
                self.assertEqual(all_output_words(transcript.transcript), all_output_words(frame))
                self.formatter.save_transcripts_to_local(transcript, label)
                self.assertTrue((self.root / "output" / f"test_{label}.csv").exists())
                self.assertFalse((self.root / "output" / f"test_{label}_edit-history.csv").exists())
        saved_baseline = pd.read_csv(self.root / "output" / "test_baseline.csv")
        self.assertEqual(saved_baseline.utterance.tolist(), ["Hello there.", "Yeah.", "Goodbye."])
        self.assertNotIn("n_words", saved_baseline.columns)
        saved_audiophile = pd.read_csv(self.root / "output" / "test_audiophile.csv")
        self.assertIn("n_words", saved_audiophile.columns)
        self.assertEqual(saved_audiophile.utterance.tolist(), saved_baseline.utterance.tolist())
        saved_backbiter = pd.read_csv(self.root / "output" / "test_backbiter.csv")
        self.assertEqual(saved_backbiter.utterance.tolist(), ["Hello there. Goodbye."])
        self.assertEqual(saved_backbiter.backchannel.tolist(), ["Yeah."])

    def test_saver_still_writes_a_supplied_edit_history(self):
        frame = utterance_frame(["Hello there."])
        transcript = Transcript(frame, natural_turn)
        transcript.set_transcript(frame)
        transcript.set_edit_history(pd.DataFrame({"action_taken": ["example edit"]}))
        self.formatter.save_transcripts_to_local(transcript, "with-edits")
        history = pd.read_csv(self.root / "output" / "test_with-edits_edit-history.csv")
        self.assertEqual(history.action_taken.tolist(), ["example edit"])

    def test_saver_still_writes_an_empty_v2_edit_history(self):
        frame = utterance_frame(["Hello there."])
        transcript = Transcript(frame, natural_turn)
        transcript.set_transcript(frame)
        transcript.set_edit_history(pd.DataFrame(columns=["action_taken"]))
        self.formatter.save_transcripts_to_local(transcript, "no-edits")
        history = pd.read_csv(self.root / "output" / "test_no-edits_edit-history.csv")
        self.assertTrue(history.empty)
        self.assertIn("action_taken", history.columns)

    def test_preset_selection_and_concatenation_use_the_selected_model_label(self):
        notebook_path = Path(__file__).resolve().parents[1] / "json-to-natural.ipynb"
        notebook = json.loads(notebook_path.read_text())
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]
        selection_source = next(source for source in sources if source.startswith("turn_model ="))
        concat_source = next(source for source in sources if source.startswith("def concatenate_transcripts_from_local("))
        for preset, expected_label in [("baseline", "audiophile"), ("backbiter", "backbiter"), ("cliffhanger", "cliffhanger"), ("natural_turn", "natural_turn-1500")]:
            with self.subTest(preset=preset):
                # Exercise the notebook with the same one-line preset selection
                # a user makes, and no corpus conversations to process.
                selection = ast.parse(selection_source)
                selection.body[0].value = ast.Name(id=preset, ctx=ast.Load())
                ast.fix_missing_locations(selection)
                namespace = {"baseline": baseline, "backbiter": backbiter, "cliffhanger": cliffhanger, "natural_turn": natural_turn, "conversation_ids": [], "pd": pd}
                exec(compile(selection, str(notebook_path), "exec"), namespace)
                self.assertEqual(namespace["output_label"], expected_label)
                utterance_frame(["Hello there."]).to_csv(self.root / "output" / f"test_{expected_label}.csv")
                exec(compile(concat_source, str(notebook_path), "exec"), namespace)
                combined = namespace["concatenate_transcripts_from_local"](
                    ["test"],
                    namespace["output_label"],
                    str(self.root / "output" / "{conversation_id}_{output_label}.csv"),
                )
                self.assertEqual(combined.utterance.tolist(), ["Hello there."])
                self.assertEqual(combined.conversation_id.tolist(), ["test"])

    def interrupted_sentence(self):
        """Left speaker's second sentence spans the right speaker's interjection."""
        return self.parse(
            [word("Hello", 0.0, 0.3), word("there", 0.4, 0.6), punctuation("."),
             word("So", 1.0, 1.2), word("I", 1.3, 1.4), word("was", 1.5, 1.7),
             word("thinking", 3.0, 3.4), punctuation(".")],
            [word("Yeah", 2.0, 2.3), punctuation(".")],
        )

    def test_token_df_keeps_every_token_with_inherited_punctuation_times(self):
        self.parse(
            [punctuation(","), word("Hello", 0.5, 0.8), punctuation("."), word("Bye", 2.0, 2.2)],
            [word("Hi", 1.0, 1.2, "0.95"), punctuation("!")],
        )
        tokens = self.formatter.json_to_token_df()
        self.assertEqual(list(tokens.columns), ["speaker", "start", "stop", "utterance", "confidence", "type"])
        self.assertEqual(tokens.utterance.tolist(), [",", "Hello", ".", "Hi", "!", "Bye"])
        self.assertEqual(tokens.speaker.tolist(), ["A", "A", "A", "B", "B", "A"])
        # leading punctuation has no preceding word, so it gets 0 like the baseline parser
        self.assertEqual(tokens.start.tolist(), [0.0, 0.5, 0.5, 1.0, 1.0, 2.0])
        self.assertEqual(tokens.stop.tolist(), [0.0, 0.8, 0.8, 1.2, 1.2, 2.2])
        self.assertEqual(tokens.type.tolist().count("punctuation"), 3)

    def test_cliffhanger_keeps_interrupted_sentences_whole(self):
        parsed = self.interrupted_sentence()
        frame = self.formatter.json_to_baseline_df(parsed)
        self.assertEqual(frame.utterance.tolist(), ["Hello there. So I was", "Yeah.", "thinking."])
        tokens = self.formatter.json_to_token_df()
        transcript = self.formatter.baseline_to_natural_turn(frame, cliffhanger, token_df=tokens)
        result = transcript.transcript
        self.assertEqual(result.utterance.tolist(), ["Hello there. So I was thinking.", "Yeah."])
        self.assertEqual(result.speaker.tolist(), ["A", "B"])
        self.assertEqual(result.start.tolist(), [0.0, 2.0])
        self.assertEqual(result.stop.tolist(), [3.4, 2.3])
        self.assertTrue(result.overlap.iloc[1])
        self.assertIsNone(transcript.edit_history)
        self.assertEqual(all_output_words(result), all_output_words(frame))

    def test_cliffhanger_saves_utterance_level_baseline_not_tokens(self):
        frame = self.formatter.json_to_baseline_df(self.interrupted_sentence())
        transcript = self.formatter.baseline_to_natural_turn(frame, cliffhanger, token_df=self.formatter.json_to_token_df())
        self.formatter.save_transcripts_to_local(transcript, "cliffhanger")
        saved_baseline = pd.read_csv(self.root / "output" / "test_baseline.csv")
        self.assertEqual(saved_baseline.utterance.tolist(), ["Hello there. So I was", "Yeah.", "thinking."])
        saved = pd.read_csv(self.root / "output" / "test_cliffhanger.csv")
        self.assertEqual(saved.utterance.tolist(), ["Hello there. So I was thinking.", "Yeah."])
        self.assertFalse((self.root / "output" / "test_cliffhanger_edit-history.csv").exists())

    def test_cliffhanger_retains_an_unterminated_final_fragment(self):
        # the right speaker's last words have no terminal punctuation; they join that speaker's preceding sentence
        self.parse(
            [word("Well", 0.0, 0.3), word("I", 0.4, 0.5), word("mean", 2.0, 2.3), punctuation(".")],
            [word("Sure", 1.0, 1.2), punctuation("."), word("Go", 3.0, 3.2), word("on", 3.3, 3.5)],
        )
        frame = self.formatter.json_to_baseline_df(self.formatter.construct_conversation_transcript())
        result = self.formatter.baseline_to_natural_turn(frame, cliffhanger, token_df=self.formatter.json_to_token_df()).transcript
        self.assertEqual(result.utterance.tolist(), ["Well I mean.", "Sure. Go on"])
        self.assertEqual(all_output_words(result), all_output_words(frame))

    def test_collapse_on_missing_punc_joins_a_speakers_turns_from_the_baseline(self):
        frame = self.formatter.json_to_baseline_df(self.interrupted_sentence())
        joiner = TranscriptConfig(collapse_on_missing_punc=True)
        result = self.formatter.baseline_to_natural_turn(frame, joiner).transcript
        self.assertEqual(result.utterance.tolist(), ["Hello there. So I was thinking.", "Yeah."])
        self.assertEqual(result.stop.tolist(), [3.4, 2.3])
        self.assertEqual(all_output_words(result), all_output_words(frame))

    def test_cliffhanger_without_tokens_raises(self):
        frame = utterance_frame(["Hello there."])
        with self.assertRaisesRegex(ValueError, "token-level data"):
            text.collapse_transcript_turns(Transcript(frame, cliffhanger))

    def test_sentence_presets_require_terminal_punctuation_cues(self):
        for flag in ("collapse_on_missing_punc", "preserve_intraturn_sentences"):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, "terminal_punc_cues"):
                    TranscriptConfig(**{flag: True, "terminal_punc_cues": []})

    def test_existing_presets_are_unchanged_by_the_new_fields(self):
        for config in (baseline, backbiter, natural_turn):
            with self.subTest(config=config.version):
                self.assertFalse(config.collapse_on_missing_punc)
                self.assertFalse(config.preserve_intraturn_sentences)


if __name__ == "__main__":
    unittest.main()
