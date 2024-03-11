from enum import Enum
from functools import lru_cache
import logging
from math import ceil
from pathlib import Path
import re

import botocore.exceptions
import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.util import minibatch
from tqdm import tqdm

from .transcript_config import (
    Transcript, 
    TranscriptConfig, 
    TranscriptConfigVersion, 
    baseline, 
)

LOGGER = logging.getLogger(__name__)


# !! Used in V1 only !!
# Use these rules to join turns when humanizing turns
AGG_RULES = {
    "speaker": "first",
    "start": "min",
    "stop": "max",
    "utterance": lambda x: " ".join(x.tolist()),
    "confidence": list,
    "backchannel": lambda x: " ".join(x.dropna().tolist()),
    "backchannel_count": "sum",
    "backchannel_speaker": "first",
    "backchannel_start": "min",
    "backchannel_stop": "max",
}


# Load and cache the spacy model used to process text
@lru_cache(None)
def load_spacy_model(spacy_model="en_core_web_sm"):
    try:
        nlp = spacy.load(spacy_model)
        return nlp
    except OSError as e:
        if "Can't find model 'en_core_web_sm'." in str(e):
            print("Can't find spacy language model. Try running: ")
            print(
                "python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz#egg=en_core_web_sm"
            )
        else:
            raise
            
    
def is_backchannel(
    utterance_str,
    turn_length,
    config: TranscriptConfig = baseline,
):
    """ !! Used in V1 transcripts only!! 
    
    If an utterance meets set criteria, call it a backchannel.

    Optional criteria include:
        1) below a word/second max
        2) contains a  specified proportion of backchannel cues and does not start with a token in
           'not_backchannel_cues'

    Parameters
    ----------
    utterance_str : str
        An utterance string to evaluate
    turn_length : float
        Length of an utterance in secconds
    config : TranscriptConfig, optional
        A TranscriptConfig model used to construct humaized transcripts

    Returns
    -------
    Bool
    """
    nlp = load_spacy_model()
    utterance_doc = nlp.make_doc(utterance_str)

    # number of non-punct tokens
    n_words = len([w for w in utterance_doc if not w.is_punct])

    # if there are no words, it's not a backchannel
    if n_words == 0:
        return False

    # if it's long enough and there are more words than the limit, it's not a backchannel
    if (n_words > config.backchannel_word_max) and (turn_length > config.backchannel_second_max):
        return False

    # if the first term is in the list of 'not_backchannel_cues, it's not a backchannel
    if any(utterance_doc[0].text.lower() == cue.lower() for cue in config.not_backchannel_cues):
        return False

    # otherwise, check the proportion of words that are backchannel cues
    _backchannel_cues = [[{"LOWER": w}] for w in config.backchannel_cues]
    backchannel_matcher = Matcher(nlp.vocab)
    backchannel_matcher.add("backchannels", [*_backchannel_cues])
    matches = backchannel_matcher(utterance_doc)
    # if the proportion of backchannel cues is sufficiently high, it's a backchannel
    return (len(matches) / n_words) >= config.backchannel_proportion

    
def _v1_join_contiguous_utterances(df):
    """Utility to condense two turns from the same speaker"""
    df = df.copy()

    # give turns an index where we change speakers
    consec_turns = (df.speaker != df.speaker.shift(1)).cumsum()

    # if we have >= 2 rows with the same speaker, join
    try:
        rules_to_use = {k: AGG_RULES[k] for k in df.columns}
        aggd = df.groupby(consec_turns).aggregate(rules_to_use)
    except KeyError as e:
        raise ValueError(
            f"DF columns must be in {list(AGG_RULES.keys())}; "
            f"The dataframe provided contains the unsupported column: {e}."
        )

    aggd.index = pd.Series(range(aggd.shape[0]), name="turn_id")
    return aggd


def _v2_join_contiguous_utterances(df):
    """Utility to condense multiple utterances from the same speaker"""
    df = df.copy()
    if 'turn_id' in df.columns:
        df.drop('turn_id', axis='columns', inplace=True)

    # give turns an index where we change speakers
    #consec_turns = (df.speaker != df.speaker.shift(1)).cumsum()
    
    list_if_many = lambda x: x if len(x) == 1 else list(x)
    
    # concatenate contiguous utterances into turns
    agg_df = (
        df
        .groupby('speaker')
        .agg(speaker=('speaker', "first"),
             start=('start', "min"),
             stop=('stop', "max"),
             utterance=('utterance', list),
             utterance_type=('utterance_type', list_if_many),
             confidence=("confidence", list_if_many),
             original_utterance=("original_utterance", list),
        )
    )
    # reset turn_id index
    agg_df.index = pd.Series(range(agg_df.shape[0]), name="turn_id")

    return agg_df


def _unstack_backchannels(
    df,
    config: TranscriptConfig = baseline,
):
    """Pivot out rows identified as backchannels"""
    df = df.copy()
    df["delta_temp"] = df.stop - df.start  # temporary time delta col
    df["is_backchannel"] = df.apply(
        lambda row: is_backchannel(row["utterance"], row["delta_temp"], config),
        axis=1,
    )
    if config.backchannel_pause_max:
        # delta temp is equivalent to pause prior to turn in this case
        df["delta_temp"] = (df.start - df.stop.shift(1)).fillna(0)
        # don't include overlapping turns
        df["is_backchannel"] = (df.is_backchannel) | (
            (df.delta_temp > 0) & (df.delta_temp <= config.backchannel_pause_max)
        )

    # by definition, there should not be two consecutive backchannels
    # therefore, if the previous row is a backchannel, current row cannot also be a backchannel
    df["is_backchannel"] = df.is_backchannel & ~df.is_backchannel.shift(1).fillna(False)

    # indicator that the next turn is a backchannel during this turn
    next_turn_is_backchannel = df.shift(-1).is_backchannel.fillna(False)

    df["backchannel"] = np.where(next_turn_is_backchannel, df.utterance.shift(-1), np.nan)
    df["backchannel_count"] = np.where(next_turn_is_backchannel, 1, 0)
    df["backchannel_speaker"] = np.where(next_turn_is_backchannel, df.speaker.shift(-1), np.nan)
    df["backchannel_start"] = np.where(next_turn_is_backchannel, df.start.shift(-1), np.nan)
    df["backchannel_stop"] = np.where(next_turn_is_backchannel, df.stop.shift(-1), np.nan)

    # keep non-backchannel rows
    df = df[~df.is_backchannel].copy()

    df.drop(["delta_temp", "is_backchannel"], axis='columns', inplace=True)
    return df


def _collapse_short_turns(speaker_df, config: TranscriptConfig = baseline):
    """Collapse if turns is less than N words or seconds"""
    enough_words = speaker_df.utterance.str.count(" ") + 1 > config.short_turn_word_max
    enough_seconds = speaker_df.stop - speaker_df.start > config.short_turn_second_max

    if config.short_turn_second_max and config.short_turn_word_max:
        return (enough_words & enough_seconds).shift(1).cumsum().fillna(0)
    if config.short_turn_second_max:
        return enough_seconds.shift(1).cumsum().fillna(0)
    if config.short_turn_word_max:
        return enough_words.shift(1).cumsum().fillna(0)


def _collapse_on_missing_punc(speaker_df, config: TranscriptConfig = baseline):
    """Collapse if turn does not end in terminal punctuation ('.', '!', '?')"""
    return (
        speaker_df.utterance.apply(lambda x: x[-1] in config.terminal_punc_cues)
        .shift(1)  # next is turn _after_ terminal punct
        .cumsum()
        .fillna(0)
    )


def _create_sentences_from_tokens(df, config: TranscriptConfig = baseline):
    dfs = []
    for _, speaker in df.groupby("speaker"):
        start = np.inf
        stop = 0
        content = []
        processed_rows = []
        for i, row in enumerate(speaker.itertuples()):
            if row.start < start:
                start = row.start
            if row.stop > stop:
                stop = row.stop
            if row.type == "pronunciation":
                content.append(row.utterance)
            if row.type == "punctuation":
                # drops back to back punctuation (a rare transcription error)
                if content:
                    content[-1] = content[-1] + row.utterance  # no space before punc
                    if row.utterance in config.terminal_punc_cues:
                        processed_row = [row.speaker, start, stop, (" ").join(content)]
                        processed_rows.append(processed_row)
                        # reset trackers
                        start = np.inf
                        stop = 0
                        content = []
            # append final row regardless of terminal punc
            if (i == (len(speaker) - 1)) and content:
                processed_row = [row.speaker, start, stop, (" ").join(content)]
                processed_rows.append(processed_row)
        df = pd.DataFrame(processed_rows, columns=["speaker", "start", "stop", "utterance"])
        dfs.append(df)
    return pd.concat(dfs).sort_values("start")


def regex_replace_tokens(s, replacement_regex, replacement_dict):
    """Uses regex substitution to substitute tokens found by `replacement_regex`
    within string `s` with the corresponding item in `replacement_dict`
    """

    def _do_replacement(match):
        matched_phrase = match.group(0)
        replacement = replacement_dict[matched_phrase.lower()]

        if matched_phrase[0].isupper():
            replacement = replacement.capitalize()

        return replacement

    return replacement_regex.sub(_do_replacement, s)



# 8/24/2023 note: not all utterance types are currently in use
class UtteranceType(Enum):
    BACKCHANNEL = "backchannel"
    FRONTCHANNEL = "frontchannel"
    AFFILIATIVE = "affiliative"
    SUBSTANTIVE = "substantive"
    MISTRANSLATION = "mistranslation"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    FRAGMENT = "fragment"
    OTHER = "other"
    EMPTY = np.nan
    
    
class EditStage(Enum):
    EXCLUDE_UTTERANCES = 0
    REMOVE_BANNED = 1
    REMOVE_SUSPICIOUS = 2
    FLAG_UNDERCONFIDENT = 3

    
def _get_backchannel_matcher(nlp, config):
    """Match each token against backchannel cues."""
    _backchannel_cues = [[{"LOWER": w}] for w in config.backchannel_cues]
    backchannel_matcher = Matcher(nlp.vocab)
    backchannel_matcher.add("backchannels", [*_backchannel_cues])
    return backchannel_matcher


def determine_utterance_type(
    utterance_strs,
    turn_length,
    config: TranscriptConfig = baseline,
):
    """Label an utterance according to set criteria.

    Optional backchannel criteria include:
        1) below a word/second max
        2) contains a  specified proportion of backchannel cues and does not start with a token in
           'not_backchannel_cues'

    Parameters
    ----------
    utterance_strs : list[str]
        A list of utterance strings to evaluate. Each string in the list is a segment of one turn. List length >= 1.
    turn_length : float
        Length of an utterance in seconds
    config : TranscriptConfig, optional
        A TranscriptConfig model used to construct humanized transcripts

    Returns
    -------
    UtteranceType(Enum) value
    """
        
    nlp = load_spacy_model()
    utterance_str = ' '.join(utterance_strs)
    utterance_doc = nlp.make_doc(utterance_str)

    tokens = [w for w in utterance_doc if not w.is_punct]
    # number of non-punct tokens
    n_words = len(tokens)
    # number of unique non-punct tokens
    n_unique_words = len(set(tokens))
    
    if n_words == 0:
        return UtteranceType.EMPTY.value

    # Utterance assignment is all within the context of Secondary turns.
    # All Primary turn segments are just 'primary', we are
    # not going to try to break down further, for now. 
    # AGR, 24 Aug 2023

    # check the proportion of words that are backchannel cues
    backchannel_matcher = _get_backchannel_matcher(nlp, config)
    matches = backchannel_matcher(utterance_doc)
    match_prop = len(matches) / n_words

    # if EVERY word is in the backchannel list, then no matter how long the utterance, it's a backchannel
    if match_prop == 1.0: 
        return UtteranceType.BACKCHANNEL.value

    # if there are more words than the limit AND utterance is longer than backchannel limit, it's not a backchannel
    if ((n_words > config.backchannel_word_max) and (turn_length > config.backchannel_second_max)):
        return UtteranceType.SECONDARY.value

    # if the first term is in the list of :not_backchannel_cues:, it's not a backchannel
    if any(utterance_doc[0].text.lower() == cue.lower() for cue in config.not_backchannel_cues):
        return UtteranceType.SECONDARY.value

    # if the proportion of backchannel cues is sufficiently high, it's a backchannel
    if (match_prop) >= config.backchannel_proportion:
        return UtteranceType.BACKCHANNEL.value
    else:
        return UtteranceType.OTHER.value


def _get_adjacent_primary_speaker(row, df, direction):
    res_dict = {
        f"{direction}_primary_turn_id": np.nan,
        f"{direction}_primary_speaker": np.nan,
        f"{direction}_primary_turn_stop": np.nan,
        f"{direction}_primary_utterance": '',
        f"{direction}_primary_confidence": np.nan,
        f"{direction}_primary_original_utterance": '',
        f"same_primary_speaker_as_{direction}": False
    }    
    if row.is_primary:
        if direction == "next":
            new_ix = row.turn_id + 1
        elif direction == "prev":
            new_ix = row.turn_id - 1
        else:
            return ValueError(':direction: value must be "next" or "prev"')
        
        res = df.loc[lambda x: x.turn_id.eq(new_ix) & x.is_primary]

        if res.size > 0:
            res_dict = {
                f"{direction}_primary_turn_id": res.turn_id.array[0],
                f"{direction}_primary_speaker": res.speaker.array[0],
                f"{direction}_primary_turn_stop": res.stop.array[0],
                f"{direction}_primary_utterance": res.utterance.array[0],
                f"{direction}_primary_original_utterance": res.original_utterance.array[0],
                f"{direction}_primary_confidence": res.confidence.array[0],
                f"same_primary_speaker_as_{direction}": res.speaker.array[0] == row.speaker
            }
    return res_dict


def _join_contiguous_primary_utterances(df):
    """Concatenate consecutive primary (non-interjection) utterances.
       To be run AFTER _label_turns."""
    tmp_df = df.copy()

    next_df = tmp_df.apply(_get_adjacent_primary_speaker, args=(tmp_df, 'next',), axis='columns', result_type='expand')
    prev_df = tmp_df.apply(_get_adjacent_primary_speaker, args=(tmp_df, 'prev',), axis='columns', result_type='expand')
    tmp_df = pd.concat([tmp_df, next_df, prev_df], axis='columns') 
    
    def _get_merged_turn(x):
        if x.same_primary_speaker_as_next & x.is_primary:
            return {
                'new_utt': [x.utterance, x.next_primary_utterance], 
                'new_conf': [x.confidence, x.next_primary_confidence],
                'new_stop': x.next_primary_turn_stop,
                'new_orig_utt': [x.original_utterance, x.next_primary_original_utterance],
            }
        else:
            return {
                'new_utt': x.utterance, 
                'new_conf': x.confidence,
                'new_stop': x.stop,
                'new_orig_utt': x.original_utterance,
            }
        
    tmp_merged = tmp_df.apply(_get_merged_turn, axis='columns', result_type='expand')
    tmp_df = pd.concat([tmp_df, tmp_merged], axis='columns')
    
    # explicitly cast :x: as boolean here to get correct results
    tmp_df['remove_this_turn'] = tmp_df.apply(
        lambda x: x.same_primary_speaker_as_prev & x.is_primary, 
        axis=1
    )
    updated_df = (
        tmp_df.loc[lambda x: ~x.remove_this_turn]
        .drop([
            'stop',
            'original_utterance',
            'utterance', 
            'confidence',
            'next_primary_speaker', 
            'prev_primary_speaker', 
            'next_primary_utterance', 
            'prev_primary_utterance', 
            'next_primary_original_utterance',
            'prev_primary_original_utterance',
            'next_primary_confidence', 
            'prev_primary_confidence', 
            'next_primary_turn_id', 
            'prev_primary_turn_id', 
            'same_primary_speaker_as_next',
            'same_primary_speaker_as_prev',
            'remove_this_turn',
        ], axis=1)
    )

    # Insert :new_stop: as :stop: and :new_utt: as :utterance:
    # (currently 3rd and 4th column, as of 17 Aug 2023)
    updated_df.insert(3, 'stop', updated_df['new_stop']) 
    updated_df.insert(4, 'utterance', updated_df['new_utt']) 
    updated_df.insert(6, 'confidence', updated_df['new_conf']) 
    updated_df.insert(7, 'original_utterance', updated_df['new_orig_utt']) 
    
    to_drop = [
        'new_stop', 
        'new_utt', 
        'new_conf', 
        'new_orig_utt', 
    ]
    updated_df.drop(to_drop, axis='columns', inplace=True)
    
    # some interjections will have lost their primary turns. so we reindex :turn_id: in those cases.
    for row in updated_df.itertuples():
        if bool(row.is_primary):
            continue
            
        row_ix = row.Index
        row_iloc_ix = updated_df.index.get_loc(row.Index)
        row_turn_id = row.turn_id

        for row2 in updated_df.iloc[row_iloc_ix + 1:].itertuples():    
            if row2.is_primary: # if row2 is primary, stop
                break
            # give this interjection the same turn_id as primary turn :row:
            row2_ix = df.index.get_loc(row2.Index)
            updated_df.at[row2_ix, 'turn_id'] = row_turn_id

    # reindex all turn IDs
    updated_df['new_turn_id'] = updated_df.groupby(['turn_id']).ngroup()
    updated_df.rename(
        columns={
            'turn_id': 'old_turn_id', 
            'new_turn_id': 'turn_id'
        }, inplace=True
    )
    ordered_cols = [
        "turn_id",
        "speaker",
        "start",
        "stop",
        "utterance",
        "utterance_type",
        "confidence",
        "original_utterance",
        "is_primary",
    ]
    return updated_df[ordered_cols]


def _label_turns(
    df,
    config: TranscriptConfig = baseline,
):
    """Label rows as primary or secondary turns (incl. backchannels and other secondary types). 
       If a row is a secondary turn (i.e. from a listener during a speaker's turn), 
       reassign its turn ID to the turn ID of the speaker's turn ID.
       
        df: pd.DataFrame
            Current transcript df (may be collapsed or uncollapsed)
        config : TranscriptConfig, optional
            A TranscriptConfig model used to construct humanized transcripts
    """
    df = df.reset_index().copy()
    
    df['turn_id'] = df.index.values
    df['is_primary'] = True
    df['interjects_turn'] = np.nan
    df['delta_temp'] = df.stop - df.start

    for row in df.itertuples():
        row_ix = row.Index
        row_iloc_ix = df.index.get_loc(row.Index)
        row_start = row.start
        row_stop = row.stop

        for row2 in df.iloc[row_iloc_ix + 1:].itertuples():
            row2_start = row2.start
            row2_stop = row2.stop

            if (row2_start < row_stop) & (row2_stop <= row_stop): 
                row2_ix = df.index.get_loc(row2.Index)

                df.at[row2_ix, 'is_primary'] = False
                df.at[row2_ix, 'interjects_turn'] = row_ix
                # this row now belongs to the primary (speaker) turn it is interjecting
                # so we give it the same turn ID as speaker turn ID
                df.at[row2_ix, 'turn_id'] = row_ix
            else:
                # once we reach a non-interjection turn, stop this inner loop
                break

    df["utterance_type"] = df.apply(
        lambda row: row.utterance_type 
        if row.is_primary 
        else determine_utterance_type(row.utterance, row.delta_temp, config,),
        axis=1,
    )
        
    df['new_turn_id'] = df.groupby(['turn_id']).ngroup()
    df.rename(
        columns={
            'turn_id': 'old_turn_id', 
            'new_turn_id': 'turn_id'
        }, 
        inplace=True
    )
    
    return df
    

def get_cleaned_utterance(utt):
    if isinstance(utt, str):
        return utt
    if isinstance(utt, float):
        print(f'WARNING: Passed float {utt} as utt!')
        return ''
    # flatten any nested lists, concat as one string, split out on spaces into list
    flat_utt = list(pd.core.common.flatten(utt)) 
    joined_utt = ' '.join(flat_utt)
    return joined_utt
    

def _combine_secondary_turns(
    df,
    config: TranscriptConfig = baseline,
):
    """If there are multiple interjections during one speaker's turn, 
        combine them into lists."""
    df = df.copy()
    
    list_if_unlisted = lambda x: x if isinstance(x, list) else list(x)
    list_if_many = lambda x: x if len(x) == 1 else list(x)
    list_from_series = lambda x: x.tolist()[0] if len(x) == 1 else list(x)  

    def agg_parse(x):
        if len(x) > 1:
            return x.tolist()
        elif (len(x) == 1) and not isinstance(x.values[0], list) and (x.name == 'utterance_type'):
            return list(x)
        else:
            return x.tolist()[0]

    return (
        df
        .assign(
            pause=lambda x: x.pause.round(4),
            delta=lambda x: x.delta.round(4),
            questions=lambda x: x.questions.astype(float),
            n_words=lambda x: x.n_words.astype(float)
        )
        .groupby(['turn_id', 'is_primary'])
        .agg(
            speaker=("speaker", "first"),
            start=("start", agg_parse),
            stop=("stop", agg_parse),
            utterance=("utterance", get_cleaned_utterance),
            utterance_parts=("utterance", agg_parse),
            utterance_type=("utterance_type", agg_parse),
            confidence=("confidence", agg_parse),
            original_utterance=("original_utterance", "first"),
            delta=("delta", agg_parse),
            pause=("pause", agg_parse),
            questions=("questions", agg_parse),
            end_question=("end_question", agg_parse),
            overlap=("overlap", agg_parse),
            n_words=("n_words", agg_parse),
        )
        # present primary turns first, before their corresponding secondary turns, if there are any.
        .sort_index(level=['turn_id', 'is_primary'], ascending=[True, False]) 
    )


def _pivot_secondary_turns(df):
    """Pivot primary/secondary turns from long to wide format."""
    pivot_df = df.unstack(level="is_primary").copy()
    pivot_df.rename(
        columns=lambda is_primary: "speaker" if is_primary else "listener", 
        level=1, 
        inplace=True
    )
    pivot_df.sort_index(axis='columns', level=1, ascending=False, inplace=True)
    pivot_df.columns = [f"{x}_{y}" for x, y in pivot_df.columns.to_flat_index()]
    pivot_df.rename(
        columns={
            "speaker_speaker": "speaker",
            "start_speaker": "start",
            "stop_speaker": "stop",
            "delta_speaker": "delta",
            "utterance_speaker": "utterance",
        }, inplace=True
    )
    ordered_cols = [
        "speaker",
        "start",
        "stop",
        "delta",
        "utterance",
        "utterance_listener",
        "utterance_type_speaker",
        "utterance_type_listener",
        "start_listener",
        "stop_listener",
        "delta_listener",
        "pause_speaker",
        "pause_listener",
        "overlap_speaker",
        "overlap_listener",
        "n_words_speaker",
        "n_words_listener",
        "questions_speaker",
        "questions_listener",
        "end_question_speaker",
        "end_question_listener",
        "overlap_speaker",
        "overlap_listener",
    ]
    return pivot_df[ordered_cols]


def _collapse_short_pauses(speaker_df, config: TranscriptConfig = baseline):
    """Collapse based on pause length between turns, set by :config.max_pause:
        Return a list of new turn indexes, with turns marked for 
        concatenation bearing the same new turn index."""
    new_row_ix = list()
    ix = 0
    
    for row in speaker_df.itertuples():
        curr_ix = row.Index
        curr_iloc_ix = speaker_df.index.get_loc(row.Index)
        try:
            next_row = speaker_df.iloc[curr_iloc_ix + 1]
        except IndexError as e:
            new_row_ix.append(ix)
            continue

        curr_stop = row.stop
        next_start = next_row.start
        curr_stop_to_next_start = next_start - curr_stop

        new_row_ix.append(ix)
        
        if curr_stop_to_next_start >= config.max_pause:    
            ix += 1
            
    return new_row_ix


def _humanize_turns_per_speaker(df, new_turn_func, join_func, config: TranscriptConfig = baseline):
    """Assign new turn ids based on new_turn_func, and group contiguous utterances
       For use in V1 and V2 transcripts. """
    per_speaker_new_turns = []
    for s, speaker_df in df.groupby("speaker"):
        new_turns = new_turn_func(speaker_df, config)
        for t, turn_df in speaker_df.groupby(new_turns):
            joined = join_func(turn_df)
            per_speaker_new_turns.append(joined)

    new_turns = (
        pd.concat(per_speaker_new_turns)
        .sort_values("start")
        .reset_index()
        .drop(["turn_id"], axis=1)
    )
    return new_turns


def get_low_conf_toks(row, conf_thresh):
    """*Helper function, not part of Transcript construction*
        Get tokens with corresponding confidence scores that fall below :conf_thresh:."""
    confs = list(pd.core.common.flatten(row.confidence))
    utt_list = row.utterance.split(' ')
    if len(confs) != len(utt_list):
        print('conf and utt list lengths DO NOT match!', f'turn id: {row.name}')
        print(f'confs: {confs}')
        print(utt_list, f'len = {len(utt_list)}')
        print(confs, f'len = {len(confs)}')
        return []
    return [utt for i, utt in enumerate(utt_list) if confs[i] < conf_thresh]


def get_low_conf_scores(row, conf_thresh):
    """*Helper function, not part of Transcript construction*
        Get token-level confidence scores that fall below :conf_thresh:."""
    confs = list(pd.core.common.flatten(row.confidence))
    return [conf for conf in confs if conf < conf_thresh]


def _is_underconfident_utterance(x, thresh):
    """Get proportion of tokens with confidence values (from AWS Transcribe) below :thresh:."""
    # AGR 8/30/2023
    # UPDATE: Remove ANY utterance in which EVERY token falls below conf threshold.
    #
    return all([tok < thresh for tok in x])
    # AGR 8/25/2023
    # Based on transcript review, the most common case of an invalid utterance -
    # where the speaker isn't actually speaking, but instead some kind of background
    # noise was picked up as speech - registers as a single token. There may be 
    # exceptional cases of multi-token "false speech" but we are taking the position
    # that they are far less common than legitimate multi-token utterances that simply
    # were determined with low confidence by AWS Transcribe. We do not want to exclude
    # actual speech, so if there is more than one token, it automatically gets to stay.
    #return (len(x) == 1) & (x[0] < thresh)


#def _exclude_underconfident_utterances(df, excl_df, config: TranscriptConfig = baseline):
def _exclude_underconfident_utterances(df, edit_stage, config: TranscriptConfig = baseline):
    """Exclude utterances with all low-confidence tokens. Return new df with 
       underconfident tokens removed, along with excl_df listing removed utterances."""
    df = df.copy()
    df['is_underconfident'] = df.confidence.apply(
        _is_underconfident_utterance, 
        args=(config.token_confidence_threshold,)
    ) 
    excl_df = df.loc[df.is_underconfident].drop('is_underconfident', axis='columns').copy()
    excl_df['utterance_post_edit'] = '<removed>'
    excl_df['utterance_pre_edit'] = df.utterance
    excl_df['confidence_post_edit'] = np.nan
    
    edit_df = excl_df.apply(
        _log_action_taken, 
        args=(edit_stage,), 
        axis='columns',
        result_type='expand',
    )    
    excl_df_cols = [
        'speaker', 
        'start', 
        'stop', 
        'original_utterance', 
        'utterance_pre_edit', 
        'utterance_post_edit',
        'confidence', 
        'confidence_post_edit',
    ]
    edit_df = pd.concat([excl_df[excl_df_cols], edit_df], axis='columns')
    df = df.loc[~df.is_underconfident].drop('is_underconfident', axis='columns')
    
    return df, edit_df


def _update_excl_df(old_df, new_df):
    if not len(old_df):
        excl_df = new_df
    else: # remove_banned_tokens may have already produced an excl_df
        excl_df = (
            pd.concat([old_df, new_df])
            .sort_index()
        )    
    return excl_df


def _make_excl_df(df, excl_df):
    """Make df of utterances which have had tokens removed by _remove_problem_tokens(). 
        If excl_df already is populated, append new df to existing one."""
    df = df.copy()
    tmp_excl_df = (
        df
        .loc[df.is_edited]
        [[
            'speaker', 
            'start', 
            'stop', 
            'original_utterance',
            'utterance_post_edit', 
            'confidence',
        ]]
    )
    return _update_excl_df(excl_df, tmp_excl_df)


def _log_action_taken(
    row, 
    edit_stage, 
    is_removed=False, 
    is_edited=False, 
    is_marked=False
):
    """Record details of edits made to transcript at each :edit_stage:."""
    update_details = { 
        'edit_stage_label': edit_stage.name.lower(),
        'edit_stage': edit_stage.value,
        'action_taken': 'unchanged' # default
    }
    # we only get here from exclude_underconfident_utterances() if we are definitely removing row
    # so simply having edit_stage == EXCLUDE_UTTERANCES tells us what the action_taken is ('removed')
    if (edit_stage == EditStage.EXCLUDE_UTTERANCES) | is_removed:
        update_details['action_taken'] = 'removed'
    elif is_edited:
        update_details['action_taken'] = 'edited'
    elif is_marked:
        update_details['action_taken'] = 'marked'
    return update_details
    
    
def _rmv_toks(row, edit_stage, config):
    """Inner function for 'remove tokens' variants (e.g., banned, suspicious)."""
    tok_list = row['utterance'].split(' ')
    remove_ix = []
    
    for i, tok in enumerate(tok_list):
        clean_tok = re.sub('[\\.,\\?!]', '', tok.lower())
    
        if edit_stage == EditStage.REMOVE_SUSPICIOUS:
            if not isinstance(row.confidence, float): 
                is_underconfident = row.confidence[i] < config.token_confidence_threshold
                is_suspicious = clean_tok in config.suspicious_tokens
                if is_underconfident and is_suspicious:
                    remove_ix.append(i)
            else:
                print('this row has a NaN confidence value, should have been removed!')
                print(row)

        elif edit_stage == EditStage.REMOVE_BANNED:
            if clean_tok in config.banned_tokens:
                remove_ix.append(i)
        else:
            return ValueError('edit_stage must be a valid entry in EditStage(Enum)')
        
    new_utt = ' '.join([tok for ix, tok in enumerate(tok_list) if ix not in remove_ix])
    new_conf = [conf for ix, conf in enumerate(row.confidence) if ix not in remove_ix] 
    if not len(new_conf):
        new_utt = '<removed>'
        new_conf = np.nan
    removed_toks = [(tok, ix) for ix, tok in enumerate(tok_list) if ix in remove_ix]
    is_edited = bool(len(removed_toks))
    entire_utt_is_removed = len(removed_toks) == len(tok_list)

    edit_details = _log_action_taken(
        row, 
        edit_stage=edit_stage, 
        is_edited=is_edited, 
        is_removed=entire_utt_is_removed,
    )

    return {
        'speaker': row.speaker,
        'start': row.start,
        'stop': row.stop,
        'original_utterance': row.original_utterance,
        'confidence': row.confidence,
        'utterance_pre_edit': row.utterance,
        'utterance_post_edit': new_utt, 
        'confidence_post_edit': new_conf,
        'is_removed': entire_utt_is_removed,
        'removed_toks': removed_toks, 
        'edit_stage_label': edit_details['edit_stage_label'],
        'edit_stage': edit_details['edit_stage'],
        'action_taken': edit_details['action_taken'],
    }

    
def _remove_problem_tokens(df, edit_df, edit_stage, config: TranscriptConfig = baseline):
    """Remove 'banned' or 'suspicious' tokens, along with any trailing punctuation."""
    if edit_stage == EditStage.REMOVE_BANNED: # always remove
        rmv_list = config.banned_tokens
    elif edit_stage == EditStage.REMOVE_SUSPICIOUS: # only remove if also low-conf
        rmv_list = config.suspicious_tokens
    else:
        return ValueError('edit_stage must be EditStage.REMOVE_BANNED or EditStage.REMOVE_SUSPICIOUS. ')
    if not len(rmv_list):
        return df, edit_df
     
    df = df.copy()
    tmp_edit_df = df.apply(_rmv_toks, args=(edit_stage, config,), axis='columns', result_type='expand')
    
    if len(edit_df) > 0:
        edit_df = pd.concat([
            edit_df, 
            # only document edit actions that resulted in a change
            tmp_edit_df.loc[tmp_edit_df.action_taken.ne('unchanged')]
        ])
    else:
        # only document edit actions that resulted in a change
        edit_df = tmp_edit_df.loc[tmp_edit_df.action_taken.ne('unchanged')]
        
    merged_df = (
        pd.concat([
            df.drop(['utterance', 'confidence'], axis='columns'), 
            tmp_edit_df[['utterance_post_edit', 'confidence_post_edit', 'is_removed']],
        ], axis='columns')
    )
    # don't put this all in one chain; boolean types get weird with loc[lambda:] syntax
    # if you need to make this all into one long df definition, use np.logical_not(x.is_removed) in lambda
    df = (
        merged_df
        .loc[~merged_df.is_removed] # drop rows with completely-removed utts
        .drop('is_removed', axis='columns')
        .rename(columns={
            'utterance_post_edit': 'utterance',
            'confidence_post_edit': 'confidence'
        })
    )
    return df, edit_df


def _mark_toks(row, edit_stage, config):
    """Inner function for _flag_underconfident_tokens().
    
        Note: _mark_toks() is very similar in structure to _rmv_toks(), but they
        differ enough so that we made the call to keep them separate, rather than
        fill one generalized function with branching logic.
    """
    tok_list = row.utterance.split(' ')
    ct = 0
    for i, tok in enumerate(tok_list):
        if row.confidence[i] < config.token_confidence_threshold:
            ct += 1
            if config.mask_underconfident_tokens:
                tok_list[i] = "<<unknown>>"
            else:
                tok_list[i] = f"<<{tok_list[i]}>>"
    updated_utt = ' '.join(tok_list)
    is_marked = ct > 0
    edit_details = _log_action_taken(row, edit_stage, is_marked=is_marked)
    
    return {
        'speaker': row.speaker,
        'start': row.start,
        'stop': row.stop,
        'original_utterance': row.original_utterance,
        'confidence': row.confidence,
        'utterance_pre_edit': row.utterance,
        'utterance_post_edit': updated_utt,
        'confidence_post_edit': row.confidence,
        'edit_stage': edit_details['edit_stage'],
        'edit_stage_label': edit_details['edit_stage_label'],
        'action_taken': edit_details['action_taken'],
    }


def _flag_underconfident_tokens(df, edit_df, edit_stage, config: TranscriptConfig = baseline):
    df = df.copy()
    tmp_edit_df = df.apply(_mark_toks, args=(edit_stage, config,), axis='columns', result_type='expand')
    df['utterance'] = tmp_edit_df['utterance_post_edit']
    
    tmp_edit_df = tmp_edit_df.loc[tmp_edit_df.action_taken.ne('unchanged')] 
    edit_df = pd.concat([edit_df, tmp_edit_df])
    
    return df, edit_df
        
        
def collapse_transcript_turns(
    transcript: Transcript,
    verbose=False
):
    """Collapses the raw AWS transcript into a dataframe where each row is a spoken turn based on a set of parameters, including the start time of the utterance, the stop time, and backchannels"""
    # start with baseline 
    # note: set config.include_token_confidence == True to include confidence scores
    df = transcript.baseline.copy() # baseline = "Audiophile" transcript config (see Reece et al., 2023 for details)
    df.start = df.start.astype(float)
    df.stop = df.stop.astype(float)
    
    ##
    # V1 logic
    ##
    if transcript.config.version == TranscriptConfigVersion.V1:
        # pivot out backchannels
        if transcript.config.pivot_backchannels:
            df = _unstack_backchannels(df, config=transcript.config)
            df = _v1_join_contiguous_utterances(df)

        # collapse any remaining short turns
        if transcript.config.collapse_short_turns:
            df = _humanize_turns_per_speaker(
                df, 
                new_turn_func=_collapse_short_turns, 
                join_func=_v1_join_contiguous_utterances, 
                config=transcript.config
            )
            df = _v1_join_contiguous_utterances(df)

        df = v1_utterance_characteristics(df)
        df.speaker = df.speaker.str.lstrip()
        
        transcript.set_transcript(df)
        return transcript
    
    ##
    # V2 logic
    ##
    elif transcript.config.version == TranscriptConfigVersion.V2:
        df['original_utterance'] = df.utterance
        df['is_edited'] = False
        edit_df = pd.DataFrame()

        # Remove confidence for punctuation, which is always 0.0 - and actual speech token
        # confidence scores are practically never zero.
        # Otherwise the presence of the zero values may increase likelihood of analysis error.
        # (e.g. taking mean confidence per utterance)
        # Notes: 
        #   * Do this before exclude_underconfident_utterances()
        if transcript.config.include_token_confidence:
            df['confidence'] = df['confidence'].apply(lambda x: np.array([c for c in x if c != 0.]))
        else:
            df.drop("confidence", axis='columns', inplace=True)
            
        # Exclude underconfident utterances BEFORE collapsing turns across short pauses!
        # Otherwise there may be underconfident tokens (which we are assuming are noise and not speech)
        # which may sit in the middle of what would otherwise be longer pauses, which would result in 
        # collapsing turns that should actually be separate. 
        if transcript.config.exclude_underconfident_utterances:
            edit_stage = EditStage.EXCLUDE_UTTERANCES
            if verbose:
                print(f'{edit_stage.name.upper()}. MIN CONFIDENCE: {transcript.config.token_confidence_threshold}')
            df, edit_df = _exclude_underconfident_utterances(df, edit_stage, config=transcript.config)
            if verbose:
                print(f'EXCLUDED {len(edit_df)} utterances.\n  See Transcript.edit_history for details.')

        # config.banned_tokens is a list of tokens which are universally unreliable, regardless of confidence.
        # If this flag is set, remove banned tokens from every utterance. 
        # If an utterance consists entirely of banned tokens, remove the utterance and reindex turn_id.
        if transcript.config.remove_banned_tokens:
            edit_stage = EditStage.REMOVE_BANNED
            if verbose:
                print(f'{edit_stage.name.upper()} TOKENS')
                print(f'{edit_stage.name.upper()} list: {transcript.config.banned_tokens}')
            old_edit_df_len = len(edit_df)
            df, edit_df = _remove_problem_tokens(df, edit_df, edit_stage=edit_stage, config=transcript.config)
            new_edit_df_len = len(edit_df.loc[edit_df.action_taken.ne('unchanged')])
            if verbose:
                print(f'{edit_stage.name.upper()} tokens from {new_edit_df_len - old_edit_df_len} utterances.\n  See Transcript.edit_history for details.')

        # config.suspicious_tokens is a list of tokens to be removed ONLY WHEN they are low-confidence.
        # If this flag is set, remove underconfident suspicious tokens from every utterance. 
        # If an utterance consists entirely of underconfident suspicious tokens, remove the utterance and reindex turn_id.
        if transcript.config.remove_suspicious_tokens:
            edit_stage = EditStage.REMOVE_SUSPICIOUS
            if verbose:
                print(f'{edit_stage.name.upper()} TOKENS')
                print(f'{edit_stage.name.upper()} list: {transcript.config.suspicious_tokens}')
            old_edit_df_len = len(edit_df)
            df, edit_df = _remove_problem_tokens(df, edit_df, edit_stage=edit_stage, config=transcript.config)
            new_edit_df_len = len(edit_df.loc[edit_df.action_taken.ne('unchanged')])
            if verbose:
                print(f'{edit_stage.name.upper()} tokens from {new_edit_df_len - old_edit_df_len} utterances.\n  See Transcript.edit_history for details.')

        # Flag underconfident tokens without removing them by adding << >> brackets around them, i.e. <<foo>>.
        # If config.mask_underconfident is also set to True, mask underconfident tokens by replacing token with <<unknown>>
        # If an utterance consists entirely of low-confidence suspicious tokens, remove the utterance and reindex turn_id.
        if transcript.config.flag_underconfident_tokens:
            edit_stage = EditStage.FLAG_UNDERCONFIDENT
            if verbose:
                print(f'{edit_stage.name} TOKENS')
            old_edit_df_len = len(edit_df)
            df, edit_df = _flag_underconfident_tokens(df, edit_df, edit_stage, config=transcript.config)
            new_edit_df_len = len(edit_df.loc[edit_df.action_taken.ne('unchanged')])
            if verbose:
                print(f'{edit_stage.name.upper()} tokens from {new_edit_df_len - old_edit_df_len} utterances.\n  See Transcript.edit_history for details.')
            
        if transcript.config.label_turns:
            df['utterance_type'] = "primary"

        # Collapse turns with short pauses between them. 
        # Max pause length defined in config.max_pause
        if transcript.config.collapse_short_pauses:
            if verbose:
                print('COLLAPSE SHORT PAUSES')
            df = _humanize_turns_per_speaker(
                df, 
                new_turn_func=_collapse_short_pauses, 
                join_func=_v2_join_contiguous_utterances, 
                config=transcript.config
            )

        if transcript.config.label_turns:
            if verbose:
                print('LABEL PRIMARY VS SECONDARY TURNS')
            df = _label_turns(df, config=transcript.config)
            df = _join_contiguous_primary_utterances(df)

        df = v2_utterance_characteristics(df)

        # Assign secondary turns (backchannels etc) same turn ID as corresponding primary turn
        if transcript.config.combine_secondary_turns:
            if verbose:
                print('COMBINE SECONDARY TURNS')
            df = _combine_secondary_turns(df, config=transcript.config)
        else:
            df['utterance_parts'] = df.utterance
            df['utterance'] = df.utterance.apply(get_cleaned_utterance)

        # Pivot out secondary turns into wide format
        if transcript.config.combine_secondary_turns and transcript.config.pivot_secondary_turns:
            if verbose:
                print('PIVOT SECONDARY TURNS')
            df = _pivot_secondary_turns(
                df[[
                    'speaker', 
                    'start', 
                    'stop', 
                    'utterance', 
                    'utterance_parts',
                    'utterance_type', 
                    'confidence', 
                    'original_utterance', 
                    'delta', 
                    'pause',
                    'questions', 
                    'end_question', 
                    'overlap', 
                    'n_words'
                ]]
            )

        # Speaker ids sometimes contain leading space. remove.
        df.speaker = df.speaker.str.lstrip()
        if len(edit_df) > 0:
            if not (transcript.config.remove_banned_tokens or transcript.config.remove_suspicious_tokens):
                edit_df['removed_toks'] = np.nan
            edit_df.speaker = edit_df.speaker.str.lstrip()
            edit_df = (
                edit_df
                .set_index(['edit_stage'], append=True)
                .sort_index()
                .rename(columns={'confidence': 'confidence_pre_edit'})
                [[
                    'edit_stage_label', 
                    'action_taken', 
                    'speaker', 
                    'start', 
                    'stop', 
                    'utterance_pre_edit',
                    'utterance_post_edit', 
                    'confidence_pre_edit', 
                    'confidence_post_edit',
                    'original_utterance', 
                ]]
            )
        transcript.set_transcript(df)
        transcript.set_edit_history(edit_df)
    else:
        return ValueError(':version: must be a valid TranscriptConfigVersion value.')
    
    return transcript
    

def v1_utterance_characteristics(
    df, overlap_offset_s=0.0, include_interruption=False, config: TranscriptConfig = baseline
):
    nlp = load_spacy_model()
    df = df.assign(
        spacy_doc=lambda x: x.utterance.apply(nlp),
        pause=lambda x: x.start - x.stop.shift(1),
        delta=lambda x: x.stop - x.start,
        questions=lambda x: x.utterance.apply(lambda s: s.count("?")),
        end_question=lambda x: x.utterance.apply(lambda s: s.endswith("?")),
    ).assign(
        overlap=lambda x: x.pause + overlap_offset_s < 0,
        n_words=lambda x: x.spacy_doc.apply(
            lambda doc: len([tok for tok in doc if not tok.is_punct])
        ),
    )

    if include_interruption:
        df["delta_temp"] = df.stop - df.start  # temporary time delta col
        df["backchannel"] = df.apply(
            lambda row: is_backchannel(row["utterance"], row["delta_temp"], config),
            axis=1,
        )
        df["interruption"] = (
            ~(df.backchannel > 0) & df.overlap & ~((df.backchannel > 0).shift(1).fillna(False))
        )
        df.drop("delta_temp", axis=1, inplace=True)
    elif not include_interruption:
        return df
    else:
        raise Exception(
            "To compute interruptions, first set 'pivot_backchannels' to True in config."
        )

    return df


def v2_utterance_characteristics(
    df, overlap_offset_s=0.0, include_interruption=False, config: TranscriptConfig = baseline
):
    """Compute additional turn statistics."""
    
    #def flatten_and_apply_nlp(s, nlp_func):
    #    flattened_s = list(pd.core.common.flatten(s)) 
    #    s = ' '.join(flattened_s)
    #    return nlp_func(s)
    
    def flatten_and_count_question_marks(s):
        flattened_s = list(pd.core.common.flatten(s)) # common.flatten() works for strs, and hybrid lists of lists-and-strs
        s = ' '.join(flattened_s)
        return s.count("?")
    
    def flatten_and_check_if_q_at_end(s):
        flattened_s = list(pd.core.common.flatten(s)) 
        s = ' '.join(flattened_s)
        return s.endswith("?")
    
    def make_utt_list(utt):
        if not isinstance(utt, str):
            # flatten any nested lists, concat as one string, split out on spaces into list
            flat_utt = list(pd.core.common.flatten(utt)) 
            joined_utt = ' '.join(flat_utt)
            split_utt = joined_utt.split(' ')
            doc = split_utt
        else:
            doc = utt.split(' ')
        return doc
    
    def get_n_words(x):
        doc = x.utt_list
        punct_list = [',', '?', '.', '!'] # need this b/c spacy is_punct was including apostrophes?!
        return len([tok for tok in doc if not tok in punct_list]) 
    
    df = df.assign(
        utt_list=lambda x: x.utterance.apply(make_utt_list),
        pause=lambda x: x.start - x.stop.shift(1),
        delta=lambda x: x.stop - x.start,
        questions=lambda x: x.utterance.apply(flatten_and_count_question_marks),
        end_question=lambda x: x.utterance.apply(flatten_and_check_if_q_at_end),
    ).assign(
        overlap=lambda x: x.pause + overlap_offset_s < 0,
        n_words=lambda x: x.apply(get_n_words, axis='columns'),
    )
    return df


def compute_speaker_stats(
    convo,
    warmup=0,
    overlap_offset_s=0.2,
    return_series=False,
    user_col="speaker",
    config: TranscriptConfig = baseline,
):
    convo = convo.loc[warmup:, :]

    if config.version == TranscriptConfigVersion.V1:
        augment_func = v1_utterance_characteristics
    elif config.version == TranscriptConfigVersion.V2:
        augment_func = v2_utterance_characteristics
        
    augmented = augment_func(
        convo, overlap_offset_s=overlap_offset_s, include_interruption=True, config=config
    )

    # These are boolean columns. Pandas will try to convert back into origial dtype (bool)
    # after aggregation, such that 0s and 1s become Falses and Trues. To fix we cast to int.
    augmented[["overlap", "interruption"]] = augmented[["overlap", "interruption"]].astype(int)

    aggs = dict(
        start=("start", "min"),
        stop=("stop", "max"),
        turns=(user_col, "count"),
        time_speaking=("delta", "sum"),
        median_turn_length=("delta", "median"),
        median_pause_length=("pause", "median"),
        questions=("questions", "sum"),
        overlaps=("overlap", "sum"),
        interruptions=("interruption", "sum"),
        total_words=("n_words", "sum"),
    )

    if "backchannels" in augmented.columns:
        aggs["backchannel_count"] = (("backchannels", "sum"),)

    stats = (
        augmented
        # per-speaker aggregates
        .groupby(user_col).aggregate(**aggs)
        # comparative statistics
        .assign(
            time_listening=lambda x: x.time_speaking.sum() - x.time_speaking,
            floor_ratio=lambda x: x.time_speaking / x.time_speaking.sum(),
            question_rate=lambda x: x.questions / x.time_speaking,
            overlaps_per_turn=lambda x: x.overlaps / (x.turns.sum() - x.turns),
            words_per_second=lambda x: x.total_words / x.time_speaking,
        )
    )

    if "backchannels" in stats.columns:
        stats["backchannels_per_second"] = stats.backchannels / stats.time_listening

    if return_series:
        return stats.unstack(-1)
    else:
        return stats

