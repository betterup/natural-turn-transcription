from enum import Enum
from functools import lru_cache
from typing import List, Optional
import re

from pydantic import BaseModel, validator

BACKCHANNEL_CUES = [
    "a",
    "ah",
    "alright",
    "awesome",
    "cool",
    "dope",
    "e",
    "exactly",
    "god",
    "gotcha",
    "huh",
    "hmm",
    "mhm",
    "mm hmm",
    "mm",
    "mmm",
    "nice",
    "oh",
    "okay",
    "really",
    "right",
    "sick",
    "sucks",
    "sure",
    "uh",
    "um",
    "wow",
    "yeah",
    "yep",
    "yes",
    "yup",
]

NOT_BACKCHANNEL_CUES = [
    "and",
    "but",
    "i",
    "i'm",
    "it",
    "it's",
    "like",
    "so",
    "that",
    "that's",
    "we",
    "we're",
    "well",
    "you",
    "you're",
]

BANNED_TOKENS = []

SUSPICIOUS_TOKENS = []

class TranscriptConfigVersion(Enum):
    """Transcript assembly in V2/natural_turn is substantially different than in V1 models. 
       Rather than try and merge both into the same logic, this flag allows for branching
       into the appropriate logic in text.py."""
    V1 = 1 # baseline
    V2 = 2 # natural_turn 
    
    
class Transcript:    
    """Transcript contains original transcript (baseline), TranscriptConfig, 
       excluded turns, and edit history (in the case tokens are removed or masked from utterances."""
    def __init__(self, df, config):
        self.baseline = df
        self.config = config
        self.edit_history = None
    
    def set_transcript(self, transcript_df):
        self.transcript = transcript_df
    
    def set_edit_history(self, edit_history):
        self.edit_history = edit_history


class TranscriptConfig(BaseModel):
    """A custom configuration to humanize transcript turns for the `transcript_df_custom` method.

    Args
    ----------
    'version': (TranscriptConfigVersion) determines branch logic in text.py; V1 is for CANDOR-era models, V2 is for natural_turn+ (2023 and later).
    'label_turns': (bool) label primary vs secondary turn (e.g. backchannels), combine primary turns
    'combine_secondary_turns': (bool) combine secondary turns into list-column
    'pivot_secondary_turns': (bool) pivot secondary turns and turn statistics into wide-format "listener" columns
    'exclude_underconfident_utterances': (bool) omit baseline utterances if proportion of underconfident tokens is greater than :max_prop_underconfident: (determined by proportion of tokens falling below :token_confidence_threshold:)
    'remove_banned_tokens': (bool) Remove tokens on banned_tokens list from entire transcript.
    'banned_tokens': (List[str]) Tokens to be removed from the transcript. BE CAREFUL WITH THIS!
    'remove_suspicious_tokens': (bool) Remove tokens on suspicious_tokens list if they are also underconfident.
    'suspicious_tokens': (List[str]) Tokens to be removed from the transcript when also underconfident.
    'flag_underconfident_tokens': (bool) Wrap unexcluded underconfident tokens in brackets.
    'mask_underconfident_tokens': (bool) Replace unexcluded underconfident tokens with <<unknown>>.
    'include_token_confidence': (bool) include token-level confidence 
    'token_confidence_threshold': (float) minimum confidence (probability) for qualifying a token as "reliable", default 1.0
    'collapse_short_pauses': (bool) collapse pauses with less than N seconds between them
    'backchannel_word_max': (int) used in 'pivot_backchannels'. The maximum number of words in an utterance to be considered a backchannel
    'backchannel_second_max': (float) used in 'pivot_backchannels'. The maximum length of an utterance (in seconds) to be considered a backchannel
    'backchannel_pause_max': (float) used in 'pivot_backchannels'. The maximum length of a pause needed to consider the next turn a backchannel
    'backchannel_proportion': (float) used in 'pivot_backchannels'. The proportion of words that are 'backchannel_cues' for a short utterance to be considered a backchannel
    'backchannel_cues': (list) used in 'pivot_backchannels'. Tokens that are considered to be backchannels
    'not_backchannel_cues': (list) used in 'pivot_backchannels'. Optional tokens that can be used to indicate the start of a short turn rather than a backchannel
    'max_pause: (float) used in 'collapse_short_pauses'. The maximum length of a pause between two turns to qualify those turns for concatenation
    
    Note: kwargs can also be passed directly to `transcript_df_custom`.
    If no kwargs are provided, 'transcript_df_custom' defaults to baseline.
    """
    # baseline needs V1, b/c it has no :utterance_parts: and will throw a KeyError in V2.
    version: TranscriptConfigVersion = TranscriptConfigVersion.V1 
    
    token_confidence_threshold: float = 1.0
    include_token_confidence: bool = True
    exclude_underconfident_utterances: bool = False
    
    remove_banned_tokens: bool = False
    banned_tokens: Optional[List[str]] = BANNED_TOKENS
    
    remove_suspicious_tokens: bool = False
    suspicious_tokens: Optional[List[str]] = SUSPICIOUS_TOKENS
    
    flag_underconfident_tokens: bool = False
    mask_underconfident_tokens: bool = False
    
    label_turns: bool = False
    combine_secondary_turns: bool = False
    pivot_secondary_turns: bool = False
    
    collapse_short_pauses: bool = False
    max_pause: Optional[float] = 0.0
    
    pivot_backchannels: bool = False
    
    short_turn_word_max: int = 0
    short_turn_second_max: float = 0.0
    
    backchannel_word_max: Optional[int] = 0
    backchannel_second_max: Optional[float] = 0.0
    backchannel_proportion: Optional[float] = 0.0
    backchannel_pause_max: Optional[float] = 0.0
    backchannel_cues: Optional[List[str]] = BACKCHANNEL_CUES
    not_backchannel_cues: Optional[List[str]] = NOT_BACKCHANNEL_CUES
    
    
    class Config:
        extra = "forbid"

    def __hash__(self):
        def _to_hashable(item):
            if isinstance(item, list):
                return tuple(item)
            elif isinstance(item, dict):
                return tuple(item.items())
            else:
                return item

        to_hash = tuple(_to_hashable(val) for val in self.__dict__.values())

        return hash((type(self),) + to_hash)

    def check_backchannel_input(cls, values):
        if values["pivot_secondary_turns"] and (
            (values["backchannel_word_max"] == 0)
            and (values["backchannel_second_max"] == 0.0)
            and (values["backchannel_pause_max"] == 0.0)
        ):
            raise ValueError(
                "Must provide a word, duration, or pause maximum in order to pivot backchannels."
            )
        if (
            (values["backchannel_second_max"] < 0)
            or (values["backchannel_word_max"] < 0)
            or (values["backchannel_pause_max"] < 0)
        ):
            raise ValueError(
                f"Thresholds must be zero or greater; {values['backchannel_second_max']}, {values['backchannel_word_max']}, and {values['backchannel_pause_max']} were passed."
            )
        return values

    @validator("backchannel_proportion")
    def check_proportion(cls, v):
        if (v > 1) or (v < 0):
            raise ValueError(f"Backchannel proportion must be between 0 and 1; {v} was passed.")
        return v

    def check_short_pause_input(cls, values):
        if values["collapse_short_pauses"] and (values["max_pause"] == 0.0):
            raise ValueError(
                "Must provide a second maximum in order to collapse short pauses."
            )
        if (values["max_pause"] < 0):
            raise ValueError(
                f"Threshold must be zero or greater; {values['max_pause']} was passed."
            )
        return values


# preset for baseline
baseline = TranscriptConfig()

# .6 token confidence threshold determined after manual testing/review of several recordings
# confidence scores do not seem to track with accuracy in a linear manner, but anything below .6
# is often hallucinated speech or incorrect transcription in AWS
token_conf = 0.6 

# preset for natural_turn
natural_turn = TranscriptConfig(
    version = TranscriptConfigVersion.V2,
    
    token_confidence_threshold=token_conf,
    include_token_confidence=True,
    exclude_underconfident_utterances=True,
    remove_banned_tokens=True,
    remove_suspicious_tokens=True,
    
    banned_tokens=BANNED_TOKENS,
    suspicious_tokens=SUSPICIOUS_TOKENS,
    
    flag_underconfident_tokens=True,
    mask_underconfident_tokens=False,
    
    label_turns=True,
    combine_secondary_turns=True,
    pivot_secondary_turns=False, 
    
    collapse_short_pauses=True,
    max_pause=1.5,
    
    backchannel_word_max=3,
    backchannel_proportion=0.5,
    
    backchannel_cues=BACKCHANNEL_CUES,
    not_backchannel_cues=NOT_BACKCHANNEL_CUES,
)

# preset for natural_turn_wide
natural_turn_wide = TranscriptConfig(
    version = TranscriptConfigVersion.V2,
    
    token_confidence_threshold=token_conf,
    include_token_confidence=True,
    exclude_underconfident_utterances=True,
    remove_banned_tokens=True,
    remove_suspicious_tokens=True,
    
    banned_tokens=BANNED_TOKENS,
    suspicious_tokens=SUSPICIOUS_TOKENS,
    
    flag_underconfident_tokens=True,
    mask_underconfident_tokens=False,
    
    label_turns=True,
    combine_secondary_turns=True,
    pivot_secondary_turns=True,  # what makes it "wide"
    
    collapse_short_pauses=True,
    max_pause=1.5,
    
    backchannel_word_max=3,
    backchannel_proportion=0.5,
    
    backchannel_cues=BACKCHANNEL_CUES,
    not_backchannel_cues=NOT_BACKCHANNEL_CUES,
)

# preset for intermediate turn model described in Cooney & Reece, 2024, nicknamed "backbiter" 
# (see Reece et al., 2023 for details)
backbiter = TranscriptConfig(
    version = TranscriptConfigVersion.V1,
    
    pivot_backchannels=True,
    backchannel_word_max=3,
    backchannel_second_max=0.0,
    backchannel_pause_max=0.0,
    backchannel_proportion=0.5,
    backchannel_cues=BACKCHANNEL_CUES,
    not_backchannel_cues=NOT_BACKCHANNEL_CUES,
    # default V2 settings, required for TranscriptConfig but not used to define backbiter
    token_confidence_threshold=token_conf,
    include_token_confidence=True,
    exclude_underconfident_utterances=False,
    remove_banned_tokens=False,
    remove_suspicious_tokens=False,
    banned_tokens=[],
    suspicious_tokens=[],
    flag_underconfident_tokens=False,
    mask_underconfident_tokens=False,
    label_turns=False,
    combine_secondary_turns=False,
    pivot_secondary_turns=False, 
    collapse_short_pauses=False,
    max_pause=1.5,

)
