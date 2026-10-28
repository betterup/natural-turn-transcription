# CANDOR transcript release decisions

## 2026-10-28 Backchannel vocabulary

The next CANDOR transcript release will retain the existing Backbiter rules while updating the shared cue vocabulary to reflect the 2024 AWS transcripts.

Final approved changes, implemented in `transcription/transcript_config.py`:

- Add `ok` and `hm` to `BACKCHANNEL_CUES`.
- Remove `a` and `e` from `BACKCHANNEL_CUES`.
- Leave `NOT_BACKCHANNEL_CUES` unchanged.

`ok` and `hm` address common spellings in the 2024 transcripts. Removing `a` and `e` avoids treating article fragments and spelled letters as acknowledgment cues; `ah`, `okay`, and `hmm` remain included. These changes apply wherever the shared cue list is used, including Backbiter and NaturalTurn.

This final decision supersedes the earlier approval to add `gosh` and `goodness`; neither has been added. No transcript outputs have been regenerated as part of this vocabulary change.

Further additions, including `gosh`, `goodness`, `absolutely`, `definitely`, `totally`, `true`, `correct`, `interesting`, and `great`, are deferred pending annotation data and a fuller assessment of the tradeoff. The release retains the existing classification and turn-assembly rules.

The existing Backbiter preset has no overlap or temporal-containment requirement and no question-answer exemption. It moves qualifying short utterances into backchannel fields rather than deleting their text. Short substantive answers can therefore be misclassified. Vocabulary updates alone do not resolve that limitation.
