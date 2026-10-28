# CANDOR transcript release decisions

## 2026-10-28 Backchannel vocabulary

The next CANDOR transcript release will retain the existing Backbiter rules while updating the shared cue vocabulary to reflect the 2024 AWS transcripts.

Approved changes:

- Add `ok`, `hm`, `gosh`, and `goodness` to `BACKCHANNEL_CUES`.
- Remove `a` and `e` from `BACKCHANNEL_CUES`.
- Leave `NOT_BACKCHANNEL_CUES` unchanged.

`ok` and `hm` address common spellings in the 2024 transcripts. `gosh` and `goodness` extend the existing treatment of reactions containing `god`. Removing `a` and `e` avoids treating article fragments and spelled letters as acknowledgment cues; `ah` remains included.

Implementation status: approved for the next vocabulary update; these changes have not yet been applied to `transcription/transcript_config.py`.

Still under discussion: `absolutely`, `definitely`, `totally`, `true`, `correct`, `interesting`, and `great`. No decision to add these seven words has been made.

The existing Backbiter preset has no overlap or temporal-containment requirement and no question-answer exemption. It moves qualifying short utterances into backchannel fields rather than deleting their text. Short substantive answers can therefore be misclassified. Vocabulary updates alone do not resolve that limitation.
