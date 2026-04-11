# Agent Hint (Min Tokens)

Use only signal codes + probability:

- `KW_CLICKBAIT`
- `HEADLINE_CLICKBAIT`
- `LOW_SOURCE_EVIDENCE`
- `HEDGE_WORDS`
- `STYLE_EMOTIONAL`
- `PUNC_EXCESS`
- `MODEL_HIGH_FAKE_CONF`
- `RF_SUPPORT_FAKE`

Decision rule:

`FAKE` if `fake_prob >= 0.60` OR (`>=2` strong codes from list above).  
`REAL` if `fake_prob < 0.45` AND `LOW_SOURCE_EVIDENCE` absent.

Reply template (short):

`fake_prob=0.87 | verdict=fake | codes=KW_CLICKBAIT,HEADLINE_CLICKBAIT,PUNC_EXCESS`
