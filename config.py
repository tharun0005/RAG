from guardrails import Guard
from guardrails.hub import ToxicLanguage

sensitive_topic_guard = Guard(name='sensitive-topic-guard')
sensitive_topic_guard.use(
    ToxicLanguage,
    threshold=0.5,
    validation_method="sentence",
    on_fail="exception"
)

toxic_language_guard = Guard(name='toxic-language-guard')
toxic_language_guard.use(
    ToxicLanguage,
    threshold=0.6,
    validation_method="sentence",
    on_fail="exception"
)

combined_guard = Guard(name='combined-safety-guard')
combined_guard.use(
    ToxicLanguage,
    threshold=0.5,
    validation_method="sentence",
    on_fail="exception"
)
