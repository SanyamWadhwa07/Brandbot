# Decision log

Fifteen choices that could reasonably have gone the other way, and why they went
this way. Decisions that were obvious are not here.

---

**1. One brand, chosen by a rule written before looking.**

`hulu_support`, 14,122 conversations. The criterion was fixed first: enough volume
to induce a taxonomy, enough resolved threads to retrieve from, and a support style
that answers in public rather than deflecting to DM. Picking the brand after
inspecting which one flattered the pipeline would have made every later number a
selection effect.

**2. Resolution scored on the customer's sign-off, not the brand's reply.**

Brands reply to almost everything, so counting replies measures staffing, not
success. A customer saying "thanks, that worked" is weak evidence but it is
evidence about the outcome. It only fires on 17% of threads, which is itself worth
knowing: for most tickets nobody can tell whether the answer helped.

**3. Conversations with more than one customer are thrown away.**

Reply chains fan in. When the brand posts a broadcast, hundreds of unrelated people
reply underneath it, and reconstructing threads by following parent pointers merges
them into one component. The largest here held 972 distinct customers. Those are not
conversations, and left in they would poison both retrieval and the golden set.

**4. Split by time, never at random.**

Retrieval in deployment can only look backwards. A random split lets the agent
retrieve a precedent written after the message it is answering, which inflates every
number and could never hold in production. The earliest 70% is the retrieval corpus,
the next 18% is for shaping prompts and taxonomy, the latest 12% is the golden set.
The golden set therefore cannot appear in the index by construction, not by a rule
someone has to remember.

**5. The taxonomy was over-clustered, hand-merged, then frozen and tagged.**

Choosing the right number of intents up front is guesswork. Splitting into twenty
groups and merging by eye is not, and the merge is where domain judgement belongs.
The result was tagged `v0.1-taxonomy-frozen` before a single golden label existed,
so the label set cannot have been shaped by what the model turned out to be good at.

**6. `langdetect` was tried and rejected, with receipts.**

It labelled "WHERE IS SEASON 3 OF FOOD WARS" as German and "I'm having major Hulu
problems" as Norwegian, both at 0.9999 confidence. Routing on it would have escalated
plain English and created a failure mode rather than removing one. What replaced it
is a script check that cannot misfire on Latin text. Exactly one message in 14,122 is
genuinely not English, so the cheap check is the right size of solution.

**7. Embeddings run locally instead of on a metered endpoint.**

The hosted free tier meters 100 texts a minute, so the index would have taken about
2.3 hours and stalled repeatedly. `nomic-embed-text` under Ollama does it in about
four minutes with no quota and no key, and separates this corpus better: for "hulu
keeps buffering" against "cancel my subscription" the hosted model returned 0.772 and
this one returns 0.453, while still ranking the genuinely related pair higher.

**8. The judge is pinned and never falls back.**

A judge that silently swaps models mid-run mixes two graders into one score and
quietly invalidates the agreement statistics. An unservable judge call raises instead.
Aliases ending in `-latest` are avoided for the same reason: the model behind them can
change under a name that looks pinned.

**9. The judge is from a different family than everything it grades.**

The agent runs on open-weight models through one provider; the judge is Gemini. A
model grading its own family rewards replies for sounding like something it would
have written. Using a different family removes the worst of that, not all of it,
which is why a third model re-grades a sample and the report states the gap.

**10. Golden-set sampling has two strata, and they stay separable.**

150 drawn uniformly, 70 drawn to top up intents the uniform draw leaves too thin to
measure. A uniform 150 gives the rarest intents about eight examples each, and a
per-class score on eight examples has an interval wide enough to be useless. But
enrichment means the label mix no longer matches traffic, so the strata are recorded
separately and only the uniform 150 estimates what real traffic looks like.

**11. The enrichment signal comes from cluster centroids, not intent definitions.**

Matching messages against the text of each intent definition was tried first and was
much worse: it put 87 of 220 messages under one intent whose definition happens to sit
near the centre of everything. Embedding models compare messages to messages well and
messages to abstract category prose badly. Centroids are averaged real messages, so
the comparison stays message to message.

**12. Which stratum an example came from is kept out of `data/gold/`.**

The labelled file holds only what a human typed. The stratum was chosen by an
embedding model, so it lives in the sampling record and is joined at load time. That
keeps one claim absolutely true: nothing that has seen model output has written into
the golden set. A test enforces it.

**13. The route label is deliberately not a function of the intent.**

If "billing always escalates" were the whole rule, route accuracy would be intent
accuracy wearing a different name and the escalation numbers would measure nothing.
The labeller answers a question about the specific message instead: could a reply
built from this brand's past replies actually resolve this? "Getting error P-DEV320
on my Roku" and "third time I'm writing about P-DEV320" share an intent and do not
share an answer.

**14. The repeat-contact rule is the broad version, and it is expensive.**

Anyone who says they already tried something, or already wrote in, goes to a human.
That is about one message in ten. The narrower reading, only escalating on explicit
prior contact, keeps more coverage but could not be applied the same way twice: "I
restarted my router" and "I tweeted you last week" shade into each other, and a rule
a human cannot follow consistently is not a rule. The cost of the broad version is
stated rather than hidden.

**15. The judge answers yes or no, not one to five.**

Nobody can say what separates a 3 from a 4, two raters will not draw that line in the
same place, and a scale that cannot be applied consistently produces an agreement
figure that flatters itself. "Would you let this send unedited" is a decision a
support lead makes dozens of times a day, and it is the decision the system actually
has to make.

**16. A second label axis was added during labelling, and the taxonomy was not touched.**

Hand-labelling surfaced that the induced taxonomy captures what a message is *about*
while what decides whether a bot can handle it is what the customer *wants*. 57% of
the golden messages are asking a question. Clustering could not have found this: it
groups by subject because subject is what the words carry. Rather than replace the
taxonomy and lose the proof that the label set predates any model output, the second
axis is recorded alongside it and used for analysis only.

**17. Both baselines make zero model calls.**

One guesses the commonest intent and sends the same hand-written line to everyone.
The other finds the nearest past message and sends whatever the brand replied then.
Neither costs a token. That is the point: the report has to show the classifier and
the drafter buy something that plain search does not already provide.

**18. The agent reads one message, and the cost of that was measured rather than assumed.**

36.7% of customer messages in this corpus are follow-ups rather than openings, and
39.7% of those cannot be understood alone: "still broken", "on roku", "yes". So
roughly one inbound message in seven is unclassifiable without its thread. That is a
real hole and the report states the number.

It was not filled, for a reason that is about evidence rather than effort. The
golden set is 220 opening messages, so there is no labelled follow-up anywhere in
the project and no way to show that thread context helps. Shipping an unmeasured
feature into a submission whose stated thesis is that the proof matters more than
the system would be the wrong trade.

Worth noting what the fix is not. A whole conversation here runs to a median of 42
words and 207 at the 99th percentile, so the entire thread fits in a prompt. There
is nothing to summarise, window, or store, and a conversation-memory component would
be machinery for a problem this data does not have.

**19. No orchestration framework, and no vector database.**

The model layer is one function that checks a cache, refuses to go live during
replay, retries, validates the schema and records spend, in about 120 lines. A
framework would wrap that in abstractions chosen by someone else, and the brief says
the code has to be explained and modified live.

The index is 9,859 vectors of 768 dimensions, 29MB, searched with one numpy dot
product in a few milliseconds. FAISS or a hosted vector store earns its place at a
million vectors; here it would be a dependency, a service and a failure mode in
exchange for nothing.

**20. `eval --replay` reads committed run records rather than re-running anything.**

Re-running the pipeline against a response cache would still need embeddings, which
needs a local model server. Reading the recorded runs needs nothing but the files in
the repo, which is what a reviewer cloning it actually has.
