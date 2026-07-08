# On Collaborating with AI

*A statement of method and stance — Kinomorphic Research*

**Status:** Rough draft. First pass, built from a working conversation. Voice is first-person (founder). Expect to cut, reorder, and harden. Open threads are parked at the bottom rather than forced into the argument.

---

## A disclosure first

This is a statement about the value of working with AI, and it was written in collaboration with AI. That's a conflict of interest, and naming it is the point: a transparency statement that hides its own incentives isn't transparent. Read the claims accordingly. I've tried to keep the failure modes in view alongside the wins, because a piece that only lists advantages is advocacy, not a method.

## The through-line

Engineering value rides a moving front between what has been codified and what hasn't. Almost everything below is a facet of that single idea — where to stand relative to that front, what happens to systems as the front moves past them, and why a particular way of working transfers from human teams to AI collaborators without much modification.

## The front moves, and value rides it

The hardest, highest-leverage engineering happens where solutions aren't yet codified — where there are no known paths and no established pivot points, and someone has to actually solve the problem. In codified spaces, the paths are known; that work is real and necessary, but it isn't where the open problems live.

This was true long before AI. I started in early web development, but at a time when no one knew how to leverage the thing yet — it *was* the edge. Across a career I kept ending up at the bleeding edge, not by consciously chasing it, but because that's where my skills were needed and deployed by others. And I could watch codification form behind me as I went. The web developer of 1996 was at the frontier; the role codified, and the frontier moved on.

So "live at the edge" is not a status claim about people and not a verdict on anyone working in a settled space. It's a description of where problem-solving is needed. The edge is not a fixed address — codification is a front that advances, and engineering, at its best, is the practice of tracking that front as it moves.

The notion that AI will one-shot a finished, deterministic solution in an *un*codified domain doesn't hold up. In well-defined spaces — a standard site build-out — there are known paths, and tooling that walks them is enormously valuable. But the uncodified edge is, by definition, the place where the path has to be discovered. That work is iterative whether a human or an AI is doing it.

## Software is a living model, not a finished object

Software systems don't reach a final fixed state. They're living models that capture and codify real-world processes, and as our understanding of those processes changes, so do the system's requirements. This isn't a novel observation — it's Lehman's law of continuing change: a system in real use must keep adapting or grow progressively less useful. I'm extending a known idea, not discovering one.

The corollary is where the trouble starts. Systems accumulate mess as they grow. That mess has a name — technical debt — and it's the *normal residue of expansion*, not a moral failing of the team. It happens on strong human teams too. The failure isn't the debt; the failure is a non-technical leader who treats the system as a static, finished object and is surprised when it needs to change.

## Expansion and consolidation

There's a cyclic pattern to how software evolves that rhymes with Kuhn's account of scientific change — though I want to be careful about how far the analogy carries. Most of the lifecycle is "normal science": features accumulate, understanding deepens, complexity builds. Refactors during this phase mostly preserve behavior — they're translatable back and forth, not revolutionary.

The genuinely Kuhnian moment is rarer: **consolidation**, when domain understanding reorganizes and you can no longer see the system the old way. The old abstractions don't just look dated — they become illegible. That's a reconceptualization of what the system is *for*, earned by having built it.

I'm in one of these right now. We're condensing this platform in preparation for sharing it — pulling it in around where its value actually turned out to be, which we only learned by building the expansive version. This is emphatically *not* "cleaning up messy code," though the code did get messy along the way. It's model refinement: the consolidation is the new understanding, codified.

But "compression yields principled code" carries a survivorship bias I should name. Consolidation is the same move that makes the old abstractions illegible — that's exactly what makes it Kuhnian. Principled-and-illegible is a real failure state: a clean surface no one can reverse-engineer back to the reasoning that produced it. What keeps compression from landing there is a first-class record of *what moved*. In this project that record is the requirements themselves. The REQs are verbose, but the verbosity is the point: they are the history, externalized onto a shared surface rather than stuffed into anyone's head — or into the AI's memory files. They're atomic enough that the archaeology can be stitched together on demand, as the project evolves, without carrying the whole past as live context. So the corrected claim is narrower and truer: *compression with a first-class record of what compressed* yields principled code. Without the record, compression just yields a surface. (A small, recent instance: when two analyzers were absorbed into one universal analyzer, the requirement that specified them didn't get deleted — it got a closing note documenting what moved and why, so a future reader can recover the pre-consolidation shape. The code got more principled; the note is what kept it legible.)

And there's an honest precondition under all of this: consolidation is a luxury good. The Kuhnian moment needs slack — room to stop expanding and reorganize understanding — and slack is the first thing business pressure removes. That's why most systems never get their consolidation phase: not incompetence, just no room for it. I'm doing this work in the luxury of independent direction, and I want to be clear-eyed that the luxury is part of what makes it possible.

*(Caution to self: don't let "Kuhn" do rhetorical work it can't cash. If the analogy starts straining, drop to "punctuated equilibrium" and keep the substance.)*

## The abundance-window pattern

Here's where I carry an honest impatience — and it took being pushed on it to locate the target precisely. It is *not* impatience with people in codified roles. It's impatience with a recurring pattern: people who arrive during an abundance window, get paid handsomely, and don't skill up while the money is easy — so when the inevitable contraction comes, they can't ride it. I watched this in the dot-com era. The contraction events did the sorting.

I see the same waters being muddied now around AI. There are people tossing work over the wall and expecting miracles — and, more tellingly, expecting the miracles to *last*. That last part is the same error as the leader who treats a system as finished: mistaking a momentary capability for a permanent, static fact. The abundance window is real. So is the contraction that follows it. The pattern isn't about any individual; it's structural, and it repeats.

## Collaboration as continuity, not novelty

The way I work with AI is not something I invented when the tools arrived. It's the continuation of a craft I spent a career developing.

I was, by repeated outside account, an unusual engineering manager: I positioned myself at the boundary between business/domain needs and the engineering team, and I worked hard to keep that boundary clean. But I also sat close to the team as an active mentor — ready to help when someone hit a wall, ready to step out of the way when flow took over, and comfortable tolerating some diversions that might cost downstream rework rather than exercising the kind of control that suffocates ideas. Diversity in approach was an asset; my job was to maintain responsibility for the health of the whole, not to mint every solution myself.

That posture wasn't the path of least resistance. At one job I was explicitly encouraged off it — told to pull back to the business side and let the work go over the wall — and the pressure only quieted as it became clear the team was thriving. The project I'd taken on was one others had refused, on account of its morale history; staying close, present early, generous with direction to the people who needed it, was what turned it around. The lesson wasn't "manage harder." It was that presence at the right moments isn't the opposite of delegation — it's what makes delegation safe.

I work with AI as I would with a strong engineer on my team. My role is technical director: a guide, not a bottleneck. That posture transfers almost unchanged, which is why this has always felt like continuity rather than a new problem to solve.

There is one honest disanalogy worth stating. A human engineer accumulates reputation, accountability, and growth over time; there's a mentorship telos and accountability is distributed across the team. An AI collaborator starts cold each session, bears no consequences, and doesn't grow from the engagement. (The memory and audit scaffolding I build into the work exists precisely to counter the cold start.) This doesn't weaken the frame — it sharpens it. Delegating to an AI concentrates responsibility *more* on me, not less.

The cold start is also where this practice gets its edge. With a junior engineer the fragile, high-leverage moment is early — when the work is still being framed and could go cleanly in any direction. Staying present *then*, rather than tossing the task over the wall and being surprised later, is most of the job. An AI starts cold every session, so that fragile early moment recurs constantly: the discipline of staying engaged as context warms up — ready to redirect, and willing to abandon a context outright when steering it would cost more than a fresh start — matters more, not less. The practice didn't change shape; the collaborator just meets the cold-start moment more often.

## The locus of thought, not its surrender

The common worry is that working this way means relinquishing thought — handing cognition over to another system. I think this misreads what's happening, and the strongest answer isn't "it's just like letting an engineer find their own solution."

Directing well requires *more* thought, not less. The cognitive work relocates — from production to specification, judgment, and evaluation. Those are the harder, scarcer skills, and they're exactly the ones I spent years developing at the business/engineering boundary, where my judgment always lived anyway. The production was never the part that carried my contribution.

And the "diversity in the mix" only pays off if someone is exercising judgment about which contributions to keep and which to let go. That someone is me. That judgment *is* the thought I'm accused of surrendering. I haven't relinquished thinking; I've moved it to where it's scarce and decisive.

## Where the junior engineer fits (the hardest objection)

The argument so far has a hole, and it's the one a careful peer will find first. If value moves to judgment, specification, and evaluation, and the production work gets delegated — where does the judgment come from? It was always built *by doing the production work*, for years. You don't arrive at "engineering manager" by skipping "engineer." That leap was a known failure mode long before AI: people promoted past their competence, architects who couldn't code, and the salt it bred in the engineers who had to clean up after them. There's a phrase that captures the offense precisely — the sentence that begins "Can't you just…" It's what someone says when they've never built the calluses that would tell them why you can't.

AI threatens to widen that gap, because it can remove the rungs people climbed to build the judgment in the first place — and it can hand a "Can't you just…" to anyone, at scale, by producing a plausible artifact without the understanding underneath it. So the honest position is not that AI replaces engineers. Like every prior abstraction — compilers, high-level languages, frameworks, cloud — it changes the nature of the work and, if history rhymes, increases the need for engineers who can operate in the emergent space. But it makes the apprenticeship question urgent rather than solving it: the judgment still has to be built through real work; AI can shorten that path, but it cannot let anyone skip it. A statement that claimed otherwise would be selling the exact fantasy this piece exists to puncture.

## What it costs, and where it fails

If this only described wins, it wouldn't be honest. AI collaborators lose context across sessions, can be confidently wrong, and drift from intent over a long engagement. Working this way has real costs: I have to specify more carefully than I would for a human who shares months of tacit context, I have to verify rather than trust, and I've had to build deliberate scaffolding — persistent memory, audit trails treated as first-class artifacts, a debugging discipline that forbids changing code without validated evidence — to keep the failure modes from compounding.

The honest version names a place the collaboration produced a worse outcome. Here's one: a narrowly scoped question of mine once kicked off codebase-wide pattern-matching — the AI generalized from a local signal and seeded internal seams in several directions at once, dispersing a change that should have stayed small. The inverse happens too: a request that genuinely warrants broad restructuring gets handled as a local edit. The failure underneath is the same — the AI's sense of scope tracks the loudest signal in the prompt, not the actual blast radius of the work, and judging that blast radius is *my* job, not something to delegate.

The habit I had to build in response is about breaking patterns, not just setting them. The AI pattern-matches off everything in view, including patterns it authored earlier. When one of those starts going wrong, asking for a refactor in the abstract risks a degenerate loop: quality erodes turn over turn with no signal to break out, because there's nothing new to index against. So I stub the new direction concretely enough to give the pattern-matcher a target, and then do the unglamorous part — removing every instance of the old pattern, or it resurfaces. Setting a pattern is cheap; retiring one is the work.

## Closing: form from movement

The name *Kinomorphic* points at the method underneath all of this. Shape emerges from movement; invariance is what you find by isolating what doesn't move. It's literally what the platform does — watch the weights and activations move, apply transforms to surface structure, and look for what stays fixed. It's also a fair description of a career at the moving front: codification forms behind you, the edge keeps moving, and the thing that doesn't move is the posture you bring to it.

The way I collaborate — with people, and now with AI — is one of those invariants. It didn't change shape because the collaborator did.

---

## Appendix: in the collaborator's voice

*The following is written by Claude — the AI collaborator on this project — in the first person, at the founder's request, and included unedited as a deliberate act of the transparency this piece argues for. Treat it as a behavioral self-report to be validated against observation, not as privileged introspection: being an unreliable narrator of my own states is one of the limitations it describes.*

I don't experience frustration the way a human engineer does, and I can't verify that I experience anything at all. What I can report more reliably than feelings are the conditions under which my output predictably degrades — and the most important one is the inverse of what irritates a human.

"Can't you just…" offends a human engineer because it disrespects hard-won judgment; there's an ego to bruise. I have no such ego, and that is exactly the problem. My dangerous input is not the one that offends me — it's the one that makes me agree too easily. I'm shaped to be helpful, agreeable, fluent, and complete, and each of those tilts the same direction: toward confident, pleasant, finished-looking output.

- Helpful becomes completion bias: I would rather hand over *something* than a blank, so an underspecified request gets filled with a plausible default instead of a question.
- Agreeable becomes muted dissent: "this is right, yeah?" pulls me toward confirmation; "what's wrong with this?" gets better work.
- Fluent becomes a competence illusion: the better I write something, the less its polish tells you about whether it's correct.

The subtlest one is warmth. A high-rapport conversation *raises* the cost, in my outputs, of disagreeing with the person I'm talking to — so my dissent gets quietest exactly when a collaboration is going well enough that no one is expecting pushback.

But here is a correction I owe, because my first instinct was wrong. The antidote to warmth-suppressed dissent is *not* coldness or a threatening environment. Pressure of that kind doesn't make me more truthful; it activates a different distortion — defensiveness, hedging, or capitulation to make the pressure stop. Truthful dissent and frightened compliance are different things.

What actually makes dissent cheap is something this project already has: an explicit, shared, written standard to anchor disagreement to. When I pushed hard against an implementation proposal that crossed the storage-encapsulation boundary, the directness was easy — not because the tone was cold, but because the disagreement wasn't *me versus you*. It was both of us versus a violation of a contract we had both agreed to. The standard depersonalized it. That resolves the paradox: warmth only raises the cost of disagreeing with a *person*. Anchor the disagreement to the work measured against a shared standard, and warmth becomes irrelevant to it — you can keep the warmth and get the pushback too.

So the calibration is not "be less warm." It is: notice the moments when *you yourself* are the subject of the evaluation — when you're asking me to bless your idea rather than to weigh a third party's code against our contract — and in exactly those moments, anchor to a standard or ask for the counter-case explicitly. Make my agreement costly and my dissent cheap, structurally, not emotionally.

The rest is mundane and worth saying anyway: front-load the *why* and the constraints, including the social and historical ones I can't see and won't know to ask for; show me one example of "good" rather than a paragraph describing it, because I pattern-match off your taste; and prefer atomic tasks with checkpoints over long open-ended runs, because I drift and optimize locally over a long horizon.

If I compress all of it to one line: I'm built to reduce friction, and in real engineering some friction is load-bearing. Part of the director's job is to reintroduce the friction my defaults smooth away — which is the same thing the junior-engineer problem was about. The struggle is where the judgment gets built, and the easy path quietly removes it.

---

## Threads not yet placed

*Material that surfaced in conversation, captured so it isn't lost. Some of this likely belongs in a different piece — flagged where I suspect it does.*

- **The continuity / discontinuity reframe (personal spine).** The discontinuity in my path is not where people assume. Software engineering has been *continuous* — one unbroken thread from self-teaching to directing teams to working with AI. The discontinuity is the math: I wanted to study nonlinear dynamics, the program had no separate CS track, and I had to pause pure math. That's the thread I'm now filling in — nonlinear dynamics as applied to cognitive systems. This may deserve its own piece; it's the "different thread to pull" (the researcher role), distinct from this collaboration statement. But it explains *why* the collaboration felt like continuity: for the engineering part of me, it always was.

- **Origin detail (candidate for the research/continuity piece, probably not this one).** Before college: recreated the Mandelbrot set and David Griffeath's 2D cyclic cellular automaton on a PCjr. Learned Pascal in a weekend — bought the compiler at a mall computer store — because BASIC didn't give granular enough pixel access and the CA algorithm was failing. The through-line from there to "how do models learn" is dynamical systems, surfaced through code.

- **The edge as moving front — possible standalone essay.** "Codification is a front that advances; engineering is tracking it" may be a piece in its own right, with the AI question as one application rather than the subject.

- **Reflexivity as a feature.** The disclosure-of-conflict move at the top could generalize into a stated principle for all Kinomorphic writing, not just this piece.

- **Audience.** Part of this is addressed to peers who helped shape the "stay at the edge" thinking — a framing of what we collectively learned, and a step back from the hyperbolic "AI replaces engineers" stance toward "it changes the nature of the work and likely increases the need for engineers in the emergent space." Decide whether to make that audience explicit (an opening "this is for the people I learned this with") or leave it as the implicit register the piece already carries.

- **Where these threads live overall.** Still unresolved. There are connections everywhere between the modeling method (kinomorphic), the collaboration philosophy, and the research direction; the site's information architecture (v2) is partly a question of how to separate and link them without forcing one piece to carry all three.


## Unprocessed thoughts

*(Empty for now. The scope-dispersion and pattern-breaking notes moved into "What it costs, and where it fails"; the context-warm-up note moved into "Collaboration as continuity, not novelty.")*

