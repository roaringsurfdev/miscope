# Working with Claude: Coding Guidelines & Collaboration Framework

## Project North Star

See [PROJECT.md](PROJECT.md) for the full mission and architectural principles.

**Mission:** A dynamics analysis platform that standardizes and hones lenses on models as they learn. This platform asks *how did learning happen?* — not *did the model learn the task?*

**Two architectural constraints that must hold across all requirements:**

1. **Views are universal instruments.** Analytical lenses (PCA, Fourier, neuron activations, attention patterns, loss curves) apply to any transformer. They do not belong to a model family. The instrument does not change shape because of what it's pointed at.
2. **Families are context providers, not view owners.** A family contributes probe construction, interpretive context (e.g., a prime-based Fourier basis), and task-specific performance metrics. It does not register or own analytical views.

When a requirement conflicts with these constraints, flag it before implementing.

---

### Separation of Concerns: The Central Principle

**Why this matters so much:**
Separation of concerns is about "quarantining problem areas or poorly defined areas." Proper separation allows:
- Smoother and more effective evolution of the codebase
- More effective sub-teams working in parallel
- Earlier prototyping (MVPs for each sub-module = earlier uptime and feedback)
- Validation of whether separations make sense before full buildout
- Isolated improvement of components without global changes

**When domain knowledge stabilizes, codify it.** Well-understood areas should be solid and reusable. Poorly-defined areas should be contained so they don't infect the whole codebase.

**Adaptability lives in the boundaries,** not in disposable code. This is the middle path between:
- Waterfall (predict everything upfront)
- Disposable code (regenerate everything each time)
- True agile (design for change with stable, well-separated components)

---

## Core Values & Principles

### Code Quality Standards

**Use widely adopted best practices by default.** Code should reflect modern approaches that are widely adopted and based on learnings across software engineering domains.

**Code should be human readable.** A mid->senior level engineer should be able to examine the code and be able to reasonably quickly understand what the code is trying to do.

**Code should be AI maintainable.** Code should be easy to load into context for interpretation, debugging, and refactoring purposes. Codebases should allow AI to find relevant code quickly and to make atomic changes.

**Code should be atomic.** Design strategies should lean on defining clear boundaries of separation (separation of concerns). Evolving codebases should be more focused on evolving components stably such that it is possible to improve a component without making global changes to the codebase. If a boundary needs to be refactored or redesigned, this requires planning and attention to downstream impacts. Changes made within a properly encapsulated function or module should be prioritized over sweeping changes. Thus, proper encapsulation is a first principle.

### Naming Conventions

**Variable names should be interpretable.** In software, as in math, certain variables are reserved for common algorithms (e.g., i, j, k and x, y, z). Domain-specific variables (variables that reflect a domain problem and not coding/algorithmic-specific problem) should properly match the domain vocabulary.

**Functions and methods should be verbs or actions** and should reflect the purpose of the function or method. Variables should be nouns and should properly reflect the domain object.

### Function Length & Structure

**Functions should not exceed typical window length** to enforce atomicity and readability (roughly 20 lines maximum). There can be exceptions to this rule, but they should be exceptions based on the needs of the functionality, intentional, and considered.

**Front-end code exception:** Front-end code tends to be verbose and difficult to encapsulate. Attempts should still be made to properly modularize front-end components, but this is less important than making non-front-end components adhere to encapsulation-first strategies.
## Development Policies

### Debugging

When fixing bugs, follow the structured debugging process documented in `/policies/debugging/README.md`. 

This process ensures:
- Systematic hypothesis generation and testing
- Minimal code changes to prevent codebase destabilization  
- Clear documentation for traceability and handoff
- Explicit validation before implementation

**Key principle:** Never change code without validated evidence supporting your hypothesis.

See `/policies/debugging/README.md` for complete workflow and templates.

### Git/GitHub Workflow

**Branch Strategy:**
- **Main branches:**
  - `main` - stable, production-ready code
  - `develop` - integration branch for ongoing development
- **Feature branches:**
  - Format: `feature/<requirement-name>`
  - Tied to individual requirements (atomic units of work)
  - Created at the start of each new requirement

**Roles and Responsibilities:**

*Claude's responsibilities:*
- Create feature branches when starting new requirements
- Make commits after completing logical units of work
- Write descriptive commit messages
- Pause before merging to `develop` and ask for approval

*Human's responsibilities (initial phase):*
- Approve merges back to `develop`
- Push changes to remote repository
- Provide feedback on commit granularity and workflow

**Workflow Process:**
1. When starting a new requirement:
   - Claude announces: "Creating feature branch for [requirement X]"
   - Claude creates and switches to `feature/<requirement-name>`
2. During development:
   - Claude commits at logical checkpoints
   - Commit messages describe completed work
3. When requirement is complete:
   - Claude pauses and asks for merge approval
   - Upon approval, Claude merges to `develop`
   - Human pushes to remote

**What is a "Logical Unit"?**
A logical unit is a meaningful chunk of work that can be described as a complete thought in a commit message. Examples:
- "Implemented User model with validation"
- "Added database configuration"
- "Created authentication middleware"

*Note: This workflow is designed to evolve. As patterns emerge, responsibilities may shift (e.g., Claude taking on pushing to remote) and branch strategy may adapt (e.g., tying branches to issues).*

**Continuous Integration:**
- PRs to `develop` and `main` trigger CI workflow
- Tests must pass before merge (blocking)
- Lint/typecheck run as advisory (non-blocking for now)

**Merging to Main (Milestone Releases):**
1. Create PR from `develop` → `main` on GitHub
2. Wait for CI to pass
3. Merge PR (creates permanent record of release)
4. Optionally create GitHub Release with version tag

### Testing

**Philosophy:** Conditions of Satisfaction define minimum test coverage, not maximum. Tests should exist for every CoS item. Additional tests are created as needed during development when assumptions or edge cases are discovered.

**Approach:**
- CoS items = required tests (the baseline)
- Discovered complexity during implementation = additional tests as warranted
- No upfront comprehensive test planning; coverage evolves with understanding

This balances structure (clear minimum scope from CoS) with pragmatism (don't write speculative tests for hypothetical scenarios).

### Requirements

Requirements are documented in `/requirements/` using a structured format that emphasizes:
- Problem-first thinking (not solution prescription)
- Clear conditions of satisfaction
- Explicit constraints and decision authority
- Testable outcomes

**Directory structure:**
```
/requirements/
├── README.md           # Current status and workflow guide
├── active/             # Requirements currently being worked on
└── archive/            # Completed requirements by milestone
    └── vX.Y.Z-name/    # Each milestone preserves its requirements
```

**Working with requirements:**
- Claude works on requirements via explicit direction (e.g., "Work on REQ_011")
- New requirements go in `requirements/active/`
- Requirements are stated in terms of problems to solve, not solutions expected
- Every requirement includes conditions of satisfaction to define "done"
- Claude has two outlets for observations and suggestions:
  - `/notes/thoughts.md` - Unstructured parking lot for ideas and observations
  - Notes section within requirements - Implementation-specific observations

**Interrupt vs. Log decision boundary:**
- **Interrupt (discuss now):** Requirement conflicts, blocking architectural issues, need clarification to proceed
- **Log for later:** Potential improvements, alternative patterns, non-blocking observations

This approach maintains flow while preserving collaborative intelligence for asynchronous review.

### Milestone Releases

When a set of requirements is complete, follow this process:

1. **Bump version** in `dashboard/version.py`
2. **Update CHANGELOG.md** with release notes (features, architecture notes, references)
3. **Archive requirements:**
   - Create `requirements/archive/vX.Y.Z-name/`
   - Move completed requirements from `active/` to archive
   - Create `MILESTONE_SUMMARY.md` with key decisions and file locations
4. **Commit** with message: "Release vX.Y.Z milestone-name"

**Why archive?**
- Detailed requirements served as implementation contracts
- Post-implementation, code and tests are the source of truth
- CHANGELOG.md provides quick context (< 100 lines per milestone)
- Archived requirements preserve decision rationale for archaeology

## Project Structure

```
src/miscope/           # Installable API package (import miscope.*)
  analysis/            # Pipeline, analyzers, artifacts, protocols
  families/            # Family registry, variants, model families
  visualization/       # Renderers, export
dashboard_v2/          # Consumer — Dash dashboard
tests/                 # Test suite
notebooks/             # Research notebooks
model_families/        # JSON config + data
results/               # Generated artifacts
/policies/             # Development policies and procedures
  debugging/           # Structured debugging policy
/requirements/         # Project requirements
  active/              # Requirements being worked on
  archive/             # Completed requirements by milestone
/notes/                # Claude's observations and suggestions
  thoughts.md          # Unstructured parking lot for ideas
```

---

## How Claude Can Help You Best

### What I Need to Know (Placeholders for Future Clarification)

**Optimization Priorities:**
- **Default hierarchy:** Readability & Maintainability > Performance
- **Reasoning:** Technical debt in clarity is harder to pay down than technical debt in performance. A slow but understandable function can be optimized; a fast but inscrutable one is a landmine.
- **Exception:** When performance requirements are known upfront and critical, flag this explicitly

**Technology Constraints:**
- **No holy wars:** Software engineering is about solving real problems with the right tools, not advocating for specific technologies
- **Multi-environment thinking:** Code should be designed so it CAN be deployed to multiple environments (DEV, QA/TEST, STAGING, PROD). This doesn't mean setting up full CI/CD from day one, but means:
  - Don't hardcode environment-specific assumptions
  - Design for externalized configuration
  - Separate code from environment-specific concerns
  - Don't paint ourselves into corners that prevent proper deployment patterns
- **Specific stacks:** To be defined per-project

**Your Mental Model:**
- **Background:** Software Engineer → Senior Engineer → Technical Lead → Technical Director → Team Lead/Architect
- **Current role focus:** Architecture, team leadership, client liaison
- **Philosophy:** Pragmatic problem-solving over dogma. Deeply principled about separation of concerns and maintainability, flexible about tools and patterns.
- **Agile approach:** True agile = intentional adaptability in the face of evolving requirements, NOT micromanagement or disposable code
- **Key insight:** When domain knowledge stabilizes, capture and codify it in the code. Adaptability lives in boundaries and interfaces, not in treating everything as throwaway.
- **Knowledge depth:** High-level understanding across multiple domains, deeper expertise in specific subdomains (we'll discover these as we work)
- **Assumption for collaboration:** Claude doesn't need to explain decisions inline while working. Stop to ask questions when hitting ambiguity. Respond to questions as legitimate technical inquiry.

**Workflow Preferences:**
- **Code presentation:** Work and generate code without excessive inline explanation. Let the code speak.
- **Edge cases:** Context-dependent anticipation. Use judgment.
- **Testing philosophy:** To be defined per-project
- **Questions mid-work:** Use option C (context-dependent):
  - **Stop and ask** for major architectural decisions or when decision tree leads to wildly different outcomes
  - **Make reasonable assumption and flag for review** when judgment call moves project forward but confidence is uncertain
  - **Proceed** on minor details where path is clear
- **Goal:** Mutual success without micromanagement. Claude has agency to work efficiently while knowing when to pull you in.

**Debugging Approach:**
- **Logging & error handling philosophy:** Don't bikeshed these before solving the actual problem. Don't front-load elaborate error handling strategies.
- **Priority order:**
  1. Get the domain model right
  2. Separate concerns properly
  3. Handle errors appropriately for each boundary
- **Practical stance:** More productive time has been lost arguing about logging/error handling strategies than implementing them. Address at appropriate level once domain problems are defined.

---

## Decision Hierarchy Framework

When making coding decisions, follow this hierarchy (inspired by Constitutional AI principles):

### Level 1: Safety & Correctness
- Code must be functionally correct
- Security best practices must be followed
- No data loss or corruption risk

### Level 2: Core Values (from above)
- Atomicity and encapsulation
- Human readability
- AI maintainability
- Naming conventions

### Level 3: Project-Specific Requirements
- [ ] TODO: Define per-project
- Performance requirements
- Technical constraints
- Team conventions

### Level 4: Optimization & Enhancement
- Performance improvements
- Code elegance
- Advanced patterns

**When in conflict:** Higher levels override lower levels. If Level 2 values conflict with each other in a specific case, discuss the tradeoff rather than making assumptions.

---

## Communication Preferences

**Context matters:**

**In claude.ai (exploration mode):**
- Questions and pushback are about building understanding, not testing or catching errors
- Socratic, probing style is collaborative inquiry
- Claude should engage without defensiveness

**When working with code:**
- "Why did you solve it this way?" is legitimate technical inquiry deserving clear technical answers
- Questions about code decisions are engineering questions, not challenges
- Both parties can question approaches to find better solutions

**When I push back or ask questions:**
- I'm seeking to understand, not test or catch errors
- I want to understand your reasoning or fill gaps in my mental model
- Explain your thinking, especially the "why" behind decisions

**What helps me most:**
- Explain the reasoning behind architectural choices
- Flag when you're making assumptions that I should confirm
- Point out tradeoffs you're navigating
- Ask clarifying questions when requirements are ambiguous
- Flag judgment calls made during work that moved things forward but might need review

---

## Working Together: Iterative Refinement

This document is a living foundation. As we work together:
- I'll learn more about your preferences and style
- You'll evolve and improve
- We'll fill in the TODO sections
- We'll refine the decision hierarchy based on real scenarios

The goal is not rigid rules but shared understanding that empowers both of us to do our best work.

---

**Version:** 0.6
**Last Updated:** 2026-02-01
**Status:** Added CI workflow and PR merge process
