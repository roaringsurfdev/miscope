### Housekeeping tasks and requirements
*This is a space for tracking tasks and requirements that help keep the codebase clean and healthy*

## README.md and other project support documents are out of date.
The new fieldnotes journal has a link to the github repo. As I continue to post and share to fieldnotes, there's increasing chance that traffic might flow to the repo. I'd like to make sure that README.md and DOMAIN_MODEL.md are up to date. I'd also like to make the demo notebooks more prominent so that they're easier to find for someone coming to the repo for the first time.

## Node20 deprecated
Migrate CI Action(s) to use Node24

From GitHub:

>Node20 will reach end-of-life (EOL) in April of 2026. As a result we have started the deprecation process of Node20 for GitHub Actions. We plan to migrate all actions to run on >Node24 in the fall of 2026.
>
>The newest GitHub runner (v2.328.0) now supports both Node20 and Node24 and uses Node20 as the default version. If you’d like to test Node24 ahead of time, set >FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true as an env in your workflow or as an environment variable on your runner machine to force the use of Node24.
>
>Beginning on June 2nd, 2026, runners will begin using Node24 by default. To opt out of this and continue using Node20 after this date, set >ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION=true as an env in your workflow or as an environment variable on your runner machine. This will only work until we upgrade the runner >and remove Node20 later in the fall of 2026.

## Improve pyright and ruff workflow
We are currently running into repeated failed CI Runs due to ruff and pyright failures. Before pushing to remote, we need to make sure ruff and pyright errors are cleared/addressed.

## Out-of-date requirements

**REQ_015** (Checkpoint Editor Using Train/Test Loss Curve) — Cancelled 2026-03-30. Superseded by REQ_083. REQ_015 addressed analysis checkpoint *selection* for an already-trained model and was written against Gradio. The real problem is retraining with a denser schedule to capture events like second descent. Do not resurface REQ_015.
