# SPOC / ChefOST

SPOC research for object-state tracking in procedural cooking videos.

## Shared guidance

- [Organization guide](https://github.com/chefOST/.github/blob/main/profile/README.md).
- [Project goals](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md#project-goals) and [paper notes](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md#2-spoc).
- [Evaluation definitions](https://github.com/chefOST/.github/blob/main/docs/Evaluation.md).
- [Contribution rules](https://github.com/chefOST/.github/blob/main/CONTRIBUTING.md) and [agent instructions](AGENTS.md).

## Data

Get annotations and vocabularies from [chefOST-data](https://github.com/chefOST/chefOST-data). Get videos from the shared [R2 bucket](https://github.com/chefOST/chefOST-data#videos).

## Outputs

Log runs and outputs to the `chefOST` W&B projects:

- Experiments: [`spoc-experiments`](https://wandb.ai/chefOST/spoc-experiments).

Use the exact destinations in [settings.json](weights_and_biases/settings.json).
