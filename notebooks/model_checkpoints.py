# %% checkpoint definitions
DEFAULT_CHECKPOINT_EPOCHS = sorted(
    list(
        set(
            [
                *range(0, 1500, 100),  # Early training - dense (~15 checkpoints)
                *range(1500, 9000, 500),  # Mid training - moderate (~15 checkpoints)
                *range(9000, 13000, 100),  # Grokking region - dense (~100 checkpoints)
                *range(13000, 25000, 500),  # Post-grokking - moderate (~24 checkpoints)
            ]
        )
    )
)
CHECKPOINT_EPOCHS_109_485 = sorted(
    list(
        set(
            [
                *range(0, 6000, 100),  # Early and Mid training combined - dense
                *range(6000, 25000, 500),  # Mid training - moderate
            ]
        )
    )
)
CHECKPOINT_EPOCHS_107_485 = sorted(
    list(
        set(
            [
                *range(0, 1500, 100),  # Early training - dense (~15 checkpoints)
                *range(1500, 9000, 500),  # Mid training - moderate (~15 checkpoints)
                *range(9000, 13000, 100),  # Canon Grokking region - dense (~100 checkpoints)
                *range(13000, 17000, 500),  # Post-grokking - moderate (~24 checkpoints)
                *range(17000, 21000, 100),  # 107/485 Grokking region - dense (~100 checkpoints)
                *range(21000, 25000, 500),  # Post-grokking - moderate (~24 checkpoints)
            ]
        )
    )
)
CHECKPOINT_EPOCHS_101_485 = sorted(
    list(
        set(
            [
                *range(0, 1500, 100),  # Early training - dense (~15 checkpoints)
                *range(1500, 9000, 500),  # Mid training - moderate (~15 checkpoints)
                *range(9000, 18000, 100),  # Canon Grokking + late grokking region - dense (~100 checkpoints)
                *range(18000, 25000, 500),  # Post-grokking - moderate (~24 checkpoints)
            ]
        )
    )
)
CHECKPOINT_EPOCHS_59_485 = sorted(
    list(
        set(
            [
                *range(0, 1500, 100),  # Early training - dense (~15 checkpoints)
                *range(1500, 9000, 500),  # Mid training - moderate (~15 checkpoints)
                *range(9000, 13000, 100),  # Grokking region - dense (~100 checkpoints)
                *range(13000, 35000, 500),  # Post-grokking - moderate (~24 checkpoints)
            ]
        )
    )
)
CHECKPOINT_EPOCHS_59_485_DENSE_INTERVENTION = sorted(
    list(
        set(
            [
                *range(0, 6500, 100),  # Early training - dense (~15 checkpoints)
                *range(6500, 9000, 500),  # Mid training - moderate (~15 checkpoints)
                *range(9000, 13000, 100),  # Grokking region - dense (~100 checkpoints)
                *range(13000, 35000, 500),  # Post-grokking - moderate (~24 checkpoints)
            ]
        )
    )
)

# %% print definition for copy/paste into training run config
print(CHECKPOINT_EPOCHS_59_485)
#print(len(CHECKPOINT_EPOCHS_59_485_DENSE_INTERVENTION))
# %%
