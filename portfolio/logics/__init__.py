import os
import importlib

strategies = {}
for strategy_name in os.listdir("logics"):
    if (
        ("__" not in strategy_name)
        & (".py" in strategy_name)
        & ("template" not in strategy_name)
    ):
        strategy_key_name = strategy_name.split(".")[0]
        print(f"Import {strategy_key_name}")
        strategies[strategy_key_name] = importlib.import_module(
            f"logics.{strategy_key_name}"
        ).run

# print(strategies)
# del strategy_key_name
