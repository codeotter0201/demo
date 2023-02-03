import os
import importlib


def get_strategies(test: bool = False) -> dict:
    strategies = {}
    if not test:
        for strategy_name in os.listdir("logics"):
            if (
                ("__" not in strategy_name)
                & (".py" in strategy_name)
                & ("template" not in strategy_name)
            ):
                strategy_key_name = strategy_name.split(".")[0]
                print(f"Import {strategy_key_name}")
                method = importlib.import_module(f"logics.{strategy_key_name}").run
                if method.config.test_mode:
                    continue
                strategies[strategy_key_name] = method
    else:
        for strategy_name in os.listdir("logics"):
            if (
                ("__" not in strategy_name)
                & (".py" in strategy_name)
                & ("template" not in strategy_name)
            ):
                strategy_key_name = strategy_name.split(".")[0]
                print(f"Import {strategy_key_name}")
                method = importlib.import_module(f"logics.{strategy_key_name}").run
                if not method.config.test_mode:
                    continue
                strategies[strategy_key_name] = method
    return strategies
