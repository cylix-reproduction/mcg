"""CLI entrypoint for the refactored MCG attack pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import punq
import yaml
from dacite import from_dict
from loguru import logger

from mcg.config import MetaConditionalAttackParams
from mcg.domain import inject as domain_inject
from mcg.domain.attack import MetaConditionalAttack
from mcg.infrastructure import inject as infra_inject


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_container(params: MetaConditionalAttackParams) -> punq.Container:
    container = punq.Container()
    container.register(MetaConditionalAttackParams, instance=params)
    infra_inject(container)
    domain_inject(container)
    return container


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the refactored MCG attack pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mcg_attack.yaml"),
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    params = from_dict(MetaConditionalAttackParams, cfg)

    container = build_container(params)
    orchestrator: MetaConditionalAttack = container.resolve(MetaConditionalAttack)

    outcome = orchestrator.run()
    logger.info("Finished with success={} mean_query={}", outcome.success, outcome.queries)


if __name__ == "__main__":
    main()
