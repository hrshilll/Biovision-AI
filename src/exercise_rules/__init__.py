"""Exercise rule engine registry."""

from __future__ import annotations

from typing import Dict, Optional

from src.exercise_rules.base import ExerciseRuleEngine
from src.exercise_rules.curl_rules import CurlRuleEngine
from src.exercise_rules.deadlift_rules import DeadliftRuleEngine
from src.exercise_rules.lunge_rules import LungeRuleEngine
from src.exercise_rules.pushup_rules import PushupRuleEngine
from src.exercise_rules.squat_rules import SquatRuleEngine

RULE_ENGINES: Dict[str, ExerciseRuleEngine] = {
    "squat_rules": SquatRuleEngine(),
    "deadlift_rules": DeadliftRuleEngine(),
    "pushup_rules": PushupRuleEngine(),
    "curl_rules": CurlRuleEngine(),
    "lunge_rules": LungeRuleEngine(),
}


def get_rule_engine(module_name: Optional[str]) -> Optional[ExerciseRuleEngine]:
    if not module_name:
        return None
    return RULE_ENGINES.get(module_name)
