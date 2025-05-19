# -*- coding: utf-8 -*-
"""Base Reward Function Class."""
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from trinity.utils.eval_utils import (
    evaluate_equation,
    extract_solution,
    simple_answer_parser,
    validate_equation,
)
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

import dashscope
import json
import os
from trinity.common.elem_prompts import reward_prompt_v1a as reward_prompt

logger = get_logger(__name__)


REWARD_FUNCTIONS = Registry("reward_functions")


class RewardFn(ABC):
    """Base Reward Function Class."""

    # TODO: add a batch version

    @abstractmethod
    def __call__(
        self,
        response: str,
        prompt: Optional[str] = None,
        truth: Optional[str] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[float, dict]:
        """Call the reward function."""


@REWARD_FUNCTIONS.register_module("accuracy_reward")
class AccuracyReward(RewardFn):
    """A reward function that rewards correct answers.
    Ref: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
    """

    def __init__(self, answer_parser: Optional[Callable[[str], str]] = None):
        self.answer_parser = answer_parser

    def __call__(
        self,
        response: str,
        prompt: Optional[str] = None,
        truth: Optional[str] = None,
        return_dict: Optional[bool] = False,
    ):
        if self.answer_parser:
            answer_parsed = self.answer_parser(response)
            truth_parsed = self.answer_parser(truth)  # type: ignore [arg-type]

        else:
            truth_parsed = parse(
                truth,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(truth_parsed) == 0:
                truth_parsed = truth

            answer_parsed = parse(
                response,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            reward = float(verify(answer_parsed, truth_parsed))
        except Exception as e:
            logger.info(f"verify failed: {e}, answer: {answer_parsed}, gold: {truth_parsed}")
            reward = 0.0
        return reward


@REWARD_FUNCTIONS.register_module("format_reward")
class FormatReward(RewardFn):
    """A reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags.
    Ref: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
    """

    def __init__(self, pattern: Optional[str] = None):
        self.pattern = pattern if pattern else r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"

    def __call__(
        self,
        response,
        prompt: Optional[str] = None,
        truth: Optional[str] = None,
        return_dict: Optional[bool] = False,
    ) -> float:
        if re.match(self.pattern, response, re.DOTALL | re.MULTILINE):
            return 0.1
        else:
            return -0.1


@REWARD_FUNCTIONS.register_module("math_reward")
class MathRewardFn(RewardFn):
    """A reward function that rewards for math task."""

    # DEFAULT_FORMAT_PATTERN = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    DEFAULT_FORMAT_PATTERN = r".*?<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    DEFAULT_ANSWER_PARSER = simple_answer_parser

    def __init__(
        self,
        answer_parser=DEFAULT_ANSWER_PARSER,
        pattern=DEFAULT_FORMAT_PATTERN,
    ) -> None:
        self.accuracy_reward = AccuracyReward(answer_parser)
        self.format_reward = FormatReward(pattern)

    def __call__(  # type: ignore
        self,
        response: str,
        prompt: Optional[str] = None,
        truth: Optional[str] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[float, dict]:
        accuracy_score = self.accuracy_reward(response, prompt, truth)

        format_score = self.format_reward(response, prompt, truth)

        if return_dict:
            return {"accuracy": accuracy_score, "format_score": format_score}

        return accuracy_score + format_score


@REWARD_FUNCTIONS.register_module("countdown_reward")
class CountDownRewardFn(RewardFn):
    """A reward function that rewards for countdown task."""

    def __init__(self):
        pass

    def __call__(
        self,
        response: str,
        prompt: Optional[str] = None,
        truth: Optional[str] = None,
        return_dict: Optional[bool] = False,
    ) -> float:
        # Copy from Jiayi-Pan/TinyZero verl/utils/reward_score/countdown.py
        truth = json.loads(truth)  # type: ignore
        target = truth["target"]  # type: ignore
        numbers = truth["numbers"]  # type: ignore

        solution_str = response
        equation = extract_solution(solution_str=solution_str)
        format_score = 0.1
        score = 1.0

        if equation is None:
            return 0

        # Validate equation uses correct numbers
        if not validate_equation(equation, numbers):
            return format_score

        # Evaluate equation
        try:
            result = evaluate_equation(equation)
            if result is None:
                return format_score

            if abs(result - target) < 1e-5:  # Account for floating point precision
                return score
            else:
                return format_score
        except Exception as e:  # noqa: F841
            return format_score


@REWARD_FUNCTIONS.register_module("elem_reward")
class ElemRewardFn(RewardFn):
    """A reward function that rewards for medical querying task."""

    def __init__(
        self,
    ) -> None:
        self.stream = False
        self.model_name = "qwen2.5-32b-instruct"
        self.enable_thinking = False
        self.temp = 0.7
        self.sys_prompt = reward_prompt

    def __call__(  # type: ignore
        self,
        answer: str,
        prompt: Optional[str] = None,
        truth: Optional[str] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[float, dict]:
        time.sleep(0.1)
        decision_score, matching_score = self.llm_reward(answer, prompt, truth)

        if return_dict:
            return {"decision_score": decision_score, "matching_score": matching_score}

        return decision_score + matching_score

    def llm_reward(self, answer, prompt, truth):
        messages = [
            {"role": "system", "content": self.sys_prompt.format(truth)},
            {"role": "user", "content": answer},

        ]
        # logger.info(f"Truth:\n{truth}")
        # logger.info(f"Rollout:\n{response}")
        try_count, max_retries = 0, 5
        while try_count <= max_retries:
            try:
                response = dashscope.Generation.call(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    model=self.model_name,
                    messages=messages,
                    stream=self.stream,
                    result_format='message',
                    enable_thinking=self.enable_thinking,
                    temperature=self.temp
                )
                if self.stream == False:
                    content = response["output"]["choices"][0]["message"]["content"]
                else:
                    full_content = ""
                    for chunk in response:
                        full_content += chunk.output.choices[0].message.content
                    content = full_content

                decision_score, matching_score = 0.0, 0.0
                pattern = r"<(\w+)>(.*?)</\1>"
                matches = re.findall(pattern, content)
                for tag_name, content in matches:
                    if tag_name == "think":
                        think = content
                    if tag_name == "decision_score":
                        decision_score = float(content)
                    if tag_name == "matching_score":
                        matching_score = float(content)
                logger.info(f"try={try_count}, reward for {answer} = {decision_score}, {matching_score}.")
                return decision_score, matching_score
            except Exception as e:
                try_count += 1
                if try_count > max_retries:
                    logger.warning("retried too many times, abort task.")
                    raise  # 抛出最后一次的异常
                else:
                    logger.warning(
                        f"error: {e}, response:{response}, retries: {try_count}")
                time.sleep(try_count * 1)



