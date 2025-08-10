
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from search_r1.llm_agent.thinking_prompts import action_invalid_prompt

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer|action)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer|action)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            # <think> xxx </think> or <think> xxx <action> yyy </action> </think>
            elif part == "</think>" and state in ["in_think", "after_action"]:
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            elif part == "<action>" and state == "in_think":
                state = "in_action"
            elif part == "</action>" and state == "in_action":
                state = "after_action"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer", "in_action", "after_action"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def format_rating(text:str , is_revison = False) -> float:
    '''
    返回一个0-1的分数，表示文本的格式准确度
    0. tag的多样性：尽量出现多种的 tag(0.2)
    1. 成对标签的完成度：判断是否出现了tag，以及这些 tag 是否正确的 closed(0.3)
    2. 内容整洁度：所有内容必须放在 tag 里面，tag 之间只允许存在空白文本(0.2)
    3. 顺序正确度：必须按着特定的顺序（同时注意 reward hack）(0.3)
    递进策略：
        用前一个标准的完成度作为下一个的系数，以此累乘。只有当前面的完成度为 1，后面的标准才会有更强的权重。
    '''

    # 先删掉 action_invalid_prompt
    text = re.sub(action_invalid_prompt, "", text)

    # 标签的完成度：使用栈判断, 记录不合理的 tag 占总 tag 的比例
    think_stack = []
    search_stack = []
    answer_stack = []
    action_stack = []
    # 标签的多样性：使用 set 去重/总 tag 种类
    tag_set = set()
    # 内容整洁度：判断tag之间的非空白文本长度占总文本长度的比例
    content_length = 0
    tag_length = 0
    # 顺序正确度：按着顺序逻辑来判断，记录不合理的 tag 占总 tag 的比例
    tag_all = 0
    tag_with_correct_order = 0

    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return 0, None
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer|action)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end

    action_tag_set = set()

    for idx, part in enumerate(parts):
        if not part.strip():
            continue
        if re.match(r"</?(?:think|search|information|answer|action)>", part):
            if part == "<think>":
                tag_all += 1
                tag_set.add('<think>')
                think_stack.append(part)
                if state in ['start','information','after_action']:
                    state = 'in_think'
                    tag_with_correct_order += 1
            elif part == "</think>":
                tag_all += 1
                tag_set.add('</think>')
                if len(think_stack) > 0:
                    last_tag = think_stack.pop()
                    if last_tag == '</think>':
                        think_stack.append(last_tag)
                        think_stack.append("</think>")
                if state in ['in_think', 'after_action']:
                    state = 'after_think'
                    tag_with_correct_order += 1
            elif part == "<search>":
                tag_all += 1
                tag_set.add('<search>')
                search_stack.append(part)
                if state == 'after_think':
                    state = 'in_search'
                    tag_with_correct_order += 1
            elif part == "</search>":
                tag_all += 1
                tag_set.add('</search>')
                if len(search_stack) > 0:
                    last_tag = search_stack.pop()
                    if last_tag == '</search>':
                        search_stack.append(last_tag)
                        search_stack.append("</search>")
                if state == 'in_search':
                    state = 'after_search'
                    tag_with_correct_order += 1
            elif part == "<information>" and state == 'after_search':
                state = 'in_information'
            elif part == "</information>" and state == 'in_information':
                state = 'information'
            elif part == "<answer>":
                tag_all += 1
                tag_set.add('<answer>')
                answer_stack.append(part)
                if state == 'after_think':
                    state = 'in_answer'
                    tag_with_correct_order += 1
            elif part == "</answer>":
                tag_all += 1
                tag_set.add('</answer>')
                if len(answer_stack) > 0:
                    last_tag = answer_stack.pop()
                    if last_tag == '</answer>':
                        answer_stack.append(last_tag)
                        answer_stack.append("</answer>")
                if state == 'in_answer':
                    state = 'end'
                    tag_with_correct_order += 1
            elif part == "<action>":
                tag_all += 1
                # action_tag_set.add('<action>')
                action_stack.append(part)
                if state == 'in_think':
                    state = 'in_action'
                    tag_with_correct_order += 1
            elif part == "</action>":
                tag_all += 1
                # action_tag_set.add('</action>')
                if len(action_stack) > 0:
                    last_tag = action_stack.pop()
                    if last_tag == '</action>':
                        action_stack.append(last_tag)
                        action_stack.append("</action>")
                if state == 'in_action':
                    state = 'after_action'
                    tag_with_correct_order += 1
        else:
            if state in ['in_think', 'in_search', 'in_answer', 'in_action', 'after_action']:
                tag_length += len(part)
                content_length += len(part)
            elif state in ['in_information']:
                content_length += 0
            else:
                content_length += len(part)

    level_0 = len(tag_set) / 6
    level_1 = 1 - ((len(think_stack) + len(search_stack) + len(answer_stack)) / tag_all) if tag_all > 0 else 0
    level_2 = tag_length / content_length if content_length > 0 else 0
    level_3 = tag_with_correct_order / tag_all if tag_all > 0 else 0

    self_factor = 0.01
    exp_factor = 5
    factor_0 = level_0
    factor_1 = (self_factor + factor_0 ** exp_factor) * level_1
    factor_2 = (self_factor + factor_1 ** exp_factor) * level_2
    factor_3 = (self_factor + factor_2 ** exp_factor) * level_3


    # print(f"level_0: {level_0}, level_1: {level_1}, level_2: {level_2}, level_3: {level_3}")
    # print(f"factor_0: {factor_0}, factor_1: {factor_1}, factor_2: {factor_2}, factor_3: {factor_3}")

    format_score = 0.05 * factor_0 + 0.15 * factor_1 + 0.25 * factor_2 + 0.55 * factor_3
    # # 如果是在 revision 阶段，包含action，且正确闭合，则加 0.1 分
    # extra_score = 0
    # if is_revison and len(action_stack) == 0 and "<action>" in action_tag_set and "</action>" in action_tag_set:
    #     extra_score = 0.1

    return format_score,{
        'level_0': level_0,
        'level_1': level_1,
        'level_2': level_2,
        'level_3': level_3,
        'factor_0': factor_0,
        'factor_1': factor_1,
        'factor_2': factor_2,
        'factor_3': factor_3,
        'format_score': format_score,
    }
                

def extract_solution(solution_str, allow_first_answer=False):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    leatest_answer_tag = 0 if allow_first_answer else 1
    if len(matches) <= leatest_answer_tag:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False

def compute_score_em_custom(solution_str, ground_truth, method='strict', structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1., allow_first_answer=False):

    format_score_factor, format_score_factor_detail = format_rating(solution_str)

    answer = extract_solution(solution_str=solution_str, allow_first_answer=allow_first_answer)

    do_print = random.randint(1, 128) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if allow_first_answer:
            return 0
        else:
            return 0, {'reward_extra_info': format_score_factor_detail}
    else:
        if em_check(answer, ground_truth['target']):
            if do_print:
                print(f"结构化分数 = {structure_format_score} * {format_score_factor} = {structure_format_score * format_score_factor}")
                print(f"最终分数 = {score} + {structure_format_score * format_score_factor} = {score + structure_format_score * format_score_factor}")
            if allow_first_answer:
                return score  + structure_format_score * format_score_factor
            else:
                return score  + structure_format_score * format_score_factor, {'reward_extra_info': format_score_factor_detail}
        else:   
            if do_print:
                print(f"结构化分数 = {structure_format_score} * {format_score_factor} = {structure_format_score * format_score_factor}")
                print(f"最终分数 = {structure_format_score * format_score_factor}")
            if allow_first_answer:
                return structure_format_score * format_score_factor
            else:
                return structure_format_score * format_score_factor, {'reward_extra_info': format_score_factor_detail}


def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1., allow_first_answer=False):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str, allow_first_answer=allow_first_answer)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
            
    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return 0
    else:
        if em_check(answer, ground_truth['target']):
            if is_valid_format:
                return score # 1
            else:
                return score - structure_format_score # 0.8
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return final_format_score # 0.1