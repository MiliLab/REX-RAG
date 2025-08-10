# from turtle import bye
from configparser import NoOptionError
from pexpect import __revision__
from sympy.sets.sets import true
import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from typing import Union
import random

from .thinking_prompts import revision_prompts, action_prompts, meta_thinking_prompt, action_invalid_prompt

# thinking_prompt = """<decision> Does my previous reasoning lead to the correct answer? I will reflect on the following aspects: \
# 1. [Knowledge Assessment] Do I already know enough to solve this problem confidently, or would retrieving external information (e.g., from Wikipedia or search) significantly reduce uncertainty?\
# 2. [Reasoning Soundness] Is there any possibility my reasoning contained a flaw (e.g., logical leap)?\
# 3. [Conflict Awareness] Could my internal knowledge be conflicting with retrieved content, or did I ignore possible contradictions?\
# Based on my self-assessment, I will decide which cognitive step to take next:\
# - [Retrieve]: retrieve new information.\
# - [Reflect]: rethink my reasoning process or assumptions.\
# - [Conflict-check]: explicitly check for contradictions between information sources.\
# - [None]: if no additional action is needed and can directly answer the question.\
# I will first return an action in this format: <action> [selected action] </action>, where [selected action] represents one action selected from [Retrieve], [Reflect], [Conflict-check], and [None]. Then, based on the action I selected, I will continue with my <think>, <search>, or <answer> operations.\
# </decision>"""

# thinking_prompts = {
#     # 0: "Does my previous reasoning lead to the correct answer? I will assess whether retrieving external information would significantly reduce uncertainty. When I judge that outside sources could clarify the issue, I quietly gather those sources first, then weave them into my ongoing reasoning.",
#     # 1: "Does my previous reasoning lead to the correct answer? I will inspect my chain of thought for hidden assumptions, logical gaps, or leaps. If I spot weaknesses, I willpause to rethink and refine my argument before moving forward.",
#     # 2: "Does my previous reasoning lead to the correct answer? I will cross-check my internal knowledge against any information I have already gathered to detect contradictions. If discrepancies emerge,I will reconcile them to maintain a coherent line of thought.",
#     # 3: "Does my previous reasoning lead to the correct answer? Having confirmed that my reasoning is solid and complete, I proceed naturally to articulate the answer without introducing extra steps."
#     0 : """<decision> Does my previous reasoning lead to the correct answer? I will reflect on the following aspects: \
# 1. [Knowledge Assessment] Do I already know enough to solve this problem confidently, or would retrieving external information (e.g., from Wikipedia or search) significantly reduce uncertainty?\
# 2. [Reasoning Soundness] Is there any possibility my reasoning contained a flaw (e.g., logical leap)?\
# 3. [Conflict Awareness] Could my internal knowledge be conflicting with retrieved content, or did I ignore possible contradictions?\
# Based on my self-assessment, I will decide which cognitive step to take next:\
# - [Retrieve]: retrieve new information.\
# - [Reflect]: rethink my reasoning process or assumptions.\
# - [Conflict-check]: explicitly check for contradictions between information sources.\
# - [None]: if no additional action is needed and can directly answer the question.\
# I will first return an action in this format: <action> [selected action] </action>, where [selected action] represents one action selected from [Retrieve], [Reflect], [Conflict-check], and [None]. Then, based on the action I selected, I will continue with my <think>, <search>, or <answer> operations.\
# </decision>"""
# }




@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    # max_revision_times: int = float('inf')

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        revision_prob = -1,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.revision_prob = revision_prob

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at first <search> or <answer> operation and keep everything up to and including the tag."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        processed_strs = []
        for resp in responses_str:
            match = re.search(r'<(search|answer)>.*?</\1>', resp, re.DOTALL)
            if match:
                end_idx = match.end()
                processed_str = resp[:end_idx]  # Keep everything up to and including the matched tag
            else:
                processed_str = resp  # If no tag found, keep the original response
            processed_strs.append(processed_str)

        if self.config.no_think_rl:
            raise ValueError('stop')
            # If no_think_rl is enabled, only keep action in the str
            actions, _, _ = self.env.postprocess_predictions(processed_strs)
            responses_str = [f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)

        responses = self._batch_tokenize(processed_strs)
        return responses, processed_strs

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding    
        # 将输入，回复，observation 拼接起来，然后将 pad 的内容放在序列最前面
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        # 将非 pad 的 token 设置注意力为 1，pad 的 token 设置注意力为 0
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        # attention 为 1 的 token， position ids varies from 1 to len(attention)
        # 如果 attention 为 0 的 token， position ids 为 0
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        # 裁剪到有效长度，最大输入长度和当前最大有效长度的最小值
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                info_strs: List[str] = None,
                pad_to_left: bool = True,
                component_mask: torch.Tensor = None
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        # 标记模型的回复，component_mask 为 1
        conponent_mask_tensors = [component_mask, torch.where(response != pad_id, torch.tensor(1, dtype=response.dtype, device=response.device), torch.tensor(pad_id, dtype=response.dtype, device=response.device))]
        '''
                   [xxxx][infoinfo]
        with mask: [xxxx][########|###] 截取对应 info 的内容作为 mask
        '''
        if info is not None:
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            info_prompts_component_mask = torch.full(info.size(),pad_id, dtype=info.dtype, device=info.device)
            # 如果是 revision 和 action prompt，将其 info mask 变回正常内容，鼓励模型学习生成
            for idx, info_str in enumerate(info_strs):
                # invalid action
                if action_invalid_prompt in info_str or info_str in action_invalid_prompt:
                    info_prompts_component_mask[idx] = torch.where(info[idx] != pad_id, torch.tensor(3, dtype=info.dtype, device=info.device), torch.tensor(pad_id, dtype=info.dtype, device=info.device))
                    continue
                # revision prompt
                is_revision = False
                for revision_propmt in list(revision_prompts.values()):
                    if revision_propmt in info_str:
                        info_mask[idx] = info[idx]
                        info_prompts_component_mask[idx] = torch.where(info[idx] != pad_id, torch.tensor(5, dtype=info.dtype, device=info.device), torch.tensor(pad_id, dtype=info.dtype, device=info.device))
                        is_revision = True
                        break

                # for idx_sen1, sen1 in enumerate(list(revision_prompts.values())):
                #     if sen1 in info_str:
                #         for idx_sen2, sen2 in enumerate(list(action_prompts.values())):
                #             if sen2 in info_str:
                #                 info_mask[idx] = info[idx]
                #                 tokens_sent1 = self.tokenizer(sen1,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
                #                 tokens_sent2 = self.tokenizer(sen2,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
                #                 mask1 = torch.tensor([5]*len(tokens_sent1),dtype=info.dtype,device=info.device)
                #                 mask2 = torch.tensor([6]*len(tokens_sent2),dtype=info.dtype,device=info.device)
                #                 mask_pad = torch.tensor([pad_id]*(len(info[idx])-len(mask1)-len(mask2)),dtype=info.dtype,device=info.device)
                #                 info_prompts_component_mask[idx] = torch.concat([mask1, mask2, mask_pad]).to(info.device).to(info.dtype)
                #                 is_revision = True
                #                 break
                #         break

                # for prompt in list(revision_prompts.values()) + list(action_prompts.values()):
                #     if prompt in info_str:
                #         info_mask[idx] = info[idx]
                #         # TODO: 这里需要区分 revision 和 action 的 component_mask
                #         breakpoint()
                #         info_prompts_component_mask[idx] = torch.where(info[idx] != pad_id, torch.tensor(2, dtype=info.dtype, device=info.device), torch.tensor(pad_id, dtype=info.dtype, device=info.device))
                #         is_revision = True
                #         break
                if is_revision:
                    continue
                # info from search
                info_prompts_component_mask[idx] = torch.where(info[idx] != pad_id, torch.tensor(4, dtype=info.dtype, device=info.device), torch.tensor(pad_id, dtype=info.dtype, device=info.device))
            tensors.append(info)
            tensors_with_mask.append(info_mask)
            conponent_mask_tensors.append(info_prompts_component_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        # 这里是根据concatenated 的内容进行排序的
        # info mask 里的 pad 不会对排序产生影响
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        concatenated_component_mask = torch.cat(conponent_mask_tensors, dim=1)
        padded_component_mask = concatenated_component_mask.gather(1, sorted_indices)


        return padded_tensor, padded_tensor_with_info, padded_component_mask

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None, next_obs: List[str] = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            # 将输入，回复，observation 拼接起来，然后将 pad 的内容放在最后面
            # info mask： info 的部分掩码为 pad token
            responses, responses_with_info_mask, component_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    next_obs,
                    pad_to_left=False,
                    component_mask=right_side['component_mask']
                )
        else:
            responses, responses_with_info_mask, component_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False,
                    component_mask=right_side['component_mask']
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len], 'component_mask': component_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
            如果数量可以被num_gpus整除，则直接返回
            如果不能被整除，则需要pad，然后remove padding
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, is_validation: bool = False, steps = None) -> Tuple[Dict, Dict]:
        self.steps = steps
        # TODO: 根据索引克隆数据
        clone_rollout_index = []
        def clone_rollout(rollout_index: Union[int, List[int]]) -> DataProto:
            """
            根据 rollout_index 克隆 rollings（直接复制全部相关的数据）
            """
            nonlocal original_left_side, original_right_side, active_mask, clone_rollout_index, turns_stats, valid_action_stats, valid_search_stats, active_num_list, rollings, meta_info, original_mask, responses_ids, responses_str, gen_batch
            if isinstance(rollout_index, int):
                rollout_index = [rollout_index]
            original_left_side = {'input_ids': torch.cat([original_left_side['input_ids'], original_left_side['input_ids'][rollout_index].detach().clone()], dim=0)}
            original_right_side = {
                'responses': torch.cat([original_right_side['responses'], original_right_side['responses'][rollout_index].detach().clone()], dim=0),
                'responses_with_info_mask': torch.cat([original_right_side['responses_with_info_mask'], original_right_side['responses_with_info_mask'][rollout_index].detach().clone()], dim=0),
                'component_mask': torch.cat([original_right_side['component_mask'], original_right_side['component_mask'][rollout_index].detach().clone()], dim=0)
            }
            active_mask = torch.cat([active_mask, active_mask[rollout_index].detach().clone()], dim=0)
            turns_stats = torch.cat([turns_stats, turns_stats[rollout_index].detach().clone()], dim=0)
            valid_action_stats = torch.cat([valid_action_stats, valid_action_stats[rollout_index].detach().clone()], dim=0)
            valid_search_stats = torch.cat([valid_search_stats, valid_search_stats[rollout_index].detach().clone()], dim=0)
            # revision_stats = torch.cat([revision_stats, revision_stats[rollout_index].detach().clone()], dim=0)
            active_num_list[-1] = active_num_list[-1] + active_mask[rollout_index].sum().item()
            rollings_clone = rollings.clone(rollout_index)
            rollings = DataProto.concat([rollings, rollings_clone])
            gen_batch = DataProto.concat([gen_batch, gen_batch.clone(rollout_index)])
            meta_info = rollings.meta_info
            original_mask = torch.cat([original_mask, torch.zeros(len(rollout_index), dtype=torch.bool)], dim=0)
            # responses_ids = torch.cat([responses_ids, responses_ids[rollout_index].detach().clone()], dim=0)
            # responses_str = responses_str + [responses_str[index] for index in rollout_index]
            clone_rollout_index.extend(rollout_index)

        """Run main LLM generation loop."""
        # gen_batch [batch_size, context_length]
        # 输入内容
        # pad 在左，内容在右
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        # 输出内容，初始为空。
        # component_mask: 0. 原始问题 1. 模型的 response 2. 插入的 revision提示词 3. 插入的 invalid action 提示词 4. 检索器返回的 information
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []], 'component_mask': initial_input_ids[:, []]}
        
        # 记录每个数据是否在仍在推理、生成轮数、有效的动作数量、有效的搜索数量以及全部的正在推理的数据数量
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # revision_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        original_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            # 如果没有数据是正在推理的
            if not active_mask.sum():
                break
            # 从右侧裁剪数据（根据这一轮批次中最长的那一个），裁剪到有效长度
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            # 根据active 掩码，筛选出正在推理的数据
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            # 生成结果，如果数量可以被num_gpus整除，则直接返回
            # 如果不能被整除，则需要pad，然后remove padding
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info        
            # 获取本轮回复的token ids和文本内容（将不同的 action 作为新一轮结束的标志）
            # 如果生成了多个 action，则取第一个 action    
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            # 对每个数据进行pad，pad到和active_mask中为True的token ids长度相同（非 activate 的数据，用pad token 填充，str 为空）
            # 从这个地方，数据的数量再次变为batch_size
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            # 获取本轮的 observation（检索到的内容）、是否结束（本轮是否是 answer 操作）、是否有效动作、是否有效搜索和是否需要进行 revision
            next_obs, dones, valid_action, is_search, responses_ids, responses_str, revision_count = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, responses_ids, gen_batch.non_tensor_batch['reward_model'] if not is_validation else None, gen_batch.non_tensor_batch['data_source'] if not is_validation else None, clone_rollout, original_mask, active_mask, is_validation=is_validation
            )

            # 在这个地方添加上，根据需要 need revison 和已经 revsion 的次数更新 dones
            # dones = (torch.tensor(dones, dtype=torch.bool) | (torch.tensor(need_revision, dtype=torch.int) >= self.config.max_revision_times)).int().tolist()
            
            # 更新 activate 掩码
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            # 记录每轮之后还有多少数据是 activate 的
            active_num_list.append(active_mask.sum().item())
            # 记录每轮之后，每个数据已经推理了多少轮
            turns_stats[curr_active_mask] += 1
            # 记录每轮之后，每个数据是否有效动作、是否有效搜索
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            # revision_stats += torch.tensor(need_revision, dtype=torch.int)

            # 将 observation 转换为 token ids，并裁剪到 config.max_obs_length
            # response 用这个 obs，同时这部分不需要 info mask，当做模型生成的目标
            next_obs_ids = self._process_next_obs(next_obs)

            # 这里需要将这一轮新复制出来的轨迹的 next obs（即 revision & action） 替换为 meta_thinking_prompt，用作模型rolling
            # next_obs_rolligngs = next_obs.copy()
            # if revision_count > 0:
            #     next_obs_rolligngs[-revision_count:] = [meta_thinking_prompt] * revision_count
            # next_obs_ids_rollings = self._process_next_obs(next_obs_rolligngs)
            

            # Update states
            # response_ids 是本轮的回复（模型的输出），next_obs_ids 是本轮的 observation（检索到的内容或者提示词）
            # 这里内容是右对齐的
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                # next_obs_ids_rollings,
                next_obs_ids
            )
            # 这里内容是左对齐的
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids,
                next_obs
            )

            ## rolling是带有提示词信息的，original_right_side不带有提示词信息
            ## 但因为仅revision 一次，rolling 不需要删提示词
            ## original_right_side 也不需要删
        # final LLM rollout
        # 如果还有数据是正在推理的，则进行最终的推理
        # 此时，由于已经达到了最大推理次数，不能再进行搜索
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)


            # # Execute in environment and process observations
            _, dones, valid_action, is_search, responses_ids, responses_str, _ = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, responses_ids, gen_batch.non_tensor_batch['reward_model'] if not is_validation else None, gen_batch.non_tensor_batch['data_source'] if not is_validation else None, clone_rollout, original_mask, active_mask, do_search=False, is_validation=is_validation
            )

            # dones = (torch.tensor(dones, dtype=torch.bool) | (torch.tensor(need_revision, dtype=torch.int) >= self.config.max_revision_times)).int().tolist()

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        # 记录每轮之后，每个数据已经推理了多少轮、是否激活、是否有效动作、是否有效搜索
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info, original_mask), clone_rollout_index
    
    # def _delete_thinking_prompt(self, contents: List[str], response_ids_length) -> str:
    #     """
    #     删除 contents 中的 thinking_prompt，并返回删除后的内容和 token ids
    #     """
    #     deleted_contents = []
    #     for content in contents:
    #         deleted_content = content
    #         for index in range(len(thinking_prompts)):
    #             deleted_content = deleted_content.replace(thinking_prompts[index], "")
    #         deleted_contents.append(deleted_content)
    #     response_ids = self._batch_tokenize(deleted_contents)
    #     # 使用 pad token 填充 response_ids
    #     if response_ids.shape[1] < response_ids_length:
    #         pad_length = response_ids_length - response_ids.shape[1]
    #         pad_tensor = torch.full((response_ids.shape[0], pad_length), self.tokenizer.pad_token_id, dtype=response_ids.dtype, device=response_ids.device)
    #         response_ids = torch.cat([response_ids, pad_tensor], dim=1)
    #     elif response_ids.shape[1] > response_ids_length:
    #         response_ids = response_ids[:, :response_ids_length]

    #     return response_ids, deleted_contents

    # 删除 content 中的 complete_prediction
    def _delete_contents(self, content, complete_prediction, response_ids_length) -> str:
        """
        删除 content 中的 complete_prediction。(仅删一条数据的)

        Args:
            content: str，原始内容
            complete_prediction: 需要删除的字符串
            response_ids_length: 需要填充到的长度
        Returns:
            str: 删除后的内容
        """
        if isinstance(content, str):
            deleted_content = content.replace(complete_prediction, "")
            response_ids = self._batch_tokenize([deleted_content])[0]
            # 使用 pad token 填充 response_ids
            if response_ids.shape[0] < response_ids_length:
                pad_length = response_ids_length - response_ids.shape[0]
                pad_tensor = torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=response_ids.dtype, device=response_ids.device)
                response_ids = torch.cat([response_ids, pad_tensor], dim=0)
            elif response_ids.shape[0] > response_ids_length:
                response_ids = response_ids[:response_ids_length]
            return response_ids, deleted_content
        else:
            deleted_contents = []
            for content_ in content:
                deleted_content = content_.replace(complete_prediction, "")
                deleted_contents.append(deleted_content)
            response_ids = self._batch_tokenize(deleted_contents)
            if response_ids.shape[1] < response_ids_length:
                pad_length = response_ids_length - response_ids.shape[1]
                pad_tensor = torch.full((response_ids.shape[0], pad_length), self.tokenizer.pad_token_id, dtype=response_ids.dtype, device=response_ids.device)
                response_ids = torch.cat([response_ids, pad_tensor], dim=1)
            elif response_ids.shape[1] > response_ids_length:
                response_ids = response_ids[:, :response_ids_length]
            return response_ids, deleted_contents


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict, original_mask: torch.Tensor) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        # 创建 info mask，info 的部分掩码为 pad token，其余部分正常为内容的 token ids
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output['component_mask'] = torch.cat([
            torch.where(left_side['input_ids'] != self.tokenizer.pad_token_id, torch.tensor(0, dtype=final_output['component_mask'].dtype, device=final_output['component_mask'].device), torch.tensor(self.tokenizer.pad_token_id, dtype=final_output['component_mask'].dtype, device=final_output['component_mask'].device)),
            right_side['component_mask']
        ], dim=1)

        final_output['original_mask'] = original_mask
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, response_ids: torch.Tensor, reward_model: List[dict], data_source: List[str], clone_rollout_fn, original_mask, active_mask=None, do_search=True, is_validation=False) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions (str)
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        # 获取本轮的 action 和内容
        cur_actions, contents, complete_predictions = self.postprocess_predictions(predictions)
        assert len(cur_actions) == len(contents) == len(complete_predictions), f"cur_actions: {len(cur_actions)}, contents: {len(contents)}, complete_predictions: {len(complete_predictions)}"
        # 记录本轮的 observation（检索到的内容）、是否结束（本轮是否是 answer 操作）、是否有效动作、是否有效搜索
        next_obs, dones, valid_action, is_search = [], [], [], []
        extended_next_obs, extended_dones, extended_valid_action, extended_is_search = [], [], [], []
        
        # 获取本轮需要的查询内容
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            # 进行查询
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        revision_count = 0

        # 遍历本轮的每个数据
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            # 如果数据不是正在推理的，则直接返回
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    '''
                    基本流程：
                        1、如果当前数据是第一次采样出错，则需要复制当前数据，并进行 revision
                    '''
                    if not is_validation:
                        from verl.trainer.main_ppo import _select_rm_score_fn
                        reward_score_fn = _select_rm_score_fn(data_source[i])
                    else:
                        reward_score_fn = None

                    if not is_validation and \
                        do_search and \
                        reward_score_fn(solution_str=complete_predictions[i], ground_truth=reward_model[i]['ground_truth'],structure_format_score= 0.2, final_format_score=0., retrieval_score=0., format_score=0., score=1., allow_first_answer=True) < 1 and \
                        original_mask[i] and \
                        random.randint(0, 100) < self.revision_prob : # * (self.steps / 500)
                        # print(f'index {i} 出错了,且进行revision')
                        clone_rollout_fn(i)
                        res = self._delete_contents(predictions[i], complete_predictions[i], response_ids.shape[1])
                        response_ids = torch.cat([response_ids, res[0].clone().detach().reshape(1, -1)], dim=0)
                        predictions = predictions + [res[1]]
                        # obs = revision_prompts[random.randint(0, len(revision_prompts) - 1)] + action_prompts[random.randint(0, len(action_prompts) - 1)]
                        obs = revision_prompts[random.randint(0, len(revision_prompts) - 1)]
                        extended_next_obs.append(obs)
                        extended_dones.append(0)
                        extended_valid_action.append(1)
                        extended_is_search.append(0)
                        revision_count += 1

                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)

                       
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                    # need_revision.append(0)
                else:
                    # 直接添加提示词，提示模型输出符合规范
                    next_obs.append(action_invalid_prompt)
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    # need_revision.append(0)
            
        # 因为每次都是pop的，所以最后应该为空
        assert len(search_results) == 0

        next_obs = next_obs + extended_next_obs
        dones = dones + extended_dones
        valid_action = valid_action + extended_valid_action
        is_search = is_search + extended_is_search
        # need_revision = need_revision + extended_need_revision
            
        return next_obs, dones, valid_action, is_search, response_ids, predictions, revision_count

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
        complete_predictions = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                    complete_prediction = match.group(0)
                else:
                    content = ''
                    action = None
                    complete_prediction = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            complete_predictions.append(complete_prediction)
            
        return actions, contents, complete_predictions

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
