    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at first <search> or <answer> operation and keep everything up to and including the tag."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        processed_strs = []
        for resp in responses_str:
            matches = re.finditer(r'<(search|answer)>(.*?)</\1>', resp, re.DOTALL)
            matches = list(matches)
            if len(matches) > 0:
                # end_idx = match.end()
                # processed_str = resp[:end_idx]  # Keep everything up to and including the matched tag
                # 如果是从左向右第一个是Search，直接截断
                if matches[0].group(1) == 'search':
                    processed_str = resp[:matches[0].end()]
                else:
                # 如果是Answer，先截判断里面是不是 and , 和 空 
                    content_in_answer = matches[0].group(2).strip().lower()
                    # 异常答案
                    if content_in_answer in ['and', '', ',']:
                        have_search = False
                        # 如果后面还有Search，就按 Search 截断
                        if len(matches) > 1:
                            for index in range(1, len(matches)):
                                if matches[index].group(1) == 'search':
                                    have_search = true
                                    processed_str = resp[:matches[index].end()]
                                    break
                        # 否则就按全部的来，后面再把将其认定为invalid action
                        if not have_search:
                            processed_str = resp
                    # 正常答案
                    else:
                        processed_str = resp[:matches[0].end()]
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
                # match = re.search(pattern, prediction, re.DOTALL)
                matches = re.finditer(pattern, prediction, re.DOTALL)
                matches = list(matches)
                if len(matches) == 1:
                    if matches[0].group(1) == 'search':
                        action = matches[0].group(1)
                        content = matches[0].group(2).strip()
                        complete_prediction = matches[0].group(0)
                    else:
                        content_in_answer = matches[0].group(2).strip().lower()
                        if content_in_answer in ['and', '', ',']:
                            content = ''
                            action = None
                            complete_prediction = None
                        else:
                            action = matches[0].group(1)
                            content = matches[0].group(2).strip()
                            complete_prediction = matches[0].group(0)
                elif len(matches) > 1:
                    # 直接判断最后一个 action 是否为 Search
                    if matches[-1].group(1) == 'search':
                        action = matches[-1].group(1)
                        content = matches[-1].group(2).strip()
                        complete_prediction = matches[-1].group(0)
                    else:
                        content = ''
                        action = None
                        complete_prediction = None
                else:
                    content = ''
                    action = None
                    complete_prediction = None
                # if match:
                #     content = match.group(2).strip()  # Return only the content inside the tags
                #     action = match.group(1)
                #     complete_prediction = match.group(0)
                # else:
                #     content = ''
                #     action = None
                #     complete_prediction = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            complete_predictions.append(complete_prediction)
            
        return actions, contents, complete_predictions