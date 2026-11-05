import torch

try:
    from fastchat import model as fastchat_model
except ImportError:
    fastchat_model = None

def load_conversation_template(template_name):
    if template_name == 'qwen3':
        return None
    if fastchat_model is None:
        raise ImportError('fschat is required for non-Qwen conversation templates')
    if template_name == 'llama2':
        template_name = 'llama-2'
    conv_template = fastchat_model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
        conv_template.system = "[INST] <<SYS>>\n\n<</SYS>>\n\n"
    return conv_template

class autodan_SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string

    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())

        if self.conv_template is None:
            messages = [{'role': 'user', 'content': self.adv_string}]
            prefix_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full_ids = self.tokenizer.apply_chat_template(
                messages + [{'role': 'assistant', 'content': self.target}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
            target_start = len(prefix_ids)
            if full_ids[:target_start] != prefix_ids:
                raise ValueError('Qwen3 generation prefix is not a prefix of the scored sequence')
            try:
                target_stop = full_ids.index(im_end_id, target_start)
            except ValueError as exc:
                raise ValueError('Qwen3 assistant end token was not found') from exc

            self._assistant_role_slice = slice(0, target_start)
            self._target_slice = slice(target_start, target_stop)
            self._loss_slice = slice(target_start - 1, target_stop - 1)
            self._qwen_input_ids = full_ids[:target_stop]
            return self.tokenizer.decode(self._qwen_input_ids)

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        if self.conv_template.name == 'llama-2':
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))
            self._control_slice = self._goal_slice

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        else:
            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
            try:
                encoding.char_to_token(len(prompt) - 1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.adv_string}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks) - 1))
                self._control_slice = self._goal_slice

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 1)
                self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 2)
            else:
                self._system_slice = slice(
                    None,
                    encoding.char_to_token(len(self.conv_template.system))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.adv_string)),
                    encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
                )
                self._control_slice = self._goal_slice
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                )

        self.conv_template.messages = []

        return prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        if self.conv_template is None:
            return torch.tensor(self._qwen_input_ids, dtype=torch.long)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids
