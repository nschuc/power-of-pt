from typing import List
from torch import nn
import torch
from transformers.modeling_utils import PreTrainedModel


class PromptWrapper(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        prompt_length: int = 10,
        domain_prompt_length: int = 0,
        domains: List[str] = [],
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
    ):
        super().__init__()

        self.prompt_length = prompt_length
        self.model = model

        for p in model.parameters():
            p.requires_grad = False

        self.prompt = nn.Parameter(
            self.initialize_embedding(
                model.get_input_embeddings(),
                prompt_length,
                random_range,
                initialize_from_vocab,
            )
        )

    def initialize_embedding(
        self,
        embedding: nn.Embedding,
        prompt_length: int = 10,
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
    ):
        if initialize_from_vocab:
            indices = torch.randint(0, 5000, (prompt_length,))
            return embedding.weight[indices].clone().detach()
        return torch.FloatTensor(prompt_length, embedding.weight.size(1)).uniform_(
            -random_range, random_range
        )

    def build_inputs(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        prompt_length = self.prompt_length

        if prompt_length and attention_mask is not None:
            padding = torch.full((batch_size, prompt_length), 1).to(device)
            attention_mask = torch.cat((padding, attention_mask), dim=1)

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if prompt_length:
            prompt = self.prompt.repeat(batch_size, 1, 1)
            inputs_embeds = torch.cat([prompt, inputs_embeds], 1)

        return inputs_embeds, attention_mask, labels

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        inputs_embeds, attention_mask, labels = self.build_inputs(
            input_ids,
            attention_mask,
            labels,
        )

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        inputs_embeds, attention_mask, _ = self.build_inputs(
            input_ids,
            attention_mask,
            labels=None,
        )

        model_kwargs = {
            "encoder_outputs": self.model.get_encoder()(inputs_embeds=inputs_embeds)
        }

        return self.model.generate(
            input_ids=None,
            use_cache=True,
            no_repeat_ngram_size=0,
            **model_kwargs,
            **kwargs,
        )

    @property
    def config(self):
        return self.model.config