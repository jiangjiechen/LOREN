# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/13
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/11/11
"""

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Generator:
    """
    Examples:
        import json
        import torch

        input_data = [
            {'context': 'My name is Sarah.', 'answer': ('Sarah', 11, 16)},
            {'context': 'My name is Sarah and I live in London.', 'answer': ('London', 31, 37)},
            {'context': 'Sarah lived in London. Jone lived in Canada.', 'answer': ('Canada', 37, 43)},
            {'context': 'Sarah lived in London. Jone lived in Canada.', 'answer': ('lived', 28, 33)},
        ]
        generator = Generator(
            'valhalla/t5-base-qg-hl',
            'your_cache_dir',
            'cuda' if torch.cuda.is_available() else 'cpu',
        )

        results = generator(input_data, beam_size=5)
        print(json.dumps(results, ensure_ascii=False, indent=4))
    """

    def __init__(self, model_name_or_path, cache_dir=None, device=DEFAULT_DEVICE, verbose=True):
        self.seed = 1111
        self.device = device
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
        )
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)
        self.model.to(device)
        # if n_gpu > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

    def __call__(self, input_data, beam_size=1, max_length=100, batch_size=8):
        all_ids_with_beam = []
        num_batches = (len(input_data) + batch_size - 1) // batch_size
        iter = tqdm(range(num_batches), desc='Generate questions') if self.verbose else range(num_batches)
        for step in iter:
            batch_start = step * batch_size
            batch_end = min((step + 1) * batch_size, len(input_data))

            batch_text = []
            for entry in input_data[batch_start:batch_end]:
                context = entry['context']
                answer, answer_start, answer_end = entry['answer']
                context = 'generate question: ' + context[:answer_start] + \
                          '<hl> ' + answer + ' <hl>' + context[answer_end:] + ' </s>'
                batch_text.append(context)

            inputs = self.tokenizer.batch_encode_plus(
                batch_text,
                padding='max_length',
                truncation='longest_first',
                max_length=max_length,
                return_tensors='pt',
            )

            for key, value in inputs.items():
                inputs[key] = value.to(self.device)

            ids_with_beam = self.model.generate(num_beams=beam_size,
                                                num_return_sequences=beam_size,
                                                no_repeat_ngram_size=3,
                                                early_stopping=True,
                                                length_penalty=1.5,
                                                repetition_penalty=1.5,
                                                min_length=3,
                                                **inputs)
            ids_with_beam = ids_with_beam.reshape([len(batch_text), beam_size, -1])
            all_ids_with_beam.extend(ids_with_beam.detach().cpu().tolist())

        for i, ids_with_beam in enumerate(all_ids_with_beam):
            input_data[i]['questions'] = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in ids_with_beam]

        return input_data


if __name__ == '__main__':
    import json

    input_data = [
        {'context': 'My name is Sarah.', 'answer': ('Sarah', 11, 16)},
        {'context': 'My name is Sarah and I live in London.', 'answer': ('London', 31, 37)},
        {'context': 'Sarah lived in London. Jone lived in Canada.', 'answer': ('Canada', 37, 43)},
        {'context': 'Sarah lived in London. Jone lived in Canada.', 'answer': ('lived', 28, 33)},
    ]

    generator = Generator(
        'valhalla/t5-base-qg-hl',
        'cache/',
        'cuda' if torch.cuda.is_available() else 'cpu',
    )

    results = generator(input_data, beam_size=5)

    print(json.dumps(results, ensure_ascii=False, indent=4))