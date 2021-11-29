import os
import re
import json
from tqdm import tqdm
from zipfile import ZipFile

from .conala import Conala
from .dataset import Dataset


class CodeSearchNet(Conala):
    def __init__(self, name, split, tokenizer, args, monolingual=False):
        self.threshold = {
            'train': 100,
            'dev': 100,
            'test': 100
        }
        Dataset.__init__(self, 'csn', split, tokenizer, args, monolingual)

    def _process_datafile(self, filename, filter_fn=None):
        ix = 0
        process_ix = 0
        samples = []
        filename = os.path.join(self.dir_name, filename)
        os.system(f'gzip -d {filename}.gz')
        with open(filename, 'r') as f:
            json_lines = f.readlines()
        for json_line in tqdm(json_lines, desc='Process samples.', leave=False):
            line = json.loads(json_line)
            code = line['code']
            code = re.sub(re.compile("'''.*?'''", re.DOTALL) , '', code)
            code = re.sub(re.compile('""".*?"""', re.DOTALL) , '', code)
            code = re.sub(r'(?m)^ *#.*\n?', '', code) # inline remove comments
            docstring = line['docstring'].split('\n')[0]
            sample = {
                'intent': docstring,
                'rewritten_intent': docstring,
                'snippet': code,
                'question_id': ix
            }
            ix += 1
            if filter_fn is None or filter_fn(sample):
                samples.append(sample)
                process_ix += 1
        return samples, ix, process_ix

    def _download_dataset(self):
        # download data
        self._download_file('https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip',
                            'python.zip')
        if not os.path.exists(os.path.join(self.dir_name, 'python')):
            os.system(f'unzip {os.path.join(self.dir_name, "python.zip")} -d {self.dir_name}')

        # define helper filter function
        def filter_fn(sample):
            exclude_list = ['if', 'while', 'for', 'with', '.com']
            process = True
            for exclude_word in exclude_list:
                if exclude_word in sample['snippet']:
                    process = False
            return process

        # process training samples
        samples = []
        if not os.path.exists(os.path.join(self.dir_name, 'csn-corpus/csn-train.json')):
            ix = 0
            processed_ix = 0

            for i in tqdm(range(14), desc='Process data files.'):
                filename = f'python/final/jsonl/train/python_train_{i}.jsonl'
                samples_in_file, ix_in_file, processed_in_file = self._process_datafile(filename, filter_fn)
                ix += ix_in_file
                processed_ix += processed_in_file
                samples += samples_in_file

            print(f'Processed {processed_ix} / {ix} training samples.')

            if not os.path.exists(os.path.join(self.dir_name, 'csn-corpus')):
                os.makedirs(os.path.join(self.dir_name, 'csn-corpus'))
            with open(os.path.join(self.dir_name, 'csn-corpus/csn-train.json'), 'w') as f:
                json.dump(samples, f)

        # process test samples
        samples = []
        if not os.path.exists(os.path.join(self.dir_name, 'csn-corpus/csn-test.json')):
            filename = 'python/final/jsonl/test/python_test_0.jsonl'
            samples, ix, processed_ix = self._process_datafile(filename, filter_fn)
            print(f'Processed {processed_ix} / {ix} testing samples.')
            with open(os.path.join(self.dir_name, 'csn-corpus/csn-test.json'), 'w') as f:
                json.dump(samples, f)


class CSNAugmentedConala(CodeSearchNet):
    def __init__(self, name, split, tokenizer, args, monolingual=False):
        self.threshold = {
            'train': 100,
            'dev': 100,
            'test': 100
        }
        Dataset.__init__(self, 'augcsn', split, tokenizer, args, monolingual)

    def _download_dataset(self):
        prev_self_name = self.name
        self.name = 'csn'; self.dir_name = os.path.join('data', self.name)
        CodeSearchNet._download_dataset(self)
        self.name = 'conala'; self.dir_name = os.path.join('data', self.name)
        Conala._download_dataset(self)
        self.name = prev_self_name; self.dir_name = os.path.join('data', self.name)

        # build training set
        output_train_path = os.path.join(self.dir_name, f'{self.name}-corpus/{self.name}-train.json')
        if not os.path.exists(output_train_path):
            os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
            csn_train_path = os.path.join('data', 'csn/csn-corpus/csn-train.json')
            conala_train_path = os.path.join('data', 'conala/conala-corpus/conala-train.json')
            with open(csn_train_path, 'r') as f:
                csn_train_json = json.load(f)
            with open(conala_train_path, 'r') as f:
                conala_train_json = json.load(f)
            aug_train_json = conala_train_json + csn_train_json
            with open(output_train_path, 'w') as f:
                json.dump(aug_train_json, f)

        # copy test set from conala
        output_test_path = os.path.join(self.dir_name, f'{self.name}-corpus/{self.name}-test.json')
        if not os.path.exists(output_test_path):
            os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
            conala_test_path = os.path.join('data', 'conala/conala-corpus/conala-test.json')
            os.system(f'cp {conala_test_path} {output_test_path}') 
