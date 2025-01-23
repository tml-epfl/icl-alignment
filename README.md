# Is In-Context Learning (ICL) Sufficient for Instruction Following in LLMs?

<a href="https://marcelluszhao.github.io/">Hao Zhao</a>, <a href="https://www.andriushchenko.me/">Maksym Andriushchenko</a>, <a href="https://scholar.google.com/citations?user=laq9cq0AAAAJ&hl=zh-CN">Francesco Croce</a>, <a href="https://people.epfl.ch/nicolas.flammarion">Nicolas Flammarion</a> (EPFL)

**Paper:** [https://arxiv.org/abs/2405.19874](https://arxiv.org/abs/2405.19874)

**ICLR 2025**

> TL;DR: We uncover that, unlike for tasks such as classification, translation, or summarization, adding more ICL demonstrations for long-context LLMs does not systematically improve instruction following performance, even with more sophisticated approaches. Moreover, we show that
> - While ICL alignment with **URIAL** (ICLR'24: Untuned LLMs with Restyled In-context Alignment) is effective, it still underperformn compared to instruction fine-tuning on established benchmarks such as MT-Bench and AlpacaEval 2.0 (LC), especially with more capable base LMs.
> - For instruction-following tasks, the demonstrations need to be carefully chosen and of high quality, with correct answers to each question, which departs from findings in prior work (<a href="https://arxiv.org/abs/2202.12837">Min et al., 2022</a>).

## Installation

```bash
conda create -n yourenvname python=3.10
conda activate yourenvname
pip install -r requirements.txt
```

## Compare ICL alignment with URIAL to instruction fine-tuning

One can find a series of bash scripts that run ICL alignment on base LLMs under the `run_scripts/mt-bench/` folder. In particular,

- `version`: it refers to the prompt that is used to do ICL. It contains two parts: (a) general rules that the base LLM should follow when generating answers, and (b) a list of instructions with corresponding answers.
- `model_name`: it specifies the base model to use in the experiment.
- `data_name`: it decides the evaluation test data.
- `repetition_penalty`: we use a repetition penalty on base models to prevent degeneration, 1.0 for MoEs and 1.15 for the remaining models.

For example, run URIAL on Mistral-7B-v0.2,
```bash  
bash run_scripts/mt-bench/mistral-7b-v2-urial.sh
```

In order to evaluate on MT-Bench, one needs to run the following command to format the generated results
```bash
python run_scripts/mt-bench/formatting_results.py Mistral-7B-v0.2.urial-vllm
```

Next, proceed to the `FastChat` library to perform the MT-Bench scoring. The AlpacaEval 2.0 benchmark adopts a similar procedure; the only difference is that the generated answers do not need to be formatted.

We improve the alignment performance via a greedy search of proper 4th/5th/6th example and list our searched results in
- `urial_prompts/urial.plus.best_4.examples.txt`
- `urial_prompts/urial.plus.best_4&5.examples.txt`
- `urial_prompts/urial.plus.best_4&5&6.examples.txt`

## Scaling experiments

Under the `urial_prompts` folder, we give prompts with varying numbers of in-context demonstrations, which are employed in the scaling experiments described in our paper.

- `random`: Through randomly sampling from the high-quality dataset multiple times, we generate a series of in-context demonstration sets that each contain $N$ examples, and insert the $N$ demonstrations into the prompt template.
- `greedy`: One could perceive the greedy search results of the proper fourth example as an alternative method of evaluating the data quality. We sort the 100 high-quality examples in descending order based on the score as evaluated by GPT-4-Turbo. Thus it allows us to create another series of in-context demonstration sets by selecting top-$N$ examples.
- `diversity`: We employ stratified sampling over the **source** annotation of OpenHermes-2.5 to collect $N$ diverse ICL examples. We get rid of the URIAL demonstrations in this experiment since their source is unknown.
- `unsupervised`: We reuse instructions from the experiment of scaling up diverse examples and remove corresponding answers. The prompt template for this scaling experiment consists of: (a) general rules that the model should follow when generating answers, (b) multiple instructions without answers, and (c) a few-shot prompt with ground truth answers for the desired output format and the URIAL prompt is used in our work.

## Evaluation 

One can find standardized code for evaluation we did in our paper from:
- <a href="https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge">MT-Bench</a>
- <a href="https://github.com/tatsu-lab/alpaca_eval">AlpacaEval 2.0</a>

## Citation

If you find this useful in your research, please consider citing:

```
@misc{zhao2024incontext,
      title={Is In-Context Learning Sufficient for Instruction Following in LLMs?}, 
      author={Hao Zhao and Maksym Andriushchenko and Francesco Croce and Nicolas Flammarion},
      year={2024},
      eprint={2405.19874},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgements

We thank the following open-sourced repositories. If aspects of these repositories appearing in our work are useful to your research, we ask that you consider citing the accompanying papers.

    [1] https://github.com/Re-Align/URIAL
    [2] https://github.com/tatsu-lab/alpaca_eval
    [3] https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
    [4] https://github.com/vllm-project/vllm
    [5] https://github.com/huggingface/transformers
    [6] https://github.com/tml-epfl/long-is-more-for-alignment
