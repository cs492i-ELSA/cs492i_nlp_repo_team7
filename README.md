# korquad-open-cs492I
KorQuad-Open Baseline Code for KAIST CS492I 2020 Fall

## Original Author
Seonhoon Kim (Naver)

## Train in NSML
```bash
sh run_nsml.sh
```

## Train in Local

```bash
sh run_local.sh
```

## How to Improve?

### Improve how to select the best answer among different contexts.

We deal with a QA task with a single question and multiple contexts (i.e., paragraphs). One of the most important issues in this type of task is in which paragraph to pick the answer span. Current naive implementation is comparing the probability in a prediction and choosing the paragraph with the largest. Implement a strategy to pick the best answer.

See the codes:
- https://github.com/dongkwan-kim/korquad-open-cs492i/blob/master/open_squad_metrics.py#L412
- https://github.com/dongkwan-kim/korquad-open-cs492i/blob/master/open_squad_metrics.py#L639
- https://github.com/dongkwan-kim/korquad-open-cs492i/blob/master/run_squad.py#L358

**NOTE: In the baseline, you can find the huge gap between validation and test performances (f1-score), since the f1-score for the validation is measured per paragraph, and the f1-score for the test is measured per question.**

![val_f1](https://raw.githubusercontent.com/dongkwan-kim/korquad-open-cs492i/master/static/val_f1.png)
![test_f1](https://raw.githubusercontent.com/dongkwan-kim/korquad-open-cs492i/master/static/test_f1.png)

### Improve how to select training samples considering a memory limit.

There are multiple contexts, but since there is a memory limit, we cannot include everything in training. The baseline is using first three samples in sequence. Which training samples should we choose to learn the best representation? 

See the codes:
- https://github.com/dongkwan-kim/korquad-open-cs492i/blob/master/open_squad.py#L594

### Change model and hyper-parameter configurations.

Do as much as you want.
