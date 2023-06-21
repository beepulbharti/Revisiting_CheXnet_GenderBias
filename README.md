# Revisiting_CheXnet_GenderBias

This repository contains the source code to reproduce the figures presented in the paper _"Revisiting the impact of sex imbalances on biased deep learning models in medical imaging"_.

## Instructions to reproduce results

Here, we describe how to reproduce the figures presented in the paper.

0. **Run `threshold_calc.ipynb`**. This will calculate the optimal thresholds

1. **Run `fairness_calculation.ipynb`**. This will calculate the number of bias

2. **Known Boolean function.** The `./experiments/boolean_function.ipynb` notebook contains the code to reproduce the results in Sec. 4.1 of the paper. To reproduce Fig. 1 in the paper, simply run the notebook. Figures should be saved in the `./figures/boolean/` folder.

3. **Synthetic image dataset.** The `./experiments/crosses/` folder contains the code to reproduce the results in Sec. 4.2. of the paper. To reproduce Fig. 2, first run `./experiments/crosses/power_m.py` and `./experiments/crosses/power_sigma.py`, and then run the notebook `./experiments/crosses/figures.ipynb`. Figures should be saved in the `./figures/crosses/` folder.

4. **Real image dataset.** The `./experiments/BBBC041/` folder contains the code to reproduce the results in Sec. 4.3. For ease of reproducibility, `./experiments/BBBC041/pretrained_model` contains the pretrained model used in the paper, `./experiments/BBBC041/train.py` contains the code used to train the model, and the `./experiments/BBBC04/demo/` folder contains the image with the respective explanations used in the paper. To reproduce Fig. 3 using the pretrained model included in this archive, simply run the notebook `./experiments/BBBC041/figures.ipynb`. Figures should be saved in the `./figures/BBBC041/` folder. To reproduce the entire process followed in this experiment:

    0. **Download the data.** We use the preprocessed dataset made available by the authors of _"Fast Hierchical Games for Image Explanations"_ at [https://zenodo.org/record/5914342#.Yo6MF5PMJhE](https://zenodo.org/record/5914342#.Yo6MF5PMJhE). The `./experiments/BBBC041/data/` folder should be structured as follows:

        ```bash
        experiments/BBBC041/
        ├── data
        │   ├── trophozoite
        │   │   ├── train
        │   │   │   ├── 0
        |   |   |   └── 1
        |   |   └── val
        |   |       ├── 0
        |   |       └── 1
        |   ├── test_cropped.json
        |   └── trainig.json
        ├── ...
        ```
    
    1. **Train the model.** Run `./experiments/BBBC041/train.py` to train the model.

    2. **Find true positive predictions.** Run `./experiments/BBBC041/true_positive.py` to find the true positive predictions of the model on the validation set.

    3. **Compute the reference value.** Run `./experiments/BBBC041/compute_reference.py` to compute the reference value used to mask features when explaining predictions with h-Shap.

    4. **Explain predictions.** Run `./experiments/BBBCO41/explain.py` to explain the predictions of the model on the true positive predictions in the validation set.
