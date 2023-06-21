# Revisiting the impact of sex imbalances on biased deep learning models in medical imaging

This repository contains the source code to reproduce the figures presented in the paper _"Revisiting the impact of sex imbalances on biased deep learning models in medical imaging"_.

## Instructions to reproduce results

Here, we describe how to reproduce the figures presented in the paper.

0. **Run `threshold_calc.ipynb`**. 

1. **Run `fairness_calculation.ipynb`**. This notebook will create two tables, **`final_bias_favor_data.csv`** and **`final_overall_bias_favor_data.csv`**, and save them in the **`tables`** folder. The **`final_overall_bias_favor_data.csv`** is **Table 1** from the paper.

3. **Run `fairness_correction.ipynb`**. This notebook will create the five tables listed below and save them in the **`tables`** folder. The table, **`train100%_female_images.csv`**, can contains the results to quickly reproduce **Table 2** from the paper.
    1. **`train100%_female_images.csv`**
    2. **`train75%_female_images.csv`**
    3. **`train50%_female_images.csv`**
    4. **`train25%_female_images.csv`**
    5. **`train0%_female_images.csv`**

4. **Run `generate_plots.ipynb`**. This notebook will create the figures **`final_all_conditions_graph.pdf`** and **`final_sub_conditions_graph.pdf`**. The **`final_sub_conditions_graph.pdf`** figure is **Fig. 2** from the paper. 
