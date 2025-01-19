# Movie Review Sentiment Analysis

## Table of Contents

- [Project Overview](#project-overview)
  - [Project Objectives](#project-objectives)
- [Data Description](#data-description)
  - [Data Source](#data-source)
- [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Results and Performance](#results-and-performance)
- [Testing with Custom Reviews](#testing-with-custom-reviews)
- [Conclusion](#conclusion)


## Project Overview

This project focuses on building a robust text classification system to identify the sentiment (positive or negative) in movie reviews. By leveraging a labeled dataset of IMDB reviews, the ultimate goal is to develop and evaluate various modeling approaches that can accurately predict review polarity.

Key highlights include:
- Performing Exploratory Data Analysis (EDA) to understand class distribution and potential biases.
- Employing multiple machine learning algorithms to classify reviews as positive or negative.
- Evaluating the models on both the provided test dataset and newly written sample reviews.
- Comparing performance metrics (like F1 score) across different models to determine the most effective approach.

The notebook contains end-to-end steps:
1. **Data Loading**  
2. **Data Preprocessing**
3. **EDA and Class Imbalance Analysis**
4. **Model Training and Evaluation** (Logistic Regression, Gradient Boosting, and others)
5. **Performance Comparison Across Models**

This overview serves as a high-level introduction to the project’s scope and structure.

### Project Objectives

1. **Build a Negative Review Detector**  
   - Develop a text classification model that automatically detects negative movie reviews.  
   - Aim for an F1 score of at least **0.85** on the test set.

2. **Leverage the IMDB Dataset**  
   - Work with polarity-labeled IMDB reviews that are split into train and test sets.

3. **Perform Data Analysis & Preprocessing**  
   - Clean and transform the text data to prepare it for modeling.  
   - Conduct Exploratory Data Analysis (EDA) to understand the data distribution and address any class imbalance.

4. **Train Multiple Models**  
   - Implement at least **three** machine learning algorithms (e.g., Logistic Regression, Gradient Boosting, etc.).  
   - Compare their performance using metrics such as accuracy, precision, recall, and F1 score.

5. **Test & Validate**  
   - Evaluate the models on both the official test set and newly written sample reviews.  
   - Investigate any performance gaps or notable differences in model predictions.

6. **Communicate Findings**  
   - Summarize your observations on model performance, data insights, and recommended approaches for improvement.

# Data Description

The IMDB movie reviews are provided in a **tab-separated (TSV) file**: `imdb_reviews.tsv`. Each row corresponds to an individual movie review with several associated fields. Below are the primary columns of interest:

- **review**:  
  The text of the movie review itself. This will serve as the input for classification.

- **pos**:  
  A binary label where `'0'` indicates a negative review and `'1'` indicates a positive review.

- **ds_part**:  
  Indicates whether a given review is part of the `'train'` or `'test'` dataset split.

Additional fields exist but the above three are the most critical for this project. 


### Data Source
The dataset was provided by:
> **Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts (2011).**  
> *Learning Word Vectors for Sentiment Analysis.*  
> The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
[Andrew L. Maas et al. (2011)](https://ai.stanford.edu/~amaas/data/sentiment/)

## Project Structure
**Notebook**: `Machine Learning for Text.ipynb`  
- Contains all the code for loading data, preprocessing, exploratory data analysis, model training, and results evaluation.

**Data**: `imdb_reviews.tsv`  
- IMDB reviews labeled as positive (`1`) or negative (`0`), along with indicators for train vs. test splits.

### Getting Started

1. **Clone or Download the Repository**  
   - Download the project folder containing the notebook (`Machine Learning for Text.ipynb`) and the dataset (`imdb_reviews.tsv`).

2. **Install Required Libraries**  
   - Ensure your Python environment has the necessary libraries (e.g., `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, etc.).
   - For advanced embeddings, consider installing Hugging Face Transformers (`transformers`), though it’s optional.

3. **Open the Notebook**  
   - Launch the notebook in your preferred environment (e.g., Jupyter Notebook or VSCode) to explore and execute the cells.

4. **Adjust Paths (If Needed)**  
   - If the dataset file (`imdb_reviews.tsv`) is located elsewhere, update the path in the data loading cells accordingly.

## Exploratory Data Analysis (EDA)

1. **Review Distribution**  
   - A quick analysis of the **train/test** split and **positive/negative** labels reveals the overall class balance.  
   - Various plots (e.g., bar charts, histograms) highlight the distribution of reviews by label.

2. **Text Length and Vocabulary**  
   - Basic text statistics (word counts, unique words) provide insight into the variety and complexity of the reviews.

3. **Handling Missing Values**  
   - Checked if any missing entries or irrelevant data exist and performed filtering if needed.

4. **Observations**  
   - The dataset appears sufficiently large to train multiple models.  
   - Initial checks suggest that positive and negative reviews are reasonably balanced, avoiding severe skew.

The EDA section in the notebook showcases plots and summary statistics that guide decisions on preprocessing and model selection.

## Model Training

1. **Data Preprocessing**  
   - Tokenization and basic text cleaning (e.g., removing stopwords, lowercasing) to prepare the text for modeling.  
   - Optionally applied TF-IDF or other feature extraction techniques.

2. **Model Implementations**  
   - **Logistic Regression**: A strong baseline model with relatively quick training.  
   - **Gradient Boosting**: Provides a powerful ensemble approach, potentially improving accuracy.  
   - **Other Algorithms**: Additional models (e.g., Support Vector Machines, Random Forests, or even neural networks) were explored to compare performance.

3. **Training Process**  
   - The `train` dataset was used for fitting the models.  
   - Hyperparameter tuning and cross-validation were employed for optimizing performance.

4. **Optional BERT-based Embeddings**  
   - For a subset of the dataset, BERT was tested to capture more nuanced language patterns.  
   - This step was computationally intensive and recommended for smaller data samples if no GPU is available.

All relevant code is included in the notebook, with clear demarcations for each model and preprocessing routine.

## Results and Performance

1. **Evaluation Metrics**  
   - **Accuracy**: Proportion of correctly classified reviews.  
   - **Precision & Recall**: Reflect the balance between identifying true positives and minimizing false positives.  
   - **F1 Score**: Harmonizes precision and recall; our key target metric is **≥ 0.85**.

2. **Model Comparisons**  
   - Presented results in a tabular format for quick reference.  
   - Examined each model’s performance on the test dataset.

3. **Observations**  
   - Logistic Regression achieved near-target F1 scores with relatively straightforward tuning.  
   - Gradient Boosting often yielded improved performance at the cost of longer training times.  
   - BERT (on a small sample) showcased strong language understanding but was computationally expensive.

4. **Key Takeaways**  
   - Simple approaches (e.g., TF-IDF + Logistic Regression) can yield robust results.  
   - Ensemble methods may further improve performance but require careful hyperparameter optimization to meet or exceed the 0.85 F1 threshold.

## Testing with Custom Reviews

To further validate model reliability, a few self-composed reviews were fed through the trained models:

- **Positive Example**: “I absolutely loved the main character’s performance. The script was engaging, and the cinematography was stunning.”  
- **Negative Example**: “The film felt disjointed and lacked a coherent storyline. I wouldn’t recommend it.”

1. **Comparison of Predictions**  
   - Models generally agreed on extreme sentiments.  
   - Occasionally, borderline or sarcastic reviews led to discrepancies.

2. **Insights**  
   - Models demonstrate consistent performance on newly authored reviews.  
   - Subtle language nuances (e.g., sarcasm, irony) can still pose challenges without more sophisticated context modeling.

These additional tests offer practical validation beyond the standard train/test split.

## Conclusion

1. **Goal Achievement**  
   - The project’s primary objective of detecting negative reviews was accomplished.  
   - Multiple models surpassed or approached the **F1 = 0.85** target, demonstrating strong sentiment classification.

2. **Notable Findings**  
   - Class balance was sufficiently close, minimizing the risk of bias toward a particular sentiment.  
   - Simpler methods (Logistic Regression + TF-IDF) provide a robust baseline with minimal tuning requirements.

3. **Potential Improvements**  
   - **Data Augmentation**: Further enhance training data diversity.  
   - **Advanced Embeddings**: Explore BERT or other transformer-based methods at scale, if GPU resources permit.  
   - **Hyperparameter Tuning**: Refine model parameters for additional performance gains.

4. **Next Steps**  
   - Integrate the best-performing model into a production environment or web application.  
   - Expand to multi-class sentiment analysis (e.g., rating-based or emotion-based) if the use case demands finer distinctions.

Overall, this project showcases the end-to-end pipeline of sentiment analysis, from raw text data to deployment-ready classification models.

---



