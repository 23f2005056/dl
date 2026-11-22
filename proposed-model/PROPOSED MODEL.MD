# Proposed Model: Stock Movement Prediction Using Tweets and Historical Prices

## 1. Project Overview

Our project aims to reproduce and extend the research presented in the 2018 ACL paper:

**Stock Movement Prediction from Tweets and Historical Prices**  
Yumo Xu and Shay B. Cohen (2018)

The objective is to predict whether a stock's price will move upward or downward by combining two sources of information:

1. **Social media information:** Tweets discussing a particular stock.
2. **Historical market information:** The stock's historical price movements.

The central idea is that stock prices are influenced by both market behaviour and public sentiment. Analysing these sources together may provide more useful information than analysing either source independently.

We will preserve the original StockNet dataset, stock universe, prediction period, and evaluation protocol wherever possible.

## 2. Proposed Model Architecture

Our experimental architecture is based on the original StockNet framework.

It consists of three major components:

- **MIE – Market Information Encoder:** Extracts useful information from tweets and historical market data.
- **VMD – Variational Movement Decoder:** Learns representations of stock movements and predicts the direction of future movement.
- **ATA – Auxiliary Temporal Attention:** Identifies which historical days are more relevant to the prediction.

The architecture combines social media information and numerical market information to produce a binary stock movement prediction.

### Overall architecture

```text
                 INPUT DATA
                     |
          +----------+----------+
          |                     |
     TWEET DATA             PRICE DATA
          |                     |
          v                     v
    Tweet Encoder         Price Encoder
    (BiGRU + Attention)   (GRU / TCN)
          |                     |
          +----------+----------+
                     |
                     v
              MIE REPRESENTATION
                     |
                     v
             VMD DECODER
          (Latent Representation
             + Prediction)
                     |
             +-------+-------+
             |               |
             v               v
        Movement          Auxiliary
        Prediction        Temporal
                          Attention
             |               |
             +-------+-------+
                     |
                     v
             FINAL PREDICTION
                UP / DOWN
```

*Note: This is a conceptual overview. The exact connections and tensor operations will follow the original implementation and the team's agreed modifications.*

## 3. Component 1: Market Information Encoder (MIE)

The Market Information Encoder extracts information from tweets and historical prices.

It has two information-processing branches.

### A. Tweet Encoder

The tweet branch processes text associated with a particular stock.

**Processing steps:**

1. Collect tweets associated with each stock.
2. Convert words into numerical vectors using pretrained Twitter GloVe embeddings.
3. Process the word sequences using a Bidirectional GRU (BiGRU).
4. Use the stock's position in the tweet to obtain relevant contextual information.
5. Apply attention to combine information from multiple tweets.

**Why BiGRU?**

A Bidirectional GRU processes text in both forward and backward directions. This helps the model understand a word using both the preceding and following words.

For example:

- "The company reported strong earnings."
- "The company reported weak earnings."

The model can learn that these statements convey different information about the stock.

### B. Price Encoder

The price branch processes historical numerical market information.

The original StockNet implementation uses a GRU-based market representation.

Our project investigates alternative price representations:

- **GRU:** Learns sequential patterns in historical price information.
- **TCN:** Uses temporal convolutional networks to learn patterns across historical time steps.

A GRU processes information sequentially, whereas a TCN uses causal and dilated convolutions to capture patterns across different time intervals.

The output of the price encoder is combined with the tweet representation as part of the market information representation.

## 4. Component 2: Variational Movement Decoder (VMD)

The VMD component learns a probabilistic representation of stock movement.

Instead of directly mapping the input to a prediction, it learns a latent representation.

A latent representation is a compact numerical summary of information that the model has extracted.

The VMD uses:

- Prior distribution
- Posterior distribution
- Reparameterization
- KL divergence
- Movement classification

### How it works

**1. Prior distribution**

The prior represents the model's expected latent movement information based on available input information.

**2. Posterior distribution**

During training, the posterior can additionally use the actual movement label.

**3. Reparameterization**

The model samples a latent representation while allowing gradients to flow during training.

**4. KL divergence**

KL divergence measures the difference between the prior and posterior distributions. It helps regularise the learned latent representation.

**5. Classification**

The resulting representation is used to predict the stock's movement direction.

The output is a two-class prediction:

- Class 0: Downward movement
- Class 1: Upward movement

The original implementation uses the prior for inference rather than relying on the actual target label.

## 5. Component 3: Auxiliary Temporal Attention (ATA)

ATA helps the model identify which historical days are important for predicting the target day's movement.

Not every historical day is equally informative.

For example, a major earnings announcement or an unusual market movement may be more relevant than an ordinary trading day.

ATA assigns attention weights to historical information.

These weights help the model focus on more informative historical periods.

The original configuration uses an auxiliary loss controlled by:

**Alpha = 0.5**

We intend to preserve the original setting in the baseline reproduction.

## 6. Our Proposed Experiments

We will conduct three primary experiments.

| Experiment | Price representation | Purpose |
|---|---|---|
| E0 | Original StockNet price encoder | Reproduce the original model |
| E1 | GRU price encoder | Evaluate the GRU-based price representation |
| E2 | TCN price encoder | Evaluate the TCN-based price representation |

### E0 – Original StockNet Reproduction

We first reproduce the original StockNet architecture using its original experimental configuration.

This provides a baseline against which our subsequent experiments can be compared.

We will preserve:

- Original 88-stock universe.
- Original historical dataset.
- Original training, validation and test periods.
- Original tweet-processing approach.
- Original movement-label definition.
- Original evaluation metrics.

### E1 – GRU Price Representation

This experiment investigates the use of a GRU-based price representation within our implementation.

The objective is to evaluate how effectively sequential market information can be captured and combined with tweet information.

### E2 – TCN Price Representation

This experiment investigates replacing the price representation with a Temporal Convolutional Network.

The objective is to determine whether temporal convolution can capture useful historical price patterns and provide competitive stock movement predictions.

**Important:** E1 and E2 are alternative price-encoder experiments, not an assumption that GRU and TCN will be combined into one encoder.

The tweet-processing, VMD and ATA components should remain consistent across experiments wherever technically feasible.

## 7. Dataset and Experimental Setup

We use the publicly available StockNet dataset.

**Dataset characteristics:**

- 88 US stocks.
- Historical price information.
- Stock-related Twitter messages.
- Research period: January 2014 – January 2016.

### Data splits

| Dataset | Start date | End date (exclusive) |
|---|---|---|
| Training | 1 January 2014 | 1 August 2015 |
| Validation | 1 August 2015 | 1 October 2015 |
| Testing | 1 October 2015 | 1 January 2016 |

The original configuration uses a five-calendar-day historical window.

The original implementation's date alignment and eligibility rules will be preserved rather than silently replacing them with a different trading-day interpretation.

### Price information

The original DataPipe implementation extracts three numerical price fields from the preprocessed price files:

- High
- Low
- Close

The precise transformation of these fields is being verified against the raw dataset before implementation is finalised.

## 8. Evaluation Metrics

We will evaluate the models using the original StockNet evaluation metrics.

### Accuracy

Accuracy measures the percentage of predictions that are correct.

\[
Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}
\]

### Matthews Correlation Coefficient (MCC)

MCC evaluates classification performance using true positives, true negatives, false positives and false negatives.

It is useful because it considers all four outcomes rather than only the overall percentage of correct predictions.

MCC ranges from -1 to +1:

- +1: Perfect predictions.
- 0: No better than random association.
- -1: Completely incorrect predictions.

We will compare E1 and E2 against E0 using the same test protocol.

No improvement will be claimed until the experiments have been completed.

## 9. Team Responsibilities

The work is divided into four modules to allow parallel development.

| Member | Responsibility |
|---|---|
| M1 | MIE, ATA and integration support |
| M2 | Price data processing, Dataset/DataLoader and GRU price encoder |
| M3 | Tweet data processing, TCN price encoder and evaluation |
| M4 | VMD implementation and training infrastructure |

Each module will follow agreed input/output interfaces to minimise integration problems.

## 10. Expected Contribution

The project has two objectives.

**Objective 1: Reproduction**

Reproduce the original StockNet architecture and establish a reliable experimental baseline.

**Objective 2: Extension**

Investigate alternative price-sequence representations, particularly GRU and TCN, while maintaining the broader multimodal learning framework.

The experiments will help us understand whether alternative temporal representations of historical market information can improve stock movement prediction when combined with social media information.

## 11. Important Research Principles

To ensure that our findings are reliable:

1. Establish E0 before making claims about improvements.
2. Preserve the original data splits and avoid test-set tuning.
3. Keep the dataset and target-label definitions consistent across experiments.
4. Avoid introducing additional external data unless explicitly approved as a separate experiment.
5. Document any deviations from the original StockNet implementation.
6. Compare models using identical evaluation metrics and test samples wherever possible.

## Conclusion

Our proposed project builds on the StockNet multimodal stock movement prediction framework.

The model combines information from tweets and historical stock prices through a Market Information Encoder, learns probabilistic movement representations through a Variational Movement Decoder, and uses Auxiliary Temporal Attention to identify relevant historical information.

Our main experimental extension is to investigate alternative price encoders, particularly GRU and TCN, while retaining the broader StockNet framework.

The final objective is to establish a reproducible baseline and determine whether these alternative price representations offer measurable improvements in stock movement prediction.