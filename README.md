# Directory structure
```
├── LICENSE
├── README.md                       <- The top-level README for developers using this project.
├── data
│   ├── external                    <- Data from third party sources.
│   ├── interim                     <- Intermediate data that has been transformed.
│   ├── processed                   <- The final, canonical data sets for modeling.
│   └── raw                         <- The original, immutable data dump.
├── docs                            <- A default Sphinx project; see sphinx-doc.org for details
├── models                          <- Trained and serialized models, model predictions, or model summaries
├── notebooks                       <- Jupyter notebooks. Naming convention is a number (for ordering),
│                                      the creator's initials, and a short `-` delimited description, e.g.
│                                      `1.0-jqp-initial-data-exploration`.
├── poetry.lock
├── pyproject.toml                  <- Will orchestrate your project and its dependencies
├── references                      <- Data dictionaries, manuals, and all other explanatory materials.
├── reports                         <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                     <- Generated graphics and figures to be used in reporting
└── src                             <- Source code for use in this project.
    ├── data                        <- Scripts to download or generate data
    │   └── __init__.py
    ├── dataWrangling               <- Cleaning and organizing data for analysis
    │   └── __init__.py
    ├── eda                         <- Exploring and summarizing data to gain insights
    │   └── __init__.py
    ├── etl                         <- Extracting, transforming, and loading data between systems.          
    │   └── __init__.py
    ├── models                      <- Scripts to train models and then use trained models to make predictions
    │   └── __init__.py
    ├── utils                       <- Miscellaneous code goes here.
    │   └── __init__.py
    └── viz                         <- Graphics and visualization development specific work should go here
        └── __init__.py
```

