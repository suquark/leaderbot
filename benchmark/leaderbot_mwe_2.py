import leaderbot as lb

# Load dataset
data = lb.data.load()

# List  models 2, 11, 12, 23, and 24 from ;# \Cref{tab:model-selection} #;
models = [
    lb.models.BradleyTerryFactor(data, n_cov_factors=0),
    lb.models.RaoKupperFactor(data, n_cov_factors=0, n_tie_factors=0),
    lb.models.RaoKupperFactor(data, n_cov_factors=0, n_tie_factors=1),
    lb.models.DavidsonFactor(data, n_cov_factors=0, n_tie_factors=0),
    lb.models.DavidsonFactor(data, n_cov_factors=0, n_tie_factors=1)
]

# Pre-train the models
for model in models: model.train()

# Compare model rankings, generating a bump chart like ;# \Cref{fig:bump_chart} #;
lb.evaluate.compare_ranks(models, rank_range=[1, 60])

# Evaluate model-selection metrics, similar to ;# \Cref{tab:model-selection} #;
mod_metrics = lb.evaluate.model_selection(models, report=True)

# Evaluate models for goodness of fit, similar to ;# \Cref{tab:goodness-fit} #;
gof_metrics = lb.evaluate.goodness_of_fit(models, metric='RMSE', report=True)
