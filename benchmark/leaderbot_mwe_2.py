import leaderbot as lb

# Load dataset
data = lb.data.load()

# List models 1, 2, 11, 12, 23, and 24 from ;#\Cref{tab:model-selection}#;
models = [
    lb.models.BradleyTerry(data, k_cov=None),
    lb.models.BradleyTerry(data, k_cov=0),
    lb.models.RaoKupper(data, k_cov=0, k_tie=0),
    lb.models.RaoKupper(data, k_cov=0, k_tie=1),
    lb.models.Davidson(data, k_cov=0, k_tie=0),
    lb.models.Davidson(data, k_cov=0, k_tie=1)]

# Pre-train the models
for model in models: model.train()

# Compare model rankings, generating a bump chart like ;#\Cref{fig:bump_chart}#;
lb.evaluate.compare_ranks(models, rank_range=[1, 60])

# Evaluate model-selection metrics, similar to ;#\Cref{tab:model-selection}#;
mod_metrics = lb.evaluate.model_selection(models, report=True)

# Evaluate models for goodness of fit, similar to ;#\Cref{tab:goodness-fit}#;
gof_metrics = lb.evaluate.goodness_of_fit(models, metric='RMSE', report=True)
