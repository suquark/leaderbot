import leaderbot as lb

# Load dataset
data = lb.data.load()

# Split data to training and test data
training_data, test_data = lb.data.split(data, test_ratio=0.1, seed=20)

# List  models 2, 11, 12, 23, and 24 from ;# \Cref{tab:model-selection} #;
models = [
    lb.models.BradleyTerryFactor(training_data, n_cov_factors=0),
    lb.models.RaoKupperFactor(training_data, n_cov_factors=0, n_tie_factors=0),
    lb.models.RaoKupperFactor(training_data, n_cov_factors=0, n_tie_factors=1),
    lb.models.DavidsonFactor(training_data, n_cov_factors=0, n_tie_factors=0),
    lb.models.DavidsonFactor(training_data, n_cov_factors=0, n_tie_factors=1)
]

# Evaluate models for generalization on test data, similar to ;# \Cref{tab:generalization} #;
gen_metrics = lb.evaluate.generalization(models, test_data=test_data,
                                                train=True, metric='RMSE', report=True)